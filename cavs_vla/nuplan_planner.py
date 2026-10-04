"""nuplan-devkit planner adapter: run the policy + safety stack in the official nuPlan closed-loop simulator.

Headline closed-loop numbers for the paper should come from the official
simulator (CLS-NR / CLS-R on Val14 / Test14-hard), not only from our log
simulator. Usage with nuplan-devkit (v1.2) — register a Hydra planner config:

    # nuplan/planning/script/config/simulation/planner/cavs_planner.yaml
    cavs_planner:
      _target_: cavs_vla.nuplan_planner.CavsPlanner
      _convert_: all
      checkpoint: /abs/path/checkpoints/cavs_v2/main/best.pt
      config_path: /abs/path/configs/default.yaml
      method: B7

and run nuplan's run_simulation.py with planner=cavs_planner. The simulator's
LQR tracker follows the returned trajectory; when the shield intervenes the
returned trajectory is the rollout of the shielded controls.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Type

import numpy as np
import torch

from .config import METHOD_TABLE, load_config
from .data.scene import (SRC_NUPLAN, TL_GREEN, TL_NONE, TL_RED, TL_YELLOW, WorldPolyline, empty_scene,
                         fill_polylines)
from .dynamics import rollout
from .features import featurize, scene_to_device
from .geometry_np import rotate_vec, to_frame
from .language import instruction_from_route, tokenize
from .safety import conformal
from .safety.hj import HJTable
from .safety.shield import SRC_EMERGENCY, SRC_FALLBACK, SRC_MPC, Shield
from .safety.verifier import certify
from .train import load_policy

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (AbstractPlanner, PlannerInitialization,
                                                                   PlannerInput)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

TYPE_MAP = {TrackedObjectType.VEHICLE: 0, TrackedObjectType.PEDESTRIAN: 1, TrackedObjectType.BICYCLE: 2}
TL_MAP = {TrafficLightStatusType.RED: TL_RED, TrafficLightStatusType.YELLOW: TL_YELLOW,
          TrafficLightStatusType.GREEN: TL_GREEN}


class CavsPlanner(AbstractPlanner):
    requires_scenario: bool = False

    def __init__(self, checkpoint: str, config_path: Optional[str] = None, method: str = "B7") -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, cfg_ck, _ = load_policy(checkpoint, self._device)
        cfg = load_config(config_path) if config_path else cfg_ck
        for sec in ("data", "feat", "model"):
            setattr(cfg, sec, getattr(cfg_ck, sec))
        self._cfg = cfg
        self._method_name = method
        self._method = METHOD_TABLE[method]
        hj = HJTable(cfg.hj.table_path, self._device) if os.path.exists(cfg.hj.table_path) else None
        if hj is None and self._method.get("hj"):
            raise FileNotFoundError(f"HJ table {cfg.hj.table_path} not found")
        self._conf = conformal.load(cfg.conformal.table_path) if os.path.exists(cfg.conformal.table_path) else None
        self._shield = Shield(cfg, hj, self._conf)
        self._a_prev = 0.0
        self._map_api = None
        self._route: set = set()

    def name(self) -> str:
        return f"CavsPlanner_{self._method_name}"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route = {str(r) for r in initialization.route_roadblock_ids}
        self._a_prev = 0.0

    # ------------------------------------------------------------------ scene construction
    def _sample_history(self, current_input: PlannerInput):
        """Pick K states spaced dt apart (newest last) from the 20 Hz history buffer."""
        d = self._cfg.data
        hist = current_input.history
        ego_states = list(hist.ego_states)
        observations = list(hist.observations)
        t_now = ego_states[-1].time_point.time_us
        times = np.array([e.time_point.time_us for e in ego_states])
        picks = []
        for k in range(d.hist_steps - 1, -1, -1):
            target = t_now - int(k * d.dt * 1e6)
            i = int(np.argmin(np.abs(times - target)))
            picks.append(i if abs(times[i] - target) < 0.6 * d.dt * 1e6 else None)
        return ego_states, observations, picks

    def _scene(self, current_input: PlannerInput) -> Dict[str, np.ndarray]:
        cfg = self._cfg
        d = cfg.data
        K = d.hist_steps
        ego_states, observations, picks = self._sample_history(current_input)
        scene = empty_scene(cfg)
        cur = ego_states[-1]
        origin = np.array([cur.rear_axle.x, cur.rear_axle.y])
        yaw0 = cur.rear_axle.heading

        def ego_vec(e: EgoState) -> np.ndarray:
            v = e.dynamic_car_state.rear_axle_velocity_2d
            h = e.rear_axle.heading
            return np.array([e.rear_axle.x, e.rear_axle.y, h, v.x * math.cos(h) - v.y * math.sin(h),
                             v.x * math.sin(h) + v.y * math.cos(h), d.ego_length, d.ego_width])

        def to_scene(s: np.ndarray) -> np.ndarray:
            s = s.copy()
            s[:2] = to_frame(s[None, :2], origin, yaw0)[0]
            s[2] = math.atan2(math.sin(s[2] - yaw0), math.cos(s[2] - yaw0))
            s[3:5] = rotate_vec(s[None, 3:5], yaw0)[0]
            return s

        tracks: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        ttype: Dict[str, int] = {}
        for k, i in enumerate(picks):
            if i is None:
                continue
            scene["traj"][0, k] = to_scene(ego_vec(ego_states[i]))
            scene["valid"][0, k] = True
            for obj in observations[i].tracked_objects.tracked_objects:
                b = obj.box
                vel = obj.velocity
                vx, vy = (vel.x, vel.y) if vel is not None else (0.0, 0.0)
                tracks[obj.track_token][k] = np.array([b.center.x, b.center.y, b.center.heading, vx, vy,
                                                       b.length, b.width])
                ttype[obj.track_token] = TYPE_MAP.get(obj.tracked_object_type, 3)
        scene["valid"][0, K - 1] = True
        scene["traj"][0, K - 1] = to_scene(ego_vec(cur))
        scene["atype"][0] = 0
        scene["center_offset"][0] = d.ego_rear_to_center
        cur_ids = sorted((tid for tid in tracks if K - 1 in tracks[tid]),
                         key=lambda tid: float(np.linalg.norm(tracks[tid][K - 1][:2] - origin)))
        for slot, tid in enumerate(cur_ids[: d.max_actors - 1], start=1):
            for k, st in tracks[tid].items():
                scene["traj"][slot, k] = to_scene(st)
                scene["valid"][slot, k] = True
            scene["atype"][slot] = ttype[tid]
        tl = {str(t.lane_connector_id): TL_MAP.get(t.status, TL_NONE) for t in current_input.traffic_light_data}
        polys: List[WorldPolyline] = []
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.CROSSWALK]
        objs = self._map_api.get_proximal_map_objects(Point2D(origin[0], origin[1]), d.map_radius, layers)
        T = d.hist_steps + d.fut_steps
        for lane in objs[SemanticMapLayer.LANE]:
            pts = np.array([[s.x, s.y] for s in lane.baseline_path.discrete_path])
            polys.append(WorldPolyline(pts=pts, kind="lane", on_route=str(lane.get_roadblock_id()) in self._route,
                                       speed_limit=float(lane.speed_limit_mps or float("nan"))))
            for bnd in (lane.left_boundary, lane.right_boundary):
                bp = np.array([[s.x, s.y] for s in bnd.discrete_path])
                polys.append(WorldPolyline(pts=bp, kind="boundary", on_route=False, speed_limit=float("nan")))
        for con in objs[SemanticMapLayer.LANE_CONNECTOR]:
            pts = np.array([[s.x, s.y] for s in con.baseline_path.discrete_path])
            state = tl.get(str(con.id))
            polys.append(WorldPolyline(pts=pts, kind="connector", on_route=str(con.get_roadblock_id()) in self._route,
                                       speed_limit=float(con.speed_limit_mps or float("nan")),
                                       tl=None if state is None else np.full(T, state, np.int8)))
        for cw in objs[SemanticMapLayer.CROSSWALK]:
            polys.append(WorldPolyline(pts=np.asarray(cw.polygon.exterior.coords)[:, :2], kind="crosswalk",
                                       on_route=False, speed_limit=float("nan")))
        route_ok = len(self._route) > 0
        fill_polylines(cfg, scene, polys, origin, yaw0, route_ok)
        sentence = instruction_from_route(scene["poly"].astype(np.float64), scene["poly_valid"], scene["poly_attr"],
                                          route_ok, cfg.feat.instr_len)
        scene["instr"][:] = tokenize(sentence, cfg.feat.instr_len)
        scene["src"][()] = SRC_NUPLAN
        scene["route_ok"][()] = route_ok
        return scene

    # ------------------------------------------------------------------ planning
    @torch.no_grad()
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        cfg = self._cfg
        dev = self._device
        K = cfg.data.hist_steps
        scene_np = self._scene(current_input)
        scene = scene_to_device({k: torch.as_tensor(v)[None] for k, v in scene_np.items()}, dev)
        feat = featurize(scene, K - 1, torch.zeros(1, dtype=torch.long, device=dev), cfg)
        out = self._model(feat)
        out = {k: v.float() if torch.is_floating_point(v) else v for k, v in out.items()}
        sel = out["mode_logits"].argmax(-1)
        ctrl = out["ctrl"][0, sel]
        trj = out["traj"][0, sel]
        u_lo = u_hi = None
        if self._method.get("cert"):
            c = certify(self._model, feat, cfg)
            u_lo, u_hi = c["u_lo"], c["u_hi"]
        rad = conformal.radius_torch(self._conf, out["agent_logscale"], feat["agents_phys"][..., 7]) if self._conf else None
        v0 = feat["ego_speed"]
        res = self._shield.step(self._method, v0, ctrl, trj, u_lo, u_hi, feat["agents_phys"], feat["agents_ok"],
                                out["agent_pred"], rad, feat["ego_dims"], torch.tensor([self._a_prev], device=dev))
        src = int(res["source"][0])
        u = ctrl.clone()
        if src in (SRC_FALLBACK, SRC_EMERGENCY):
            u[:, :, 0] = res["cmd"][0, 0]
            u[:, :, 1] = res["cmd"][0, 1]
        elif src == SRC_MPC:
            u[:, 0] = res["cmd"][0]
        self._a_prev = float(u[0, 0, 0])
        local = rollout(v0, u, cfg)[0].cpu().numpy()                     # (H, 4) in the current ego frame
        ego = current_input.history.ego_states[-1]
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = math.cos(h0), math.sin(h0)
        params = get_pacifica_parameters()
        t0 = ego.time_point.time_us
        r = int(round(cfg.feat.action_dt / cfg.data.dt))
        acc = u[0, :, 0].repeat_interleave(r).cpu().numpy()
        kap = u[0, :, 1].repeat_interleave(r).cpu().numpy()
        states = [ego]
        for i in range(local.shape[0]):
            lx, ly, lh, lv = local[i]
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x0 + lx * c - ly * s, y0 + lx * s + ly * c, h0 + lh),
                rear_axle_velocity_2d=StateVector2D(float(lv), 0.0),
                rear_axle_acceleration_2d=StateVector2D(float(acc[i]), 0.0),
                tire_steering_angle=float(math.atan(params.wheel_base * kap[i])),
                time_point=TimePoint(int(t0 + (i + 1) * cfg.data.dt * 1e6)),
                vehicle_parameters=params, is_in_auto_mode=True))
        return InterpolatedTrajectory(states)
