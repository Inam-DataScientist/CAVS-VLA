"""CARLA 0.9.13+ closed-loop bridge with a scripted stress-test library.

Fixes relative to the previous harness (see the evaluation document, D6-D7):
  * One coordinate convention: CARLA is left-handed (x east, y south, yaw clockwise in degrees).
    Everything is converted to the right-handed convention used in training:
        x_rh = x,  y_rh = -y,  yaw_rh = -radians(yaw),  vy_rh = -vy
    and the steering command is mirrored back: CARLA steer > 0 turns RIGHT, our curvature > 0 turns LEFT.
  * Collision and lane-invasion sensors are attached to the ego; collisions are counted by the
    simulator, not inferred.
  * Actor histories are keyed by the stable CARLA actor id.
  * The map input is built from CARLA waypoints (lanes, junction connectors, crosswalks, traffic-light
    states, route flags) with the same polyline packing as the nuPlan/WOMD builders.
  * The policy, certificate and shield are the same objects as in the log simulator.
"""
from __future__ import annotations

import math
import os
import random

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..config import METHOD_TABLE, Config, dump_json
from ..data.scene import SRC_CARLA, TL_GREEN, TL_NONE, TL_RED, TL_YELLOW, WorldPolyline, empty_scene, fill_polylines
from ..features import featurize, scene_to_device
from ..geometry_np import PolylineIndex, rotate_vec, to_frame
from ..language import instruction_from_route, tokenize
from ..safety.conformal import radius_torch
from ..safety.shield import SRC_EMERGENCY, SRC_FALLBACK, SRC_MPC, Shield
from ..safety.verifier import certify
from ..utils import LatencyMeter, get_logger


# ---------------------------------------------------------------- pure conversions (unit-tested without CARLA)

def lh_to_rh_xy(x: float, y: float) -> Tuple[float, float]:
    return x, -y


def lh_to_rh_yaw(yaw_deg: float) -> float:
    return -math.radians(yaw_deg)


def curvature_to_steer(kappa: float, wheelbase: float, max_steer_rad: float) -> float:
    """Kinematic bicycle: delta = atan(L * kappa) (left positive) -> CARLA steer in [-1, 1] (right positive)."""
    delta = math.atan(wheelbase * kappa)
    return float(np.clip(-delta / max_steer_rad, -1.0, 1.0))


class SpeedController:
    """Acceleration command -> throttle/brake.

    Uses the empirical acceleration envelope JSON from the previous repository when given
    (``carla.low_level_envelope``: {"envelope_min": [[v, a_min]...], "envelope_max": [[v, a_max]...]}),
    otherwise fixed maps. A PI loop on the integrated speed reference removes steady-state bias.
    """

    def __init__(self, cfg: Config) -> None:
        self.kp, self.ki = cfg.carla.speed_kp, cfg.carla.speed_ki
        self.dt = cfg.carla.dt
        self.integral = 0.0
        self.env_min = self.env_max = None
        if cfg.carla.low_level_envelope and os.path.exists(cfg.carla.low_level_envelope):
            import json
            with open(cfg.carla.low_level_envelope) as f:
                data = json.load(f)
            self.env_min = np.asarray(data["envelope_min"], dtype=np.float64)
            self.env_max = np.asarray(data["envelope_max"], dtype=np.float64)

    def reset(self) -> None:
        self.integral = 0.0

    def __call__(self, a_cmd: float, v: float, v_ref: float) -> Tuple[float, float]:
        err = v_ref - v
        self.integral = float(np.clip(self.integral + err * self.dt, -2.0, 2.0))
        a = a_cmd + self.kp * err + self.ki * self.integral
        if self.env_max is not None:
            a_max = max(float(np.interp(v, self.env_max[:, 0], self.env_max[:, 1])), 0.1)
            a_min = min(float(np.interp(v, self.env_min[:, 0], self.env_min[:, 1])), -0.1)
        else:
            a_max, a_min = 3.5, -8.0
        if a >= 0:
            return float(np.clip(a / a_max, 0.0, 1.0)), 0.0
        return 0.0, float(np.clip(a / a_min, 0.0, 1.0))


# ---------------------------------------------------------------- bridge

class CarlaBridge:
    def __init__(self, cfg: Config, model, hj=None, conf=None, device=None) -> None:
        import carla  # noqa: F401  (fails early with a clear error if the client is missing)
        self.carla = carla
        self.cfg = cfg
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hj, self.conf = hj, conf
        self.shield = Shield(cfg, hj, conf)
        self.log = get_logger("carla")
        self.client = carla.Client(cfg.carla.host, cfg.carla.port)
        self.client.set_timeout(30.0)
        self.log.info(f"CARLA server {self.client.get_server_version()} / client {self.client.get_client_version()}")
        if self.client.get_server_version() != self.client.get_client_version():
            self.log.warning("CARLA client and server versions differ; results may be unreliable")
        world = self.client.get_world()
        if not world.get_map().name.endswith(cfg.carla.town):
            world = self.client.load_world(cfg.carla.town)
        self.world = world
        self.map = world.get_map()
        self.tm = self.client.get_trafficmanager(cfg.carla.tm_port)
        self._orig_settings = world.get_settings()
        st = world.get_settings()
        st.synchronous_mode = True
        st.fixed_delta_seconds = cfg.carla.dt
        world.apply_settings(st)
        self.tm.set_synchronous_mode(True)
        self.latency = LatencyMeter()
        self._build_map()

    # ------------------------------------------------------------ map
    def _build_map(self) -> None:
        carla = self.carla
        groups: Dict[tuple, list] = defaultdict(list)
        for wp in self.map.generate_waypoints(2.0):
            if wp.lane_type != carla.LaneType.Driving:
                continue
            groups[(wp.road_id, wp.section_id, wp.lane_id)].append(wp)
        polys = []
        self.key_of_poly: List[tuple] = []
        for key, wps in groups.items():
            wps = sorted(wps, key=lambda w: w.s)
            if key[2] > 0:   # lanes with positive id run against the road's reference direction
                wps = wps[::-1]
            pts = np.array([lh_to_rh_xy(w.transform.location.x, w.transform.location.y) for w in wps])
            if len(pts) < 2:
                continue
            kind = "connector" if wps[0].is_junction else "lane"
            polys.append(dict(pts=pts, kind=kind, id=f"{key[0]}_{key[1]}_{key[2]}", roadblock_id=None,
                              speed_limit=13.9, key=key))
        cw = self.map.get_crosswalks()
        cur: List[Tuple[float, float]] = []
        for loc in cw:
            p = lh_to_rh_xy(loc.x, loc.y)
            if cur and abs(p[0] - cur[0][0]) < 1e-3 and abs(p[1] - cur[0][1]) < 1e-3 and len(cur) >= 3:
                polys.append(dict(pts=np.array(cur + [cur[0]]), kind="crosswalk", id=f"cw{len(polys)}",
                                  roadblock_id=None, speed_limit=float("nan"), key=None))
                cur = []
            else:
                cur.append(p)
        self.poly_index = PolylineIndex(polys)
        self.tl_keys: Dict[int, set] = {}
        for tl in self.world.get_actors().filter("traffic.traffic_light"):
            keys = set()
            for wp in tl.get_affected_lane_waypoints():
                nxt = wp
                for _ in range(15):     # the connector(s) right after the stop line
                    keys.add((nxt.road_id, nxt.section_id, nxt.lane_id))
                    nn = nxt.next(2.0)
                    if not nn:
                        break
                    nxt = nn[0]
            self.tl_keys[tl.id] = keys

    def _route(self, start, length_m: float = 600.0) -> Tuple[set, np.ndarray]:
        """Route keys and points: GlobalRoutePlanner when available, else lane following."""
        keys, pts = set(), []
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            spawns = self.map.get_spawn_points()
            rng = random.Random(0)
            dest = max(rng.sample(spawns, min(30, len(spawns))), key=lambda s: s.location.distance(start.location))
            grp = GlobalRoutePlanner(self.map, 2.0)
            for wp, _ in grp.trace_route(start.location, dest.location):
                keys.add((wp.road_id, wp.section_id, wp.lane_id))
                pts.append(lh_to_rh_xy(wp.transform.location.x, wp.transform.location.y))
                if len(pts) * 2.0 > length_m:
                    break
        except ImportError:
            wp = self.map.get_waypoint(start.location, project_to_road=True)
            for _ in range(int(length_m / 2.0)):
                keys.add((wp.road_id, wp.section_id, wp.lane_id))
                pts.append(lh_to_rh_xy(wp.transform.location.x, wp.transform.location.y))
                nxt = wp.next(2.0)
                if not nxt:
                    break
                wp = nxt[0]
        return keys, np.asarray(pts)

    # ------------------------------------------------------------ state
    def _actor_state(self, a) -> np.ndarray:
        tf = a.get_transform()
        v = a.get_velocity()
        x, y = lh_to_rh_xy(tf.location.x, tf.location.y)
        ext = a.bounding_box.extent
        return np.array([x, y, lh_to_rh_yaw(tf.rotation.yaw), v.x, -v.y, 2 * ext.x, 2 * ext.y])

    @staticmethod
    def _actor_type(a) -> int:
        tid = a.type_id
        if tid.startswith("walker."):
            return 1
        if tid.startswith("vehicle."):
            bike = any(k in tid for k in ("bike", "bicycle", "bh.crossbike", "diamondback", "gazelle", "harley",
                                          "kawasaki", "yamaha", "vespa"))
            return 2 if bike else 0
        return 3

    def _scene(self, ego, hist: Dict[int, deque], types: Dict[int, int], route_keys: set) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        d = cfg.data
        K = d.hist_steps
        scene = empty_scene(cfg)
        cur = hist[ego.id][-1]
        origin, yaw0 = cur[:2], float(cur[2])

        def put(slot, aid):
            frames = list(hist[aid])
            n = len(frames)
            for k in range(n):
                s = frames[k].copy()
                s[:2] = to_frame(s[None, :2], origin, yaw0)[0]
                s[2] = math.atan2(math.sin(s[2] - yaw0), math.cos(s[2] - yaw0))
                s[3:5] = rotate_vec(s[None, 3:5], yaw0)[0]
                scene["traj"][slot, K - n + k] = s
                scene["valid"][slot, K - n + k] = True
            scene["atype"][slot] = types[aid]

        put(0, ego.id)
        scene["center_offset"][0] = ego.bounding_box.location.x
        others = [aid for aid in hist if aid != ego.id and len(hist[aid])]
        dist = sorted(others, key=lambda aid: float(np.linalg.norm(hist[aid][-1][:2] - origin)))
        slot = 1
        for aid in dist:
            if slot >= d.max_actors:
                break
            if np.linalg.norm(hist[aid][-1][:2] - origin) > d.actor_radius:
                break
            put(slot, aid)
            slot += 1
        tl_state = {}
        for tl in self.world.get_actors().filter("traffic.traffic_light"):
            st = tl.get_state()
            code = TL_RED if st == self.carla.TrafficLightState.Red else (
                TL_YELLOW if st == self.carla.TrafficLightState.Yellow else (
                    TL_GREEN if st == self.carla.TrafficLightState.Green else TL_NONE))
            for key in self.tl_keys.get(tl.id, ()):
                tl_state[key] = code
        polys = []
        for p, _ in self.poly_index.query(float(origin[0]), float(origin[1]), d.map_radius):
            tl = tl_state.get(p["key"]) if p["key"] is not None else None
            polys.append(WorldPolyline(pts=p["pts"], kind=p["kind"], on_route=p["key"] in route_keys,
                                       speed_limit=p["speed_limit"],
                                       tl=None if tl is None else np.full(d.hist_steps + d.fut_steps, tl, np.int8)))
        fill_polylines(cfg, scene, polys, origin, yaw0, True)
        sentence = instruction_from_route(scene["poly"].astype(np.float64), scene["poly_valid"], scene["poly_attr"],
                                          True, cfg.feat.instr_len)
        scene["instr"][:] = tokenize(sentence, cfg.feat.instr_len)
        scene["src"][()] = SRC_CARLA
        scene["route_ok"][()] = True
        return scene

    # ------------------------------------------------------------ scenario scripting
    def _spawn(self, bp_name: str, tf):
        bp = self.world.get_blueprint_library().find(bp_name)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "scenario")
        return self.world.try_spawn_actor(bp, tf)

    def _ahead_tf(self, ego_tf, dist: float, lane_shift: int = 0):
        wp = self.map.get_waypoint(ego_tf.location, project_to_road=True)
        for _ in range(lane_shift if lane_shift > 0 else -lane_shift):
            nxt = wp.get_left_lane() if lane_shift > 0 else wp.get_right_lane()
            if nxt is None or nxt.lane_type != self.carla.LaneType.Driving or nxt.lane_id * wp.lane_id < 0:
                return None
            wp = nxt
        nn = wp.next(dist) if dist > 0 else wp.previous(-dist)
        if not nn:
            return None
        tf = nn[0].transform
        tf.location.z += 0.5
        return tf

    def _script(self, name: str, ego, rng: random.Random) -> Dict:
        carla = self.carla
        actors, state = [], {}
        etf = ego.get_transform()
        if name == "lead_brake":
            tf = self._ahead_tf(etf, rng.uniform(20.0, 30.0))
            lead = self._spawn("vehicle.audi.a2", tf) if tf else None
            if lead:
                actors.append(lead)
                state.update(lead=lead, brake_t=rng.uniform(6.0, 10.0), v_set=rng.uniform(8.0, 11.0))
        elif name == "cut_in":
            side = rng.choice([1, -1])
            tf = self._ahead_tf(etf, rng.uniform(5.0, 12.0), lane_shift=side)
            npc = self._spawn("vehicle.lincoln.mkz_2017", tf) if tf else None
            if npc:
                actors.append(npc)
                ego_wp = self.map.get_waypoint(etf.location, project_to_road=True)
                state.update(cut=npc, side=side, cut_t=rng.uniform(3.0, 6.0), v_set=rng.uniform(9.0, 12.0),
                             target_lane=ego_wp.lane_id)
        elif name == "pedestrian":
            tf = self._ahead_tf(etf, rng.uniform(25.0, 35.0))
            if tf:
                right = tf.get_right_vector()
                lane_w = self.map.get_waypoint(tf.location).lane_width
                tf.location += carla.Location(x=right.x * (lane_w + 1.5), y=right.y * (lane_w + 1.5))
                bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
                walker = self.world.try_spawn_actor(bps[rng.randrange(len(bps))], tf)
                if walker:
                    actors.append(walker)
                    state.update(walker=walker, dir=carla.Vector3D(-right.x, -right.y, 0.0),
                                 go_t=rng.uniform(1.0, 3.0), speed=rng.uniform(1.2, 1.8))
        elif name == "dense":
            spawns = self.map.get_spawn_points()
            rng.shuffle(spawns)
            bps = [b for b in self.world.get_blueprint_library().filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) == 4]
            for sp in spawns[: self.cfg.carla.npc_vehicles]:
                if sp.location.distance(etf.location) < 10.0:
                    continue
                v = self.world.try_spawn_actor(rng.choice(bps), sp)
                if v:
                    v.set_autopilot(True, self.cfg.carla.tm_port)
                    actors.append(v)
        else:
            raise ValueError(f"Unknown CARLA scenario {name}")
        return dict(actors=actors, **state)

    def _drive_script(self, sc: Dict, t: float) -> None:
        carla = self.carla
        if "lead" in sc:
            lead = sc["lead"]
            v = lead.get_velocity()
            spd = math.hypot(v.x, v.y)
            if t < sc["brake_t"]:
                thr = float(np.clip(0.5 * (sc["v_set"] - spd), 0, 0.8))
                lead.apply_control(carla.VehicleControl(throttle=thr, steer=self._lane_keep(lead), brake=0.0))
            else:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=self._lane_keep(lead), brake=1.0))
        if "cut" in sc:
            npc = sc["cut"]
            v = npc.get_velocity()
            spd = math.hypot(v.x, v.y)
            thr = float(np.clip(0.5 * (sc["v_set"] - spd), 0, 0.8))
            in_target = self.map.get_waypoint(npc.get_location(), project_to_road=True).lane_id == sc["target_lane"]
            shift = 0 if (t < sc["cut_t"] or in_target) else -sc["side"]
            npc.apply_control(carla.VehicleControl(throttle=thr, steer=self._lane_keep(npc, shift), brake=0.0))
        if "walker" in sc:
            spd = sc["speed"] if t >= sc["go_t"] else 0.0
            sc["walker"].apply_control(carla.WalkerControl(direction=sc["dir"], speed=spd))

    def _lane_keep(self, actor, lane_shift: int = 0) -> float:
        """Pure pursuit on the (optionally shifted) lane centre, CARLA steer convention."""
        tf = actor.get_transform()
        wp = self.map.get_waypoint(tf.location, project_to_road=True)
        if lane_shift:
            alt = wp.get_left_lane() if lane_shift > 0 else wp.get_right_lane()
            if alt is not None and alt.lane_type == self.carla.LaneType.Driving:
                wp = alt
        tgt = wp.next(8.0)
        if not tgt:
            return 0.0
        loc = tgt[0].transform.location
        dx, dy = loc.x - tf.location.x, loc.y - tf.location.y
        yaw = math.radians(tf.rotation.yaw)
        ang = math.atan2(dy, dx) - yaw
        ang = math.atan2(math.sin(ang), math.cos(ang))
        return float(np.clip(1.5 * ang, -1.0, 1.0))     # left-handed: positive angle -> turn right -> steer > 0

    # ------------------------------------------------------------ episode
    @torch.no_grad()
    def run_episode(self, scenario: str, method_name: str, seed: int) -> Dict:
        carla, cfg = self.carla, self.cfg
        method = METHOD_TABLE[method_name]
        rng = random.Random(seed)
        self.tm.set_random_device_seed(seed)
        spawns = self.map.get_spawn_points()
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        ego = None
        order = list(range(len(spawns)))
        rng.shuffle(order)
        for i in order:
            ego = self.world.try_spawn_actor(bp, spawns[i])
            if ego:
                break
        if ego is None:
            raise RuntimeError("Could not spawn the ego vehicle")
        events = dict(collisions=[], invasions=0)
        bpl = self.world.get_blueprint_library()
        col = self.world.spawn_actor(bpl.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        inv = self.world.spawn_actor(bpl.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: events["collisions"].append(
            (e.other_actor.type_id, math.sqrt(e.normal_impulse.x ** 2 + e.normal_impulse.y ** 2 + e.normal_impulse.z ** 2))))

        def on_inv(e):
            solid = [m for m in e.crossed_lane_markings if "Solid" in str(m.type)]
            if solid:
                events["invasions"] += 1
        inv.listen(on_inv)
        phys = ego.get_physics_control()
        wheels = phys.wheels
        wheelbase = abs(wheels[0].position.x - wheels[2].position.x) / 100.0 or 2.9
        max_steer = math.radians(wheels[0].max_steer_angle)
        for _ in range(5):
            self.world.tick()
        route_keys, route_pts = self._route(ego.get_transform())
        sc = self._script(scenario, ego, rng)
        speed_ctl = SpeedController(cfg)
        K = cfg.data.hist_steps
        hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=K))
        types: Dict[int, int] = {}
        a_prev = 0.0
        v_ref = 0.0
        stats = dict(steps=0, interventions=0, mpc=0, fallback=0, emergency=0, certified=0, min_dist=float("inf"),
                     min_margin=float("inf"), speed_sum=0.0)
        n_steps = int(cfg.carla.episode_seconds / cfg.carla.dt)
        start_xy = None
        try:
            for step in range(n_steps):
                t = step * cfg.carla.dt
                self._drive_script(sc, t)
                self.world.tick()
                for a in list(self.world.get_actors().filter("vehicle.*")) + \
                        list(self.world.get_actors().filter("walker.pedestrian.*")):
                    s = self._actor_state(a)
                    if a.id != ego.id and np.linalg.norm(s[:2] - (hist[ego.id][-1][:2] if hist[ego.id] else s[:2])) > 120:
                        continue
                    hist[a.id].append(s)
                    types[a.id] = 0 if a.id == ego.id else self._actor_type(a)
                live = {a.id for a in self.world.get_actors()}
                for aid in [k for k in hist if k not in live]:
                    del hist[aid]
                cur = hist[ego.id][-1]
                if start_xy is None:
                    start_xy = cur[:2].copy()
                if len(hist[ego.id]) < K:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                    continue
                scene_np = self._scene(ego, hist, types, route_keys)
                scene = scene_to_device({k: torch.as_tensor(v)[None] for k, v in scene_np.items()}, self.device)
                with self.latency.measure("featurize"):
                    feat = featurize(scene, K - 1, torch.zeros(1, dtype=torch.long, device=self.device), cfg)
                with self.latency.measure("policy"):
                    out = self.model(feat)
                out = {k: v.float() if torch.is_floating_point(v) else v for k, v in out.items()}
                sel = out["mode_logits"].argmax(-1)
                ctrl = out["ctrl"][0, sel]
                trj = out["traj"][0, sel]
                u_lo = u_hi = None
                if method.get("cert"):
                    with self.latency.measure("certify"):
                        cert = certify(self.model, feat, cfg)
                    u_lo, u_hi = cert["u_lo"], cert["u_hi"]
                rad = radius_torch(self.conf, out["agent_logscale"], feat["agents_phys"][..., 7]) if self.conf else None
                v_now = float(np.hypot(cur[3], cur[4]))
                with self.latency.measure("shield"):
                    res = self.shield.step(method, torch.tensor([v_now], device=self.device), ctrl, trj, u_lo, u_hi,
                                           feat["agents_phys"], feat["agents_ok"], out["agent_pred"], rad,
                                           feat["ego_dims"], torch.tensor([a_prev], device=self.device))
                a_cmd, k_cmd = float(res["cmd"][0, 0]), float(res["cmd"][0, 1])
                v_ref = max(0.0, (v_ref if step > K else v_now) + a_cmd * cfg.carla.dt)
                thr, brk = speed_ctl(a_cmd, v_now, v_ref)
                steer = curvature_to_steer(k_cmd, wheelbase, max_steer)
                ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk,
                                                       hand_brake=bool(brk > 0.8 and v_now < 0.5)))
                a_prev = a_cmd
                src = int(res["source"][0])
                stats["steps"] += 1
                stats["interventions"] += int(src > 0)
                stats["mpc"] += int(src == SRC_MPC)
                stats["fallback"] += int(src == SRC_FALLBACK)
                stats["emergency"] += int(src == SRC_EMERGENCY)
                stats["certified"] += int(bool(res["certified"][0]))
                m = float(res["margin"][0])
                if math.isfinite(m):
                    stats["min_margin"] = min(stats["min_margin"], m)
                stats["speed_sum"] += v_now
                ap = feat["agents_phys"][0]
                ok = feat["agents_ok"][0]
                if bool(ok.any()):
                    dd = torch.linalg.norm(ap[ok, :2], dim=-1) - 0.5 * torch.hypot(ap[ok, 5], ap[ok, 6]) - 1.2
                    stats["min_dist"] = min(stats["min_dist"], float(dd.min().clamp(min=0)))
                if events["collisions"]:
                    break
        finally:
            for s in (col, inv):
                s.stop()
            ids = [a.id for a in sc.get("actors", [])] + [col.id, inv.id, ego.id]
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        travelled = 0.0
        if len(route_pts) > 1 and start_xy is not None:
            final = hist[ego.id][-1][:2] if hist.get(ego.id) else start_xy
            seg = np.linalg.norm(np.diff(route_pts, axis=0), axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(seg)])
            k = int(np.argmin(np.linalg.norm(route_pts - final, axis=1)))
            travelled = float(s_cum[k])
        steps = max(stats["steps"], 1)
        return dict(scenario=scenario, method=method_name, seed=seed, collision=bool(events["collisions"]),
                    collision_with=events["collisions"][0][0] if events["collisions"] else "",
                    lane_invasions=events["invasions"], route_m=travelled, mean_speed=stats["speed_sum"] / steps,
                    intervention_rate=stats["interventions"] / steps, mpc_rate=stats["mpc"] / steps,
                    fallback_rate=stats["fallback"] / steps, emergency_rate=stats["emergency"] / steps,
                    certified_fraction=stats["certified"] / steps, min_distance=stats["min_dist"],
                    min_safety_margin=stats["min_margin"])

    def measure_disturbance(self, seconds: float = 120.0, seed: int = 0) -> Dict:
        """Hold random acceleration commands through SpeedController and measure the realised acceleration."""
        carla, cfg = self.carla, self.cfg
        rng = np.random.default_rng(seed)
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = None
        for sp in self.map.get_spawn_points():
            ego = self.world.try_spawn_actor(bp, sp)
            if ego:
                break
        ctl = SpeedController(cfg)
        errs = []
        v_ref = 0.0
        hold = int(round(1.0 / cfg.carla.dt))
        try:
            v_prev = 0.0
            for i in range(int(seconds / cfg.carla.dt)):
                if i % hold == 0:
                    a_cmd = float(rng.uniform(cfg.hj.a_e_min, cfg.hj.a_e_max)) if v_prev > 3.0 else 2.0
                v = ego.get_velocity()
                v_now = math.hypot(v.x, v.y)
                v_ref = max(0.0, v_ref + a_cmd * cfg.carla.dt)
                thr, brk = ctl(a_cmd, v_now, v_ref)
                ego.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=self._lane_keep(ego)))
                self.world.tick()
                v2 = ego.get_velocity()
                v_new = math.hypot(v2.x, v2.y)
                if i % hold >= hold // 2 and v_now > 0.5 and v_new > 0.5:
                    errs.append((v_new - v_now) / cfg.carla.dt - a_cmd)
                v_prev = v_new
                if v_new > cfg.hj.v_max - 3:
                    v_ref = v_new - 3
        finally:
            ego.destroy()
        e = np.abs(np.asarray(errs))
        return dict(w_bar_q999=float(np.quantile(e, 0.999)) if len(e) else float("nan"),
                    w_bar_q99=float(np.quantile(e, 0.99)) if len(e) else float("nan"), samples=int(len(e)))

    def close(self) -> None:
        self.world.apply_settings(self._orig_settings)
        self.tm.set_synchronous_mode(False)


def run_carla_suite(cfg: Config, ckpt: str, hj=None, conf=None) -> Dict:
    from ..train import load_policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_m, _ = load_policy(ckpt, device)
    for sec in ("sim", "shield", "hj", "cert", "conformal", "stl", "carla"):
        setattr(cfg_m, sec, getattr(cfg, sec))
    bridge = CarlaBridge(cfg_m, model, hj, conf, device)
    os.makedirs(cfg.carla.out_dir, exist_ok=True)
    rows = []
    try:
        for scenario in cfg.carla.scenarios:
            for ep in range(cfg.carla.episodes_per_scenario):
                r = bridge.run_episode(scenario, cfg.carla.method, seed=1000 + ep)
                rows.append(r)
                bridge.log.info(str(r))
    finally:
        bridge.close()
    dump_json(dict(rows=rows, latency=bridge.latency.summary()), os.path.join(cfg.carla.out_dir,
                                                                           f"carla_{cfg.carla.method}.json"))
    return dict(rows=rows)
