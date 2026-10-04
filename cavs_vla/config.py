"""Typed configuration for the whole CAVS-VLA v2 pipeline.

Every stage (data build, training, verification, HJ, simulation, CARLA) reads
one nested ``Config``. YAML files override defaults; ``--set a.b=c`` style
overrides are applied last. The resolved config is snapshotted next to every
artifact so that each result can be traced back to its exact settings.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    # Temporal contract (identical for nuPlan, WOMD and CARLA).
    dt: float = 0.1                  # seconds between frames (10 Hz)
    hist_steps: int = 11             # K, includes the current frame -> 1.0 s history
    fut_steps: int = 80              # H -> 8.0 s future
    # Raw scene capacity (fixed tensor shapes, padded with validity masks).
    max_actors: int = 64             # actor 0 is always the ego
    max_polylines: int = 192
    poly_points: int = 20            # points per polyline chunk
    poly_chunk_len: float = 30.0     # maximum chunk length in metres
    map_radius: float = 110.0        # polylines intersecting this radius are stored
    actor_radius: float = 100.0      # actors within this radius at t0 are stored
    window_stride_s: float = 1.0     # spacing between consecutive windows of a log
    max_windows_per_log: int = 100000
    # Navigation route as an input (nuPlan scene.roadblock_ids). False = ablation.
    use_route: bool = True
    # nuPlan
    nuplan_db_root: str = ""
    nuplan_map_root: str = ""
    nuplan_map_version: str = "nuplan-maps-v1.0"
    map_backend: str = "auto"        # auto | devkit | gpkg | json | none
    json_map_path: str = ""          # map JSON for map_backend=json (synthetic data, CARLA exports)
    # Waymo Open Motion Dataset (Scenario protos in TFRecord files)
    waymo_root: str = ""
    waymo_file_glob: str = "**/*.tfrecord*"
    waymo_max_files: int = 0         # 0 = all
    sources: List[str] = field(default_factory=lambda: ["nuplan"])
    # Splits are by log (nuPlan) or by file (WOMD); never by window.
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    split_seed: int = 42
    split_file: str = ""             # optional JSON {log_name: "train"|"val"|"test"}
    out_dir: str = "data/cavs_v2"
    num_workers: int = 8
    # Ego vehicle geometry (nuPlan Chrysler Pacifica; WOMD uses box centres).
    ego_length: float = 5.176
    ego_width: float = 2.297
    ego_rear_to_center: float = 1.461
    hash_outputs: bool = True


@dataclass
class FeatureConfig:
    num_agents: int = 32             # N agents given to the policy
    num_polylines: int = 64          # P polylines given to the policy
    instr_len: int = 16              # instruction tokens
    pos_scale: float = 50.0          # fixed physical scales (no fitted min/max)
    vel_scale: float = 20.0
    size_scale: float = 10.0
    ego_history_dropout: float = 0.5 # train-time only; counters the ego-status shortcut
    action_dt: float = 0.2           # control (action-token) period in seconds
    a_min: float = -8.0              # acceleration token range (m/s^2)
    a_max: float = 4.0
    n_acc_bins: int = 25
    kappa_max: float = 0.25          # curvature token range (1/m)
    n_kappa_bins: int = 41
    kappa_min_ds: float = 0.5        # below this travelled distance curvature is undefined -> 0


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 6
    enc_layers: int = 4
    dec_layers: int = 2
    ff_mult: int = 4
    n_modes: int = 6
    num_intents: int = 8
    ln_eps: float = 1e-5


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    amp: str = "bf16"                # bf16 | fp16 | off
    compile: bool = False
    seed: int = 42
    num_workers: int = 8
    log_every: int = 50
    val_every: int = 1
    max_train_batches: int = 0       # 0 = full epoch (used by selftest to keep runs short)
    max_val_batches: int = 0
    ckpt_dir: str = "checkpoints/cavs_v2"
    resume: str = ""
    # Loss weights
    w_reg: float = 1.0
    w_cls: float = 0.5
    w_tok: float = 0.2
    w_agent: float = 0.5
    w_intent: float = 0.1
    w_safe: float = 0.0              # differentiable HJ / proximity penalty (CEG stage)
    safe_margin: float = 0.5
    # Counterexample-guided refinement
    ceg_epochs: int = 3
    ceg_lr: float = 5e-5
    ceg_hard_weight: float = 4.0
    ceg_w_safe: float = 0.5


@dataclass
class HJConfig:
    d_lo: float = -2.0
    d_hi: float = 100.0
    n_d: int = 205
    v_max: float = 30.0
    n_v: int = 61
    a_e_min: float = -7.0            # ego braking authority assumed by the safety layer
    a_e_max: float = 3.0
    a_o_min: float = -8.0            # other agent's worst-case braking
    a_o_max: float = 3.0
    w_bar: float = 0.6               # bound on |a_actual - a_command| (from estimate-disturbance)
    d_min: float = 0.5               # bumper-to-bumper gap that counts as unsafe
    horizon: float = 7.0             # seconds of backward propagation
    cfl: float = 0.45
    table_path: str = "artifacts/hj_table.npz"


@dataclass
class CertConfig:
    # Perception-uncertainty box X0 in physical units.
    eps_agent_pos: float = 0.3       # m
    eps_agent_vel: float = 0.5       # m/s
    eps_agent_heading_deg: float = 3.0
    eps_ego_speed: float = 0.2       # m/s
    eps_map_pos: float = 0.2         # m
    enabled: bool = True


@dataclass
class ConformalConfig:
    alpha: float = 0.05
    min_scale: float = 0.2           # floor on predicted scale used in the score
    table_path: str = "artifacts/conformal.json"
    max_batches: int = 200


@dataclass
class ShieldConfig:
    commit_s: float = 0.3            # control committed before the next check (replan + latency)
    check_horizon_s: float = 6.0
    hj_margin: float = 0.5           # extra margin on the HJ value (m)
    corridor_margin: float = 0.4     # lateral margin added to half ego width (m)
    lead_heading_deg: float = 35.0   # heading difference that still counts as "same direction"
    max_interactions: int = 12       # agents considered per scene
    time_margin_s: float = 0.5       # crossing-window time margin
    mpc_steps: int = 20              # action steps optimised by MPC (x action_dt)
    mpc_outer: int = 4
    mpc_inner: int = 20
    mpc_lr: float = 0.15
    w_track: float = 1.0
    w_ctrl_a: float = 0.3
    w_ctrl_k: float = 50.0
    w_jerk: float = 0.1
    rho0: float = 5.0
    fallback_candidates: int = 25
    rear_soft_weight: float = 0.5    # RSS-style: discourage braking harder than needed with a close follower


@dataclass
class STLConfig:
    d_min: float = 0.5
    v_max: float = 25.0
    lane_dev_max: float = 2.5


@dataclass
class SimConfig:
    num_scenarios: int = 1000
    batch_size: int = 32
    split: str = "test"
    reactive: bool = True            # IDM agents along logged paths (CLS-R style); False = log replay
    idm_T: float = 1.5
    idm_a: float = 1.5
    idm_b: float = 2.0
    idm_s0: float = 2.0
    idm_delta: float = 4.0
    # Plant (actuator) model -> model mismatch w.r.t. the kinematic bicycle used for planning.
    tau_a: float = 0.1
    tau_k: float = 0.05
    accel_noise: float = 0.2         # uniform bounded noise on executed acceleration
    delay_steps: int = 0
    replan_interval: int = 1         # policy period in sim steps
    latency_steps: int = 0           # policy output delay in sim steps
    num_controlled: int = 1          # >1 = multi-CAV coordination experiment
    comm_age_steps: int = 1
    methods: List[str] = field(default_factory=lambda: ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    out_dir: str = "results/closed_loop"
    collision_ignore_static_below_speed: float = 0.0
    lane_half_width: float = 1.75
    offroad_tolerance: float = 0.75
    deadlock_s: float = 5.0


@dataclass
class CarlaConfig:
    host: str = "localhost"
    port: int = 2000
    tm_port: int = 8000
    town: str = "Town05"
    dt: float = 0.1
    episodes_per_scenario: int = 20
    scenarios: List[str] = field(default_factory=lambda: ["lead_brake", "cut_in", "pedestrian", "dense"])
    episode_seconds: float = 30.0
    npc_vehicles: int = 40
    method: str = "B7"
    out_dir: str = "results/carla"
    low_level_envelope: str = ""     # optional models/disturbance_set_W_v2.json from the old repo
    speed_kp: float = 0.6
    speed_ki: float = 0.05


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hj: HJConfig = field(default_factory=HJConfig)
    cert: CertConfig = field(default_factory=CertConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    stl: STLConfig = field(default_factory=STLConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    carla: CarlaConfig = field(default_factory=CarlaConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dumps(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _merge_into(dc: Any, values: Dict[str, Any], path: str = "") -> None:
    names = {f.name: f for f in fields(dc)}
    for key, val in values.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{path}{key}'. Valid keys: {sorted(names)}")
        cur = getattr(dc, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise TypeError(f"Config section '{path}{key}' expects a mapping, got {type(val).__name__}")
            _merge_into(cur, val, path=f"{path}{key}.")
        else:
            setattr(dc, key, _coerce(cur, val, f"{path}{key}"))


def _coerce(cur: Any, val: Any, name: str) -> Any:
    if cur is None or val is None:
        return val
    if isinstance(cur, bool):
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"Cannot parse boolean for {name}: {val}")
        return bool(val)
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, list):
        if isinstance(val, str):
            parsed = yaml.safe_load(val)
            if not isinstance(parsed, list):
                parsed = [parsed]
            return parsed
        return list(val)
    if isinstance(cur, str):
        return str(val)
    return val


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """Turn ['train.lr=1e-4', 'data.sources=[nuplan,waymo]'] into a nested dict."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        val = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r") as f:
            values = yaml.safe_load(f) or {}
        _merge_into(cfg, values)
    if overrides:
        _merge_into(cfg, parse_overrides(overrides))
    validate_config(cfg)
    return cfg


def config_from_dict(values: Dict[str, Any]) -> Config:
    cfg = Config()
    _merge_into(cfg, copy.deepcopy(values))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    d, f = cfg.data, cfg.feat
    steps_per_action = f.action_dt / d.dt
    if abs(steps_per_action - round(steps_per_action)) > 1e-6 or round(steps_per_action) < 1:
        raise ValueError(f"feat.action_dt ({f.action_dt}) must be an integer multiple of data.dt ({d.dt})")
    if d.fut_steps % int(round(steps_per_action)) != 0:
        raise ValueError("data.fut_steps must be divisible by action_dt/dt")
    if d.hist_steps < 2:
        raise ValueError("data.hist_steps must be >= 2")
    if f.num_agents >= d.max_actors:
        raise ValueError("feat.num_agents must be smaller than data.max_actors (actor 0 is the observer)")
    if f.num_polylines > d.max_polylines:
        raise ValueError("feat.num_polylines must be <= data.max_polylines")
    if abs(sum(d.split_ratios) - 1.0) > 1e-6 or len(d.split_ratios) != 3:
        raise ValueError("data.split_ratios must be three numbers summing to 1")
    if cfg.model.d_model % cfg.model.n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if cfg.hj.a_e_min >= 0 or cfg.hj.a_o_min >= 0:
        raise ValueError("HJ braking bounds must be negative")
    if cfg.hj.a_e_min < f.a_min:
        raise ValueError("hj.a_e_min cannot exceed the policy's acceleration token range (feat.a_min)")
    for m in cfg.sim.methods:
        if m not in METHOD_TABLE:
            raise ValueError(f"Unknown method {m}; valid: {sorted(METHOD_TABLE)}")
    if cfg.sim.num_controlled < 1:
        raise ValueError("sim.num_controlled must be >= 1")


# Ablation matrix of the paper (master plan §38) as explicit switches.
METHOD_TABLE: Dict[str, Dict[str, Any]] = {
    # B0: expert replay of the logged ego trajectory (reference controller).
    "B0": dict(expert=True, cert=False, hj=False, mpc=False, geometric=False, fallback="none"),
    # B1: policy only.
    "B1": dict(expert=False, cert=False, hj=False, mpc=False, geometric=False, fallback="none"),
    # B2: policy + always-on MPC filter (collision constraints from conformal predictions).
    "B2": dict(expert=False, cert=False, hj=False, mpc=True, geometric=True, fallback="none", mpc_always=True),
    # B3: policy + HJ check; unsafe -> HJ least-restrictive backup (no MPC).
    "B3": dict(expert=False, cert=False, hj=True, mpc=False, geometric=False, fallback="hj"),
    # B4: policy + neural certificate + geometric tube check; unsafe -> maximum braking.
    "B4": dict(expert=False, cert=True, hj=False, mpc=False, geometric=True, fallback="brake"),
    # B5: neural certificate + geometric check + MPC on failure.
    "B5": dict(expert=False, cert=True, hj=False, mpc=True, geometric=True, fallback="brake"),
    # B6: HJ + MPC with the policy's point output (no input uncertainty).
    "B6": dict(expert=False, cert=False, hj=True, mpc=True, geometric=True, fallback="hj"),
    # B7: full stack: certificate + HJ + MPC + verified HJ backup.
    "B7": dict(expert=False, cert=True, hj=True, mpc=True, geometric=True, fallback="hj"),
}


def dump_json(obj: Any, path: str) -> None:
    import numpy as np

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
