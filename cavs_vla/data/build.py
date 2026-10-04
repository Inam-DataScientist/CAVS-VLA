"""Dataset build: logs/files -> per-log staging chunks -> merged memory-mapped splits.

Layout of ``data.out_dir``::

    staging/<split>/<log>.npz      one chunk per nuPlan log / WOMD file
    <split>/<field>.npy            merged memory-mapped arrays (N, ...)
    <split>/logs.json              log name of every window (for log-level analyses)
    manifest.json                  config, counts, route sources, hashes, environment

Splits are assigned per log (nuPlan) or per file (WOMD), never per window.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config, config_from_dict, dump_json
from ..utils import environment_info, get_logger, sha256_file, stable_hash01
from .scene import scene_field_specs, stack_scenes

SPLITS = ("train", "val", "test")


def assign_split(name: str, cfg: Config, split_map: Optional[Dict[str, str]] = None) -> str:
    if split_map is not None:
        if name not in split_map:
            raise KeyError(f"Log '{name}' missing from split file")
        sp = split_map[name]
        if sp not in SPLITS:
            raise ValueError(f"Invalid split '{sp}' for {name}")
        return sp
    low = name.lower()
    # Respect official WOMD folder names when present.
    for tag, sp in (("training", "train"), ("validation", "val"), ("testing", "test")):
        if f"/{tag}" in low or low.startswith(tag):
            return sp
    u = stable_hash01(os.path.basename(name), cfg.data.split_seed)
    r = cfg.data.split_ratios
    return "train" if u < r[0] else ("val" if u < r[0] + r[1] else "test")


def _process_nuplan(cfg_dict: dict, db_path: str, out_path: str) -> dict:
    from .maps import make_map_source
    from .nuplan import build_log_scenes

    cfg = config_from_dict(cfg_dict)
    d = cfg.data
    backend = d.map_backend
    map_src = make_map_source(backend, d.nuplan_map_root, d.nuplan_map_version, d.json_map_path)
    scenes, meta = build_log_scenes(cfg, db_path, map_src)
    if scenes:
        np.savez(out_path, **stack_scenes(scenes))
    meta["chunk"] = out_path if scenes else None
    meta["map_backend"] = map_src.backend
    return meta


def _process_waymo(cfg_dict: dict, path: str, out_path: str) -> dict:
    from .waymo import build_file_scenes

    cfg = config_from_dict(cfg_dict)
    scenes, meta = build_file_scenes(cfg, path)
    if scenes:
        np.savez(out_path, **stack_scenes(scenes))
    meta["chunk"] = out_path if scenes else None
    return meta


def discover_inputs(cfg: Config) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    d = cfg.data
    if "nuplan" in d.sources:
        if not d.nuplan_db_root:
            raise ValueError("data.sources contains nuplan but data.nuplan_db_root is empty")
        dbs = sorted(glob.glob(os.path.join(d.nuplan_db_root, "**", "*.db"), recursive=True))
        if not dbs:
            raise FileNotFoundError(f"No .db files under {d.nuplan_db_root}")
        items += [("nuplan", p) for p in dbs]
    if "waymo" in d.sources:
        if not d.waymo_root:
            raise ValueError("data.sources contains waymo but data.waymo_root is empty")
        files = sorted(glob.glob(os.path.join(d.waymo_root, d.waymo_file_glob), recursive=True))
        if d.waymo_max_files:
            files = files[: d.waymo_max_files]
        if not files:
            raise FileNotFoundError(f"No WOMD files matching {d.waymo_file_glob} under {d.waymo_root}")
        items += [("waymo", p) for p in files]
    unknown = set(d.sources) - {"nuplan", "waymo"}
    if unknown:
        raise ValueError(f"Unknown data sources {sorted(unknown)}")
    return items


def build_dataset(cfg: Config, overwrite: bool = False) -> dict:
    log = get_logger("build")
    d = cfg.data
    out = d.out_dir
    if os.path.exists(os.path.join(out, "manifest.json")) and not overwrite:
        raise FileExistsError(f"{out} already holds a built dataset; pass --overwrite to rebuild")
    staging = os.path.join(out, "staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    for sp in SPLITS:
        os.makedirs(os.path.join(staging, sp), exist_ok=True)
    split_map = None
    if d.split_file:
        with open(d.split_file) as f:
            split_map = json.load(f)
    items = discover_inputs(cfg)
    jobs = []
    for kind, path in items:
        name = os.path.splitext(os.path.basename(path))[0]
        rel = os.path.relpath(path, d.nuplan_db_root if kind == "nuplan" else d.waymo_root)
        sp = assign_split(name if kind == "nuplan" else rel, cfg, split_map)
        out_path = os.path.join(staging, sp, f"{kind}__{name}.npz")
        jobs.append((kind, path, sp, out_path))
    log.info(f"{len(jobs)} inputs: " + ", ".join(f"{sp}={sum(1 for j in jobs if j[2] == sp)}" for sp in SPLITS))
    cfg_dict = cfg.to_dict()
    metas: List[dict] = []
    failures: List[dict] = []
    workers = max(1, int(d.num_workers))
    if workers == 1:
        for kind, path, sp, out_path in jobs:
            try:
                fn = _process_nuplan if kind == "nuplan" else _process_waymo
                m = fn(cfg_dict, path, out_path)
                m.update(split=sp, source=kind)
                metas.append(m)
                log.info(f"[{sp}] {os.path.basename(path)}: {m['windows']} windows")
            except Exception as exc:
                failures.append(dict(path=path, error=repr(exc), trace=traceback.format_exc()))
                log.error(f"FAILED {path}: {exc!r}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for kind, path, sp, out_path in jobs:
                fn = _process_nuplan if kind == "nuplan" else _process_waymo
                futs[ex.submit(fn, cfg_dict, path, out_path)] = (kind, path, sp)
            for fut in as_completed(futs):
                kind, path, sp = futs[fut]
                try:
                    m = fut.result()
                    m.update(split=sp, source=kind)
                    metas.append(m)
                    log.info(f"[{sp}] {os.path.basename(path)}: {m['windows']} windows")
                except Exception as exc:
                    failures.append(dict(path=path, error=repr(exc)))
                    log.error(f"FAILED {path}: {exc!r}")
    if failures:
        dump_json(failures, os.path.join(out, "build_failures.json"))
        if len(failures) == len(jobs):
            raise RuntimeError(f"All {len(jobs)} inputs failed; see {out}/build_failures.json")
        log.warning(f"{len(failures)} inputs failed (listed in build_failures.json); they are excluded")
    counts = {}
    hashes = {}
    for sp in SPLITS:
        chunks = sorted(m["chunk"] for m in metas if m["split"] == sp and m.get("chunk"))
        counts[sp] = merge_split(cfg, chunks, os.path.join(out, sp))
        if d.hash_outputs and counts[sp]:
            hashes[sp] = {f: sha256_file(os.path.join(out, sp, f"{f}.npy")) for f in scene_field_specs(cfg)}
    # Split integrity: no log may appear in two splits.
    logs_by_split = {sp: {m["log"] for m in metas if m["split"] == sp} for sp in SPLITS}
    inter = {f"{a}&{b}": len(logs_by_split[a] & logs_by_split[b])
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(inter.values()):
        raise RuntimeError(f"Split leakage detected: {inter}")
    manifest = dict(config=cfg_dict, counts=counts, logs=sorted(metas, key=lambda m: m["log"]),
                    split_log_intersections=inter, hashes=hashes, environment=environment_info(),
                    route_sources=sorted({m.get("route_source", "none") for m in metas}),
                    failures=len(failures))
    dump_json(manifest, os.path.join(out, "manifest.json"))
    shutil.rmtree(staging)
    log.info(f"Dataset built at {out}: {counts}")
    return manifest


def merge_split(cfg: Config, chunk_paths: List[str], out_dir: str) -> int:
    specs = scene_field_specs(cfg)
    os.makedirs(out_dir, exist_ok=True)
    sizes = []
    for p in chunk_paths:
        with np.load(p) as z:
            sizes.append(int(z["traj"].shape[0]))
    total = int(sum(sizes))
    names: List[str] = []
    if total == 0:
        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump(dict(n=0, fields=list(specs)), f)
        with open(os.path.join(out_dir, "logs.json"), "w") as f:
            json.dump([], f)
        return 0
    mms = {k: np.lib.format.open_memmap(os.path.join(out_dir, f"{k}.npy"), mode="w+", dtype=dt,
                                        shape=(total,) + shape) for k, (shape, dt) in specs.items()}
    pos = 0
    for p, n in zip(chunk_paths, sizes):
        with np.load(p) as z:
            for k in specs:
                mms[k][pos:pos + n] = z[k]
        names += [os.path.basename(p)[:-4]] * n
        pos += n
    for m in mms.values():
        m.flush()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(dict(n=total, fields=list(specs)), f)
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(names, f)
    return total


class RawSceneStore:
    """Read-only access to a merged split (memory-mapped)."""

    def __init__(self, cfg: Config, split_dir: str) -> None:
        with open(os.path.join(split_dir, "index.json")) as f:
            idx = json.load(f)
        self.n = int(idx["n"])
        self.specs = scene_field_specs(cfg)
        self.arrays = {}
        if self.n:
            for k, (shape, dt) in self.specs.items():
                arr = np.load(os.path.join(split_dir, f"{k}.npy"), mmap_mode="r")
                if arr.shape != (self.n,) + shape:
                    raise ValueError(f"{split_dir}/{k}.npy has shape {arr.shape}, expected {(self.n,) + shape}. "
                                     f"Was the dataset built with a different config?")
                self.arrays[k] = arr
        with open(os.path.join(split_dir, "logs.json")) as f:
            self.logs = json.load(f)

    def __len__(self) -> int:
        return self.n

    def get(self, i: int) -> Dict[str, np.ndarray]:
        return {k: np.array(v[i]) for k, v in self.arrays.items()}

    def get_batch(self, idx: List[int]) -> Dict[str, np.ndarray]:
        idx_sorted = np.asarray(idx)
        return {k: np.asarray(v[idx_sorted]) for k, v in self.arrays.items()}
