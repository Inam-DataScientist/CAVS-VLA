"""Training (single GPU or torchrun DDP) and counterexample-guided refinement."""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .config import Config, config_from_dict, dump_json
from .data.dataset import RawSceneDataset, WeightedIndexSampler
from .features import featurize, scene_to_device
from .model.vla import VLAPolicy, policy_losses, proximity_penalty
from .utils import environment_info, get_logger, seed_everything, sha256_file


def _ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _is_main() -> bool:
    return _rank() == 0


def amp_dtype(cfg: Config) -> Optional[torch.dtype]:
    if cfg.train.amp == "bf16":
        return torch.bfloat16
    if cfg.train.amp == "fp16":
        return torch.float16
    return None


def autocast_ctx(device: torch.device, dtype: Optional[torch.dtype]):
    """bf16/fp16 autocast on CUDA, a no-op elsewhere."""
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model(cfg: Config, device: torch.device) -> VLAPolicy:
    return VLAPolicy(cfg).to(device)


def save_checkpoint(path: str, model, opt, sched, epoch: int, step: int, cfg: Config, extra: Dict) -> None:
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    tmp = path + ".tmp"
    torch.save(dict(model=raw.state_dict(), optimizer=opt.state_dict() if opt else None,
                    scheduler=sched.state_dict() if sched else None, epoch=epoch, step=step,
                    config=cfg.to_dict(), extra=extra, environment=environment_info()), tmp)
    os.replace(tmp, path)


def load_policy(path: str, device: torch.device, cfg_override: Optional[Config] = None):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg_override or config_from_dict(ck["config"])
    model = VLAPolicy(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def _param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in n or "query" in n or "tok_pos" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def _schedule(opt, warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate_loader(model, loader, cfg: Config, device, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    K = cfg.data.hist_steps
    sums: Dict[str, float] = {}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        scene = scene_to_device(batch, device)
        B = scene["traj"].shape[0]
        actor = torch.zeros(B, dtype=torch.long, device=device)
        feat = featurize(scene, K - 1, actor, cfg, train=False, with_targets=True)
        with autocast_ctx(device, amp_dtype(cfg)):
            out = model(feat)
        out = {k: v.float() if torch.is_floating_point(v) else v for k, v in out.items()}
        loss, logs = policy_losses(out, feat, cfg)
        logs["loss"] = float(loss)
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v * B
        n += B
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(cfg: Config, run_name: str = "run", init_ckpt: str = "", sampler_weights: Optional[np.ndarray] = None,
          epochs: Optional[int] = None, lr: Optional[float] = None, w_safe: Optional[float] = None,
          hj_table=None) -> str:
    tc = cfg.train
    if _ddp() and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(tc.seed + _rank())
    out_dir = os.path.join(tc.ckpt_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger("train", os.path.join(out_dir, "train.log") if _is_main() else None)
    ds_tr = RawSceneDataset(cfg, "train")
    ds_va = RawSceneDataset(cfg, "val")
    if len(ds_tr) == 0:
        raise RuntimeError("Empty training split")
    if sampler_weights is not None:
        world = dist.get_world_size() if dist.is_initialized() else 1
        sampler = WeightedIndexSampler(sampler_weights, len(ds_tr) // world, seed=tc.seed + 1000 * _rank())
    elif _ddp():
        sampler = DistributedSampler(ds_tr, shuffle=True, seed=tc.seed, drop_last=True)
    else:
        sampler = RandomSampler(ds_tr)
    kw = dict(num_workers=tc.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=tc.num_workers > 0)
    dl_tr = DataLoader(ds_tr, batch_size=tc.batch_size, sampler=sampler, drop_last=len(ds_tr) >= tc.batch_size, **kw)
    dl_va = DataLoader(ds_va, batch_size=tc.batch_size, sampler=SequentialSampler(ds_va), **kw) if len(ds_va) else None
    model = build_model(cfg, device)
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        log.info(f"initialised from {init_ckpt}")
    n_params = sum(p.numel() for p in model.parameters())
    if tc.compile:
        model = torch.compile(model)
    if _ddp():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    n_epochs = epochs if epochs is not None else tc.epochs
    base_lr = lr if lr is not None else tc.lr
    opt = torch.optim.AdamW(_param_groups(model, tc.weight_decay), lr=base_lr, betas=(0.9, 0.98))
    steps_per_epoch = len(dl_tr) if not tc.max_train_batches else min(len(dl_tr), tc.max_train_batches)
    sched = _schedule(opt, tc.warmup_steps if not init_ckpt else 0, n_epochs * max(steps_per_epoch, 1))
    dtype = amp_dtype(cfg)
    use_amp = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    start_epoch, step = 0, 0
    if tc.resume:
        ck = torch.load(tc.resume, map_location=device, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        log.info(f"resumed from {tc.resume} at epoch {start_epoch}")
    ws = tc.w_safe if w_safe is None else w_safe
    gen = torch.Generator(device=device).manual_seed(tc.seed + 17 * _rank())
    K = cfg.data.hist_steps
    manifest = os.path.join(cfg.data.out_dir, "manifest.json")
    meta = dict(params=n_params, manifest_sha256=sha256_file(manifest) if os.path.exists(manifest) else None,
                train_size=len(ds_tr), val_size=len(ds_va), run=run_name)
    if _is_main():
        with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
            f.write(cfg.dumps())
        dump_json(dict(meta, environment=environment_info()), os.path.join(out_dir, "run_meta.json"))
        log.info(f"params={n_params:,} train={len(ds_tr)} val={len(ds_va)} device={device} amp={tc.amp}")
    history: List[Dict] = []
    best = float("inf")
    for epoch in range(start_epoch, n_epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        agg: Dict[str, float] = {}
        nb = 0
        for bi, batch in enumerate(dl_tr):
            if tc.max_train_batches and bi >= tc.max_train_batches:
                break
            scene = scene_to_device(batch, device)
            B = scene["traj"].shape[0]
            actor = torch.zeros(B, dtype=torch.long, device=device)
            feat = featurize(scene, K - 1, actor, cfg, train=True, with_targets=True, generator=gen)
            with autocast_ctx(device, dtype):
                out = model(feat)
            out = {k: v.float() if torch.is_floating_point(v) else v for k, v in out.items()}
            pen = ws * proximity_penalty(out, feat, cfg, hj_table) if ws > 0 else None
            loss, logs = policy_losses(out, feat, cfg, safety_penalty=pen)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}: {logs}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            nb += 1
            logs["loss"] = float(loss)
            logs["grad_norm"] = float(gnorm)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            if _is_main() and step % tc.log_every == 0:
                log.info(f"ep {epoch} step {step} " + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                         + f" lr={sched.get_last_lr()[0]:.2e}")
        rec = {f"train_{k}": v / max(nb, 1) for k, v in agg.items()}
        rec.update(epoch=epoch, step=step, seconds=time.time() - t0)
        if dl_va is not None and (epoch + 1) % tc.val_every == 0:
            raw = model.module if hasattr(model, "module") else model
            val = evaluate_loader(raw, dl_va, cfg, device, tc.max_val_batches)
            rec.update({f"val_{k}": v for k, v in val.items()})
        score = rec.get("val_minADE", rec.get("train_minADE", rec["train_loss"]))
        history.append(rec)
        if _is_main():
            log.info(f"epoch {epoch} done: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                                       if isinstance(v, float)))
            save_checkpoint(os.path.join(out_dir, "last.pt"), model, opt, sched, epoch, step, cfg, dict(meta, **rec))
            if score < best:
                best = score
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, opt, sched, epoch, step, cfg,
                                dict(meta, **rec, selection="val_minADE" if dl_va is not None else "train_minADE"))
            dump_json(history, os.path.join(out_dir, "training_history.json"))
    if _ddp():
        dist.barrier()
    return os.path.join(out_dir, "best.pt")


def ceg_refine(cfg: Config, init_ckpt: str, failures_path: str, run_name: str, hj_table=None) -> str:
    """Counterexample-guided refinement: up-weight failing training scenes and add the safety penalty.

    ``failures_path`` is written by ``mine-failures`` and lists *training-split* scene indices only,
    so no validation/test scenario influences the refined policy.
    """
    with open(failures_path) as f:
        fails = json.load(f)
    if fails.get("split") != "train":
        raise ValueError("CEG refinement must mine failures on the training split only")
    ds = RawSceneDataset(cfg, "train")
    w = np.ones(len(ds), dtype=np.float64)
    for idx in fails["indices"]:
        if 0 <= idx < len(w):
            w[idx] = cfg.train.ceg_hard_weight
    return train(cfg, run_name=run_name, init_ckpt=init_ckpt, sampler_weights=w, epochs=cfg.train.ceg_epochs,
                 lr=cfg.train.ceg_lr, w_safe=cfg.train.ceg_w_safe, hj_table=hj_table)
