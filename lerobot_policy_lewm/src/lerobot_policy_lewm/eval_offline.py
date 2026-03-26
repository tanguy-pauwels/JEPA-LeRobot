from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins


def _resolve_policy_path(policy_path: str) -> Path:
    """
    Resolve local policy path for eval.

    Supports:
    - exact local pretrained_model directory
    - run directory containing checkpoints/<step>/pretrained_model
    - legacy '/checkpoints/last/pretrained_model' hint by mapping to latest numeric checkpoint
    """
    p = Path(policy_path).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p

    if p.is_dir() and (p / "checkpoints").is_dir():
        # run dir
        checkpoints_dir = p / "checkpoints"
    elif p.is_dir() and p.name == "checkpoints":
        # checkpoints root dir
        checkpoints_dir = p
    elif p.is_dir() and p.parent.name == "checkpoints":
        # checkpoints/<step> or checkpoints/last
        if p.name.isdigit():
            candidate = p / "pretrained_model"
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return candidate
        checkpoints_dir = p.parent
    elif "checkpoints" in p.parts:
        # Handles missing ".../checkpoints/last/pretrained_model" (or similar hints) by backing up to checkpoints root.
        checkpoints_idx = p.parts.index("checkpoints")
        checkpoints_dir = Path(*p.parts[: checkpoints_idx + 1])
    else:
        return p

    step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        return p
    latest = max(step_dirs, key=lambda d: int(d.name))
    candidate = latest / "pretrained_model"
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate
    return p


def _final_action_target(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 3:
        return action[:, -1]
    if action.ndim == 2:
        return action
    raise ValueError(f"Unexpected action shape={tuple(action.shape)}")


def _to_cpu_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").float().numpy()


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std == 0.0 or b_std == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_per_dim_metrics(target: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    err = pred - target
    sq = err**2
    abs_err = np.abs(err)

    rows: list[dict[str, Any]] = []
    for dim in range(target.shape[1]):
        rows.append(
            {
                "dim": dim,
                "mse": float(np.mean(sq[:, dim])),
                "mae": float(np.mean(abs_err[:, dim])),
                "corr": _pearson_corr(target[:, dim], pred[:, dim]),
                "target_mean": float(np.mean(target[:, dim])),
                "target_std": float(np.std(target[:, dim])),
                "pred_mean": float(np.mean(pred[:, dim])),
                "pred_std": float(np.std(pred[:, dim])),
            }
        )
    return rows


def compute_per_episode_metrics(
    episode_ids: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
) -> list[dict[str, Any]]:
    err = pred - target
    sq = err**2
    abs_err = np.abs(err)

    bucket: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sq": 0.0, "abs": 0.0, "max_abs": 0.0})
    for idx in range(target.shape[0]):
        ep = int(episode_ids[idx])
        sample_sq = float(np.mean(sq[idx]))
        sample_abs = float(np.mean(abs_err[idx]))
        sample_max_abs = float(np.max(abs_err[idx]))

        bucket[ep]["count"] += 1
        bucket[ep]["sq"] += sample_sq
        bucket[ep]["abs"] += sample_abs
        bucket[ep]["max_abs"] = max(bucket[ep]["max_abs"], sample_max_abs)

    rows: list[dict[str, Any]] = []
    for ep in sorted(bucket):
        cnt = bucket[ep]["count"]
        rows.append(
            {
                "episode_index": ep,
                "num_samples": int(cnt),
                "mse": bucket[ep]["sq"] / cnt,
                "mae": bucket[ep]["abs"] / cnt,
                "max_abs_error": bucket[ep]["max_abs"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LeWM evaluation on local/Hub LeRobot dataset")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    register_third_party_plugins()
    resolved_policy_path = _resolve_policy_path(args.policy_path)
    if not resolved_policy_path.exists():
        raise FileNotFoundError(
            f"Policy path not found: {args.policy_path}\n"
            "Tip: pass a local directory containing config.json (usually .../checkpoints/<step>/pretrained_model)."
        )
    if resolved_policy_path.is_dir() and not (resolved_policy_path / "config.json").is_file():
        raise FileNotFoundError(
            f"config.json not found in: {resolved_policy_path}\n"
            "Tip: pass .../checkpoints/<step>/pretrained_model (or run dir with checkpoints)."
        )

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    cfg = PreTrainedConfig.from_pretrained(resolved_policy_path)
    cfg.pretrained_path = resolved_policy_path
    cfg.device = args.device

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    all_target: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_episode: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches:
                break

            policy_batch = {k: v for k, v in raw_batch.items() if torch.is_tensor(v)}
            batch = preprocessor(policy_batch)
            pred = policy.select_action(batch)
            pred = postprocessor(pred)

            target_tensor = _final_action_target(raw_batch[ACTION])
            pred_tensor = pred
            if pred_tensor.ndim == 3:
                pred_tensor = pred_tensor[:, -1]
            elif pred_tensor.ndim != 2:
                raise ValueError(f"Unexpected predicted action shape={tuple(pred_tensor.shape)}")

            target_np = _to_cpu_numpy(target_tensor)
            pred_np = _to_cpu_numpy(pred_tensor)
            ep_np = raw_batch.get("episode_index")
            if ep_np is None:
                ep_idx = np.full((target_np.shape[0],), -1, dtype=np.int64)
            else:
                ep_idx = _to_cpu_numpy(ep_np).astype(np.int64)

            all_target.append(target_np)
            all_pred.append(pred_np)
            all_episode.append(ep_idx)

            if len(sample_rows) < args.max_samples:
                room = args.max_samples - len(sample_rows)
                take_n = min(room, target_np.shape[0])
                for i in range(take_n):
                    err_vec = pred_np[i] - target_np[i]
                    for dim in range(target_np.shape[1]):
                        sample_rows.append(
                            {
                                "batch_index": batch_idx,
                                "sample_index_in_batch": i,
                                "episode_index": int(ep_idx[i]),
                                "dim": dim,
                                "target": float(target_np[i, dim]),
                                "pred": float(pred_np[i, dim]),
                                "error": float(err_vec[dim]),
                                "abs_error": float(abs(err_vec[dim])),
                            }
                        )

    if not all_target:
        raise RuntimeError("No samples processed. Check dataset and max-batches.")

    target = np.concatenate(all_target, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    episode_ids = np.concatenate(all_episode, axis=0)

    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    per_dim_rows = compute_per_dim_metrics(target, pred)
    per_episode_rows = compute_per_episode_metrics(episode_ids, target, pred)

    summary = {
        "policy_path": str(resolved_policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "device": args.device,
        "num_samples": int(target.shape[0]),
        "action_dim": int(target.shape[1]),
        "global_mse": mse,
        "global_mae": mae,
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "avg_l2_per_sample": float(np.mean(np.linalg.norm(diff, axis=1))),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "offline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_csv(out_dir / "per_dim_metrics.csv", per_dim_rows)
    _write_csv(out_dir / "per_episode_metrics.csv", per_episode_rows)
    _write_csv(out_dir / "samples_predictions.csv", sample_rows)

    print("[eval_offline] Wrote artifacts:")
    print(f"- {out_dir / 'offline_summary.json'}")
    print(f"- {out_dir / 'per_dim_metrics.csv'}")
    print(f"- {out_dir / 'per_episode_metrics.csv'}")
    print(f"- {out_dir / 'samples_predictions.csv'}")


if __name__ == "__main__":
    main()
