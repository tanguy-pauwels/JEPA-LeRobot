"""Shared validation helpers for LeRobot -> LeWM conversion scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_HDF5_KEYS = (
    "pixels",
    "action",
    "proprio",
    "state",
    "episode_idx",
    "step_idx",
    "done",
    "timestamp",
    "index",
    "task_index",
    "ep_len",
    "ep_offset",
)

_PER_STEP_KEYS = (
    "pixels",
    "action",
    "proprio",
    "state",
    "episode_idx",
    "step_idx",
    "done",
    "timestamp",
    "index",
    "task_index",
)


def sanitize_repo_id(repo_id: str) -> str:
    """Convert a HF repo id into a filesystem-safe name."""
    return repo_id.replace("/", "__")


def parse_episode_range(spec: str, total_episodes: int) -> list[int]:
    """Parse a LeRobot split string formatted as 'start:end'."""
    if ":" not in spec:
        raise ValueError(f"Unsupported split format '{spec}'. Expected 'start:end'.")
    start_str, end_str = spec.split(":", 1)
    start, end = int(start_str), int(end_str)
    if start < 0 or end < 0:
        raise ValueError(f"Split range must be positive, got '{spec}'.")
    if end < start:
        raise ValueError(f"Split range end must be >= start, got '{spec}'.")
    if end > total_episodes:
        raise ValueError(
            f"Split range '{spec}' exceeds total episodes ({total_episodes})."
        )
    return list(range(start, end))


def _as_1d_numpy(array_like: Any) -> np.ndarray:
    """Convert scalars/tensors/lists to a flat numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like.reshape(-1)
    if hasattr(array_like, "detach") and hasattr(array_like, "cpu"):
        return array_like.detach().cpu().numpy().reshape(-1)
    if isinstance(array_like, (list, tuple)):
        out = []
        for item in array_like:
            if hasattr(item, "item"):
                out.append(item.item())
            else:
                out.append(item)
        return np.asarray(out).reshape(-1)
    if hasattr(array_like, "item"):
        return np.asarray([array_like.item()])
    return np.asarray(array_like).reshape(-1)


def collect_source_episode_issues(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    done: np.ndarray,
    require_terminal_done: bool = True,
    require_no_early_done: bool = True,
    require_contiguous_steps: bool = True,
) -> dict[int, list[str]]:
    """Validate episode termination/step consistency on source LeRobot rows."""
    ep = _as_1d_numpy(episode_idx).astype(np.int64)
    step = _as_1d_numpy(step_idx).astype(np.int64)
    done_arr = _as_1d_numpy(done).astype(bool)

    if not (len(ep) == len(step) == len(done_arr)):
        return {
            -1: [
                "Column length mismatch between episode_index, frame_index and next.done."
            ]
        }
    if len(ep) == 0:
        return {}

    issues: dict[int, list[str]] = {}
    boundaries = np.where(np.diff(ep) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(ep)]))

    seen: set[int] = set()
    for start, end in zip(starts, ends):
        episode = int(ep[start])
        if episode in seen:
            issues.setdefault(episode, []).append("Episode rows are not contiguous.")
        seen.add(episode)

        step_slice = step[start:end]
        done_slice = done_arr[start:end]

        if len(step_slice) == 0:
            issues.setdefault(episode, []).append("Episode contains zero rows.")
            continue

        if require_contiguous_steps:
            if int(step_slice[0]) != 0:
                issues.setdefault(episode, []).append(
                    f"step_idx starts at {int(step_slice[0])}, expected 0."
                )
            expected = np.arange(int(step_slice[0]), int(step_slice[0]) + len(step_slice))
            if not np.array_equal(step_slice, expected):
                issues.setdefault(episode, []).append(
                    "step_idx is not contiguous within the episode."
                )

        if require_terminal_done and not bool(done_slice[-1]):
            issues.setdefault(episode, []).append(
                "Last step has next.done=False (episode not terminated cleanly)."
            )

        if require_no_early_done and np.any(done_slice[:-1]):
            issues.setdefault(episode, []).append(
                "Found next.done=True before the episode terminal step."
            )

    return issues


def list_hdf5_files(target_path: Path) -> list[Path]:
    """Resolve a list of .h5 files from a file or directory target."""
    if target_path.is_file():
        return [target_path] if target_path.suffix == ".h5" else []
    if not target_path.exists():
        return []
    return sorted(p for p in target_path.rglob("*.h5") if p.is_file())


def inspect_hdf5_file(path: Path) -> dict[str, Any]:
    """Return a compact schema/shape summary for one HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 inspection.") from exc

    summary: dict[str, Any] = {"file": str(path), "datasets": {}}
    with h5py.File(path, "r") as h5f:
        for key in sorted(h5f.keys()):
            ds = h5f[key]
            summary["datasets"][key] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
            }

        if "ep_len" in h5f:
            ep_len = np.asarray(h5f["ep_len"][:], dtype=np.int64)
            summary["num_episodes"] = int(len(ep_len))
            summary["total_steps"] = int(ep_len.sum())
        elif "episode_idx" in h5f:
            summary["num_episodes"] = int(len(np.unique(h5f["episode_idx"][:])))
            summary["total_steps"] = int(h5f["episode_idx"].shape[0])
        else:
            summary["num_episodes"] = None
            summary["total_steps"] = None

    return summary


def validate_hdf5_file(path: Path, strict_done: bool = True) -> list[str]:
    """Return a list of validation errors for one HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 validation.") from exc

    errors: list[str] = []
    with h5py.File(path, "r") as h5f:
        missing = [k for k in REQUIRED_HDF5_KEYS if k not in h5f]
        if missing:
            errors.append(f"Missing required keys: {missing}")
            return errors

        lengths = {k: int(h5f[k].shape[0]) for k in _PER_STEP_KEYS}
        n_rows = lengths["episode_idx"]
        bad_lengths = {k: v for k, v in lengths.items() if v != n_rows}
        if bad_lengths:
            errors.append(
                "Per-step dataset lengths mismatch against episode_idx: "
                f"{bad_lengths}"
            )

        pixels = h5f["pixels"]
        if pixels.ndim != 4:
            errors.append(f"'pixels' must be rank-4, got rank-{pixels.ndim}.")
        else:
            channels = int(pixels.shape[-1])
            if channels not in (1, 3):
                errors.append(
                    f"'pixels' last dimension must be 1 or 3, got {channels}."
                )

        ep_len = np.asarray(h5f["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5f["ep_offset"][:], dtype=np.int64)
        episode_idx = np.asarray(h5f["episode_idx"][:], dtype=np.int64)
        step_idx = np.asarray(h5f["step_idx"][:], dtype=np.int64)
        done = np.asarray(h5f["done"][:], dtype=bool)

        if ep_len.ndim != 1 or ep_offset.ndim != 1:
            errors.append("'ep_len' and 'ep_offset' must be rank-1 arrays.")
            return errors

        if len(ep_len) != len(ep_offset):
            errors.append(
                f"'ep_len' and 'ep_offset' length mismatch: {len(ep_len)} vs {len(ep_offset)}."
            )

        if len(ep_offset) > 0 and int(ep_offset[0]) != 0:
            errors.append(f"First ep_offset must be 0, got {int(ep_offset[0])}.")

        if np.any(ep_len <= 0):
            errors.append("All ep_len values must be > 0.")

        if np.any(np.diff(ep_offset) < 0):
            errors.append("ep_offset must be non-decreasing.")

        total_from_meta = int(ep_len.sum()) if len(ep_len) else 0
        if total_from_meta != n_rows:
            errors.append(
                f"Total rows mismatch: sum(ep_len)={total_from_meta}, rows={n_rows}."
            )

        for idx, (start, length) in enumerate(zip(ep_offset, ep_len)):
            start_i = int(start)
            length_i = int(length)
            end_i = start_i + length_i

            if start_i < 0 or end_i > n_rows:
                errors.append(
                    f"Episode {idx} has invalid bounds [{start_i}, {end_i}) for {n_rows} rows."
                )
                continue

            step_slice = step_idx[start_i:end_i]
            done_slice = done[start_i:end_i]
            ep_slice = episode_idx[start_i:end_i]

            if len(step_slice) != length_i:
                errors.append(
                    f"Episode {idx} expected {length_i} rows, found {len(step_slice)}."
                )
                continue

            expected_step = np.arange(length_i, dtype=np.int64)
            if not np.array_equal(step_slice, expected_step):
                errors.append(
                    f"Episode {idx} step_idx is not contiguous from 0 to {length_i - 1}."
                )

            if len(np.unique(ep_slice)) != 1:
                errors.append(f"Episode slice {idx} mixes multiple episode_idx values.")

            if strict_done:
                if not bool(done_slice[-1]):
                    errors.append(
                        f"Episode {idx} terminal step has done=False (not cleanly terminated)."
                    )
                if np.any(done_slice[:-1]):
                    errors.append(
                        f"Episode {idx} has done=True before terminal step."
                    )

    return errors

