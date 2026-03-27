#!/usr/bin/env python3
"""Convert LeRobot datasets (parquet+mp4) to LeWM-compatible HDF5 files."""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import ctypes
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import draccus
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_dataset_utils import (  # noqa: E402
    collect_source_episode_issues,
    parse_episode_range,
    sanitize_repo_id,
)

DEFAULT_DATASET = "lerobot/koch_pick_place_1_lego"
VALID_DIRTY_POLICY = ("fail", "drop", "warn")
VALID_COMPRESSION = ("none", "lzf", "gzip")
VALID_MEMORY_GUARD_MODE = ("off", "warn", "error")
TABULAR_COLUMNS = [
    "action",
    "observation.state",
    "episode_index",
    "frame_index",
    "next.done",
    "timestamp",
    "index",
    "task_index",
]


@dataclass
class ConvertConfig:
    repo_id: str = DEFAULT_DATASET
    datasets_dir: str = "datasets"
    raw_subdir: str = "raw"
    hdf5_subdir: str = "hdf5"
    splits: list[str] = field(default_factory=list)
    camera_keys: list[str] = field(default_factory=list)
    image_size: int | None = None
    compression: str = "lzf"
    overwrite: bool = False
    require_terminal_done: bool = True
    dirty_episode_policy: str = "fail"
    report_filename: str = "conversion_report.json"
    video_backend: str | None = None
    progress_every: int = 500
    heartbeat_seconds: float = 10.0
    num_workers: int = 0
    auto_max_workers: int = 2
    episode_batch_size: int = 16
    max_pending_tasks: int = 0
    memory_guard_mode: str = "warn"
    max_inflight_memory_ratio: float = 0.40
    worker_memory_buffer_mb: int = 256
    auto_install_torch: bool = True


@dataclass
class EpisodeWorkItem:
    episode_index: int
    row_start: int
    row_stop: int
    camera_keys: list[str]
    video_paths: dict[str, str]
    from_timestamps_s: dict[str, float]
    timestamps_s: np.ndarray
    image_size: int | None
    preferred_backend: str | None
    tolerance_s: float


@dataclass
class EpisodeWorkResult:
    episode_index: int
    row_start: int
    row_stop: int
    pixels_by_camera: dict[str, np.ndarray]
    backend_used: dict[str, str]


def _split_episodes(meta: Any, split_name: str) -> list[int]:
    split_spec = meta.info["splits"][split_name]
    return parse_episode_range(split_spec, total_episodes=meta.total_episodes)


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _column_to_numpy(dataset: Any, key: str) -> np.ndarray:
    try:
        col = dataset[key]
    except Exception:
        col = dataset[:][key]
    if isinstance(col, np.ndarray):
        return col.reshape(-1)
    if hasattr(col, "detach") and hasattr(col, "cpu"):
        return col.detach().cpu().numpy().reshape(-1)

    out = []
    for item in col:
        out.append(_to_scalar(item))
    return np.asarray(out).reshape(-1)


def _select_columns_compat(dataset_obj: Any, columns: list[str]) -> Any:
    """Compatibility helper for LeRobotDataset versions with/without select_columns."""
    if hasattr(dataset_obj, "select_columns"):
        return dataset_obj.select_columns(columns)
    if hasattr(dataset_obj, "hf_dataset"):
        return dataset_obj.hf_dataset.select_columns(columns)
    raise AttributeError("Could not access a select_columns-compatible API.")


def _get_episode_row(episodes_table: Any, ep_idx: int) -> dict[str, Any]:
    """Return one episode metadata row for both pandas and HF datasets backends."""
    if hasattr(episodes_table, "iloc"):
        row = episodes_table.iloc[ep_idx]
        if hasattr(row, "to_dict"):
            return row.to_dict()
        return dict(row)
    row = episodes_table[int(ep_idx)]
    if isinstance(row, dict):
        return row
    return dict(row)


def _episodes_lengths(meta: Any, episodes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lengths = []
    episodes_table = meta.episodes
    for ep_idx in episodes:
        row = _get_episode_row(episodes_table, ep_idx)
        if "dataset_from_index" in row and "dataset_to_index" in row:
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            lengths.append(end - start)
        elif "length" in row:
            lengths.append(int(row["length"]))
        else:
            raise KeyError(
                "Episode metadata row is missing both "
                "'dataset_from_index/dataset_to_index' and 'length'."
            )
    ep_len = np.asarray(lengths, dtype=np.int64)
    ep_offset = np.zeros_like(ep_len)
    if len(ep_offset) > 1:
        ep_offset[1:] = np.cumsum(ep_len[:-1], dtype=np.int64)
    return ep_len, ep_offset


def _get_compression(compression: str) -> str | None:
    if compression == "none":
        return None
    return compression


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prevalidate_source_split(
    repo_id: str,
    root: Path,
    episodes: list[int],
    require_terminal_done: bool,
) -> dict[int, list[str]]:
    source_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        download_videos=False,
    )
    tiny = _select_columns_compat(
        source_dataset, ["episode_index", "frame_index", "next.done"]
    )
    episode_idx = _column_to_numpy(tiny, "episode_index")
    step_idx = _column_to_numpy(tiny, "frame_index")
    done = _column_to_numpy(tiny, "next.done")

    return collect_source_episode_issues(
        episode_idx=episode_idx,
        step_idx=step_idx,
        done=done,
        require_terminal_done=require_terminal_done,
        require_no_early_done=True,
        require_contiguous_steps=True,
    )


def _slice_column_values(dataset_obj: Any, key: str, row_start: int, row_stop: int) -> Any:
    try:
        col = dataset_obj[key]
        return col[row_start:row_stop]
    except Exception:
        rows = dataset_obj[row_start:row_stop]
        if isinstance(rows, dict):
            return rows[key]
        return rows[key]


def _values_to_vector(values: Any, dtype: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.reshape(-1)
    elif hasattr(values, "detach") and hasattr(values, "cpu"):
        arr = values.detach().cpu().numpy().reshape(-1)
    else:
        out = []
        for item in values:
            out.append(_to_scalar(item))
        arr = np.asarray(out).reshape(-1)
    return arr.astype(dtype, copy=False)


def _values_to_matrix(values: Any, dtype: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values
    elif hasattr(values, "detach") and hasattr(values, "cpu"):
        arr = values.detach().cpu().numpy()
    else:
        rows = []
        for item in values:
            if isinstance(item, np.ndarray):
                rows.append(item.reshape(-1))
            elif hasattr(item, "detach") and hasattr(item, "cpu"):
                rows.append(item.detach().cpu().numpy().reshape(-1))
            else:
                rows.append(np.asarray(item).reshape(-1))
        arr = np.stack(rows, axis=0)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(dtype, copy=False)


def _read_episode_tabular_slice(
    columns_ds: Any, row_start: int, row_stop: int
) -> dict[str, np.ndarray]:
    return {
        "action": _values_to_matrix(
            _slice_column_values(columns_ds, "action", row_start, row_stop),
            np.float32,
        ),
        "state": _values_to_matrix(
            _slice_column_values(columns_ds, "observation.state", row_start, row_stop),
            np.float32,
        ),
        "episode_idx": _values_to_vector(
            _slice_column_values(columns_ds, "episode_index", row_start, row_stop),
            np.int64,
        ),
        "step_idx": _values_to_vector(
            _slice_column_values(columns_ds, "frame_index", row_start, row_stop),
            np.int64,
        ),
        "done": _values_to_vector(
            _slice_column_values(columns_ds, "next.done", row_start, row_stop),
            np.bool_,
        ),
        "timestamp": _values_to_vector(
            _slice_column_values(columns_ds, "timestamp", row_start, row_stop),
            np.float32,
        ),
        "index": _values_to_vector(
            _slice_column_values(columns_ds, "index", row_start, row_stop),
            np.int64,
        ),
        "task_index": _values_to_vector(
            _slice_column_values(columns_ds, "task_index", row_start, row_stop),
            np.int64,
        ),
    }


def _nearest_indices(frame_ts_s: np.ndarray, query_ts_s: np.ndarray) -> np.ndarray:
    if frame_ts_s.ndim != 1 or len(frame_ts_s) == 0:
        raise ValueError("frame_ts_s must be a non-empty 1D array.")
    ts = frame_ts_s.astype(np.float64, copy=False)
    q = query_ts_s.astype(np.float64, copy=False)
    ts = np.maximum.accumulate(ts)
    idx = np.searchsorted(ts, q, side="left")
    idx = np.clip(idx, 0, len(ts) - 1)
    prev_idx = np.clip(idx - 1, 0, len(ts) - 1)
    pick_prev = np.abs(q - ts[prev_idx]) <= np.abs(ts[idx] - q)
    idx[pick_prev] = prev_idx[pick_prev]
    return idx.astype(np.int64, copy=False)


def _indices_from_fps(
    query_ts_s: np.ndarray, fps: float, frame_count: int
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0.")
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    idx = np.rint(query_ts_s.astype(np.float64) * fps).astype(np.int64)
    return np.clip(idx, 0, frame_count - 1)


def _normalize_decoded_frames(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise RuntimeError(f"Decoded frames have unexpected shape: {frames.shape}")
    if frames.shape[-1] in (1, 3):
        out = frames
    elif frames.shape[1] in (1, 3):
        out = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise RuntimeError(f"Unexpected channel layout in decoded frames: {frames.shape}")

    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating):
            out = (out * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out = out.clip(0, 255).astype(np.uint8)
    return out


def _decode_with_decord(video_path: Path, query_ts_s: np.ndarray) -> np.ndarray:
    import decord

    vr = decord.VideoReader(str(video_path))
    frame_count = len(vr)
    if frame_count <= 0:
        raise RuntimeError(f"decord returned empty video for {video_path}.")

    all_idx = np.arange(frame_count, dtype=np.int64)
    all_frames = vr.get_batch(all_idx.tolist()).asnumpy()
    all_frames = _normalize_decoded_frames(all_frames)

    frame_ts_s: np.ndarray | None = None
    if hasattr(vr, "get_frame_timestamp"):
        try:
            ts = np.asarray(vr.get_frame_timestamp(all_idx.tolist()))
            if ts.ndim == 2:
                ts = ts[:, 0]
            if ts.ndim == 1 and len(ts) == frame_count:
                finite = np.isfinite(ts)
                if finite.all():
                    frame_ts_s = ts.astype(np.float64, copy=False)
        except Exception:
            frame_ts_s = None

    if frame_ts_s is not None:
        select_idx = _nearest_indices(frame_ts_s, query_ts_s)
    else:
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
        if fps <= 0:
            raise RuntimeError("decord could not infer FPS and timestamps are unavailable.")
        select_idx = _indices_from_fps(query_ts_s, fps=fps, frame_count=frame_count)

    return all_frames[select_idx]


def _decode_with_pyav(video_path: Path, query_ts_s: np.ndarray) -> np.ndarray:
    import av

    decoded_frames: list[np.ndarray] = []
    frame_ts_s: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        avg_rate = float(stream.average_rate) if stream.average_rate else 0.0
        fallback_fps = avg_rate if avg_rate > 0 else 30.0
        for i, frame in enumerate(container.decode(stream)):
            arr = frame.to_ndarray(format="rgb24")
            decoded_frames.append(arr)
            if frame.time is not None:
                frame_ts_s.append(float(frame.time))
            elif frame.pts is not None and stream.time_base is not None:
                frame_ts_s.append(float(frame.pts * stream.time_base))
            else:
                frame_ts_s.append(i / fallback_fps)

    if not decoded_frames:
        raise RuntimeError(f"PyAV returned empty video for {video_path}.")

    frames_np = np.stack(decoded_frames, axis=0)
    frames_np = _normalize_decoded_frames(frames_np)
    ts_np = np.asarray(frame_ts_s, dtype=np.float64)
    select_idx = _nearest_indices(ts_np, query_ts_s)
    return frames_np[select_idx]


def _decode_with_lerobot_seek(
    video_path: Path, query_ts_s: np.ndarray, tolerance_s: float, backend: str | None
) -> np.ndarray:
    frames = decode_video_frames(
        video_path=video_path,
        timestamps=query_ts_s.astype(np.float64).tolist(),
        tolerance_s=tolerance_s,
        backend=backend,
    )
    if hasattr(frames, "detach") and hasattr(frames, "cpu"):
        frames_np = frames.detach().cpu().numpy()
    else:
        frames_np = np.asarray(frames)
    return _normalize_decoded_frames(frames_np)


def _decode_frames_linear(
    video_path: Path,
    query_ts_s: np.ndarray,
    preferred_backend: str | None,
    tolerance_s: float,
) -> tuple[np.ndarray, str]:
    pref = (preferred_backend or "").strip().lower()
    if pref in ("", "auto", "default"):
        candidates = ["decord", "pyav", "lerobot"]
    elif pref in ("pyav", "av"):
        candidates = ["pyav", "decord", "lerobot"]
    elif pref == "decord":
        candidates = ["decord", "pyav", "lerobot"]
    else:
        candidates = [pref, "decord", "pyav", "lerobot"]

    errors: list[str] = []
    for backend in candidates:
        try:
            if backend == "decord":
                return _decode_with_decord(video_path, query_ts_s), "decord"
            if backend in ("pyav", "av"):
                return _decode_with_pyav(video_path, query_ts_s), "pyav"
            return (
                _decode_with_lerobot_seek(
                    video_path=video_path,
                    query_ts_s=query_ts_s,
                    tolerance_s=tolerance_s,
                    backend=preferred_backend,
                ),
                "lerobot_seek",
            )
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(
        f"Unable to decode {video_path} with backends {candidates}. Errors={errors}"
    )


def _ensure_torch_installed(auto_install: bool) -> None:
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        if not auto_install:
            raise

    cmd = [sys.executable, "-m", "pip", "install", "torch"]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Torch is required for fast resize and automatic installation failed. "
            f"Command={' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    importlib.invalidate_caches()
    import torch  # noqa: F401


def _resize_frames_fast(frames_hwc_uint8: np.ndarray, image_size: int | None) -> np.ndarray:
    if image_size is None:
        return frames_hwc_uint8
    if frames_hwc_uint8.shape[1] == image_size and frames_hwc_uint8.shape[2] == image_size:
        return frames_hwc_uint8

    import torch
    import torch.nn.functional as F

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    tensor = (
        torch.from_numpy(frames_hwc_uint8)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    tensor = tensor / 255.0
    with torch.inference_mode():
        try:
            resized = F.interpolate(
                tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        except TypeError:
            resized = F.interpolate(
                tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

    out = (
        (resized.clamp(0, 1) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    return out


def process_episode_worker(task: EpisodeWorkItem) -> EpisodeWorkResult:
    row_count = task.row_stop - task.row_start
    if row_count <= 0:
        raise ValueError(f"Invalid row slice [{task.row_start}, {task.row_stop}).")
    if len(task.timestamps_s) != row_count:
        raise ValueError(
            f"timestamps size mismatch for episode {task.episode_index}: "
            f"{len(task.timestamps_s)} vs row_count={row_count}"
        )

    pixels_by_camera: dict[str, np.ndarray] = {}
    backend_used: dict[str, str] = {}
    timestamps = task.timestamps_s.astype(np.float64, copy=False)
    for camera in task.camera_keys:
        video_path = Path(task.video_paths[camera])
        query_ts_s = timestamps + float(task.from_timestamps_s[camera])
        frames_np, used = _decode_frames_linear(
            video_path=video_path,
            query_ts_s=query_ts_s,
            preferred_backend=task.preferred_backend,
            tolerance_s=task.tolerance_s,
        )
        frames_np = _resize_frames_fast(frames_np, task.image_size)
        if frames_np.shape[0] != row_count:
            raise RuntimeError(
                f"Decoded row count mismatch for ep={task.episode_index}, "
                f"cam={camera}: got={frames_np.shape[0]} expected={row_count}"
            )
        pixels_by_camera[camera] = frames_np
        backend_used[camera] = used

    return EpisodeWorkResult(
        episode_index=task.episode_index,
        row_start=task.row_start,
        row_stop=task.row_stop,
        pixels_by_camera=pixels_by_camera,
        backend_used=backend_used,
    )


def _compute_chunk_rows(total_rows: int, pixel_shape: tuple[int, int, int]) -> int:
    bytes_per_row = int(np.prod(np.asarray(pixel_shape, dtype=np.int64)))
    target_bytes = 8 * 1024 * 1024
    if bytes_per_row <= 0:
        return 1
    chunk_rows = target_bytes // bytes_per_row
    chunk_rows = max(1, min(int(chunk_rows), total_rows))
    return chunk_rows


def _get_available_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                return int(mem_status.ullAvailPhys)
        except Exception:
            return None
    return None


def _resolve_num_workers(cfg: ConvertConfig, episode_count: int) -> int:
    if episode_count <= 1:
        return 1
    if cfg.num_workers > 0:
        return min(cfg.num_workers, episode_count)
    cpu = os.cpu_count() or 1
    auto_cap = max(1, cfg.auto_max_workers)
    return max(1, min(episode_count, cpu - 1, auto_cap))


def _maybe_guard_memory_risk(
    cfg: ConvertConfig,
    split_name: str,
    ep_len: np.ndarray,
    pixel_shape: tuple[int, int, int],
    camera_count: int,
    max_workers: int,
    max_pending: int,
) -> None:
    if cfg.memory_guard_mode == "off":
        return
    if len(ep_len) == 0:
        return

    p75_rows = int(np.percentile(ep_len, 75))
    p75_rows = max(1, p75_rows)
    bytes_per_row_pixels = int(np.prod(np.asarray(pixel_shape, dtype=np.int64))) * camera_count
    episode_pixels_bytes = p75_rows * bytes_per_row_pixels

    # Rough upper bound: results in flight + worker decode buffers + process overhead.
    est_inflight_bytes = episode_pixels_bytes * max(1, max_pending)
    est_worker_bytes = episode_pixels_bytes * max(1, max_workers) * 2
    est_proc_overhead = max(1, max_workers) * int(cfg.worker_memory_buffer_mb) * 1024 * 1024
    estimated_bytes = est_inflight_bytes + est_worker_bytes + est_proc_overhead

    available = _get_available_memory_bytes()
    if available is None or available <= 0:
        print(
            f"[convert][{split_name}][warn] cannot estimate available RAM; "
            f"memory guard skipped. workers={max_workers} pending={max_pending}",
            flush=True,
        )
        return

    ratio = estimated_bytes / float(available)
    msg = (
        f"[convert][{split_name}] memory_guard estimate: p75_episode_rows={p75_rows}, "
        f"est_peak={estimated_bytes / (1024**3):.2f}GiB, "
        f"available={available / (1024**3):.2f}GiB, ratio={ratio:.2f}, "
        f"threshold={cfg.max_inflight_memory_ratio:.2f}, workers={max_workers}, pending={max_pending}"
    )
    if ratio >= cfg.max_inflight_memory_ratio:
        if cfg.memory_guard_mode == "error":
            raise MemoryError(
                msg
                + ". Refuse to start conversion. Lower --num_workers/--max_pending_tasks "
                "or raise --max_inflight_memory_ratio."
            )
        print(f"{msg} [warning: high OOM risk]", flush=True)
    elif ratio >= cfg.max_inflight_memory_ratio * 0.75:
        print(f"{msg} [warning: elevated memory pressure]", flush=True)


def _write_tabular_slice_to_all_files(
    camera_files: dict[str, h5py.File], row_slice: slice, tabular: dict[str, np.ndarray]
) -> None:
    for h5f in camera_files.values():
        h5f["action"][row_slice] = tabular["action"]
        h5f["proprio"][row_slice] = tabular["state"]
        h5f["state"][row_slice] = tabular["state"]
        h5f["episode_idx"][row_slice] = tabular["episode_idx"]
        h5f["step_idx"][row_slice] = tabular["step_idx"]
        h5f["done"][row_slice] = tabular["done"]
        h5f["timestamp"][row_slice] = tabular["timestamp"]
        h5f["index"][row_slice] = tabular["index"]
        h5f["task_index"][row_slice] = tabular["task_index"]


def _write_pixels_result_to_files(
    camera_files: dict[str, h5py.File], result: EpisodeWorkResult
) -> int:
    row_slice = slice(result.row_start, result.row_stop)
    row_count = result.row_stop - result.row_start
    for camera, frames in result.pixels_by_camera.items():
        if frames.shape[0] != row_count:
            raise RuntimeError(
                f"Worker returned invalid frame count for camera={camera}. "
                f"got={frames.shape[0]} expected={row_count}"
            )
        camera_files[camera]["pixels"][row_slice] = frames
    return row_count


def _convert_split(
    cfg: ConvertConfig,
    split_name: str,
    episodes: list[int],
    root: Path,
    output_dir: Path,
    camera_keys: list[str],
) -> list[str]:
    print(
        f"[convert][{split_name}] loading tabular columns "
        f"(episodes={len(episodes)}, cameras={camera_keys})",
        flush=True,
    )
    ds = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=root,
        episodes=episodes,
        download_videos=False,
        video_backend=cfg.video_backend,
    )
    if len(ds) == 0:
        raise RuntimeError(f"Split '{split_name}' contains zero frames after filtering.")

    ep_len, ep_offset = _episodes_lengths(ds.meta, episodes)
    total_rows = int(ep_len.sum())
    if total_rows != len(ds):
        raise RuntimeError(
            f"Unexpected row count for split '{split_name}': len(ds)={len(ds)} "
            f"vs sum(ep_len)={total_rows}."
        )

    cols = _select_columns_compat(ds, TABULAR_COLUMNS)

    first_cam = camera_keys[0]
    feat_shape = tuple(ds.meta.features[first_cam]["shape"])
    channels = int(feat_shape[-1])
    if cfg.image_size is None:
        image_hw = (int(feat_shape[0]), int(feat_shape[1]))
    else:
        image_hw = (cfg.image_size, cfg.image_size)
    pixel_shape = (image_hw[0], image_hw[1], channels)

    print(
        f"[convert][{split_name}] total_rows={total_rows}, "
        f"fps={int(ds.meta.fps)}, image_size={cfg.image_size or 'native'}",
        flush=True,
    )

    file_handles: dict[str, h5py.File] = {}
    produced_files: list[str] = []
    try:
        compression = _get_compression(cfg.compression)
        chunk_rows = _compute_chunk_rows(total_rows=total_rows, pixel_shape=pixel_shape)
        pixel_chunks = (chunk_rows,) + pixel_shape
        scalar_chunks = (chunk_rows,)

        action_probe = _read_episode_tabular_slice(cols, 0, min(1, total_rows))["action"]
        state_probe = _read_episode_tabular_slice(cols, 0, min(1, total_rows))["state"]
        action_dim = int(action_probe.shape[1])
        state_dim = int(state_probe.shape[1])

        for camera in camera_keys:
            camera_slug = camera.replace(".", "_")
            out_path = output_dir / f"{split_name}__{camera_slug}.h5"
            if out_path.exists() and not cfg.overwrite:
                raise FileExistsError(
                    f"Output file already exists: {out_path}. Use --overwrite=true."
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            h5f = h5py.File(out_path, "w")
            file_handles[camera] = h5f
            produced_files.append(str(out_path))

            h5f.create_dataset(
                "pixels",
                shape=(total_rows,) + pixel_shape,
                dtype=np.uint8,
                chunks=pixel_chunks,
                compression=compression,
            )
            h5f.create_dataset(
                "action",
                shape=(total_rows, action_dim),
                dtype=np.float32,
                chunks=(chunk_rows, action_dim),
                compression=compression,
            )
            h5f.create_dataset(
                "proprio",
                shape=(total_rows, state_dim),
                dtype=np.float32,
                chunks=(chunk_rows, state_dim),
                compression=compression,
            )
            h5f.create_dataset(
                "state",
                shape=(total_rows, state_dim),
                dtype=np.float32,
                chunks=(chunk_rows, state_dim),
                compression=compression,
            )
            h5f.create_dataset("episode_idx", shape=(total_rows,), dtype=np.int64, chunks=scalar_chunks)
            h5f.create_dataset("step_idx", shape=(total_rows,), dtype=np.int64, chunks=scalar_chunks)
            h5f.create_dataset("done", shape=(total_rows,), dtype=np.bool_, chunks=scalar_chunks)
            h5f.create_dataset("timestamp", shape=(total_rows,), dtype=np.float32, chunks=scalar_chunks)
            h5f.create_dataset("index", shape=(total_rows,), dtype=np.int64, chunks=scalar_chunks)
            h5f.create_dataset("task_index", shape=(total_rows,), dtype=np.int64, chunks=scalar_chunks)
            h5f.create_dataset("ep_len", data=ep_len, dtype=np.int64)
            h5f.create_dataset("ep_offset", data=ep_offset, dtype=np.int64)

            h5f.attrs["source_repo_id"] = cfg.repo_id
            h5f.attrs["source_split"] = split_name
            h5f.attrs["source_camera_key"] = camera
            h5f.attrs["fps"] = int(ds.meta.fps)
            h5f.attrs["generated_by"] = "convert_lerobot_to_hdf5.py"

        episode_batch_size = (
            cfg.episode_batch_size if cfg.episode_batch_size > 0 else len(episodes)
        )
        max_workers = _resolve_num_workers(cfg, len(episodes))
        max_pending = (
            cfg.max_pending_tasks
            if cfg.max_pending_tasks > 0
            else max(1, min(max_workers, 2))
        )
        use_pool = max_workers > 1
        tolerance_s = float(getattr(ds, "tolerance_s", 1e-4))

        _maybe_guard_memory_risk(
            cfg=cfg,
            split_name=split_name,
            ep_len=ep_len,
            pixel_shape=pixel_shape,
            camera_count=len(camera_keys),
            max_workers=max_workers,
            max_pending=max_pending,
        )
        if sys.platform == "darwin" and max_workers > 4:
            print(
                f"[convert][{split_name}][warn] workers={max_workers} on macOS may cause "
                "instability with video decode backends. Prefer <=4 unless benchmarked.",
                flush=True,
            )

        print(
            f"[convert][{split_name}] workers={max_workers} spawn={'yes' if use_pool else 'no'} "
            f"episode_batch_size={episode_batch_size} max_pending={max_pending}",
            flush=True,
        )

        started = time.perf_counter()
        last_log = started
        last_heartbeat = started
        rows_submitted = 0
        rows_written_pixels = 0
        row_cursor = 0
        fallback_counts: dict[str, int] = {}
        submitted_episodes = 0
        completed_episodes = 0

        mp_context = multiprocessing.get_context("spawn")
        executor: ProcessPoolExecutor | None = None
        if use_pool:
            executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)

        try:
            pending: dict[Future[EpisodeWorkResult], tuple[int, int, int]] = {}

            def flush_ready(block_until_one: bool) -> None:
                nonlocal rows_written_pixels, last_log, last_heartbeat, completed_episodes
                if not pending:
                    return
                if block_until_one:
                    done = set()
                    while not done:
                        done, _ = wait(
                            pending.keys(),
                            return_when=FIRST_COMPLETED,
                            timeout=cfg.heartbeat_seconds,
                        )
                        if done:
                            break
                        now = time.perf_counter()
                        if now - last_heartbeat >= cfg.heartbeat_seconds:
                            elapsed = now - started
                            rate = rows_written_pixels / elapsed if elapsed > 0 else 0.0
                            print(
                                f"[convert][{split_name}] heartbeat: "
                                f"episodes completed={completed_episodes}/{len(episodes)} "
                                f"rows={rows_written_pixels}/{total_rows} "
                                f"pending={len(pending)} rate={rate:.1f} rows/s",
                                flush=True,
                            )
                            last_heartbeat = now
                else:
                    done = list(pending.keys())
                for fut in done:
                    ep_idx, row_start, row_stop = pending.pop(fut)
                    result = fut.result()
                    wrote = _write_pixels_result_to_files(file_handles, result)
                    rows_written_pixels += wrote
                    completed_episodes += 1
                    for used in result.backend_used.values():
                        fallback_counts[used] = fallback_counts.get(used, 0) + 1
                    now = time.perf_counter()
                    elapsed = now - started
                    rate = rows_written_pixels / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[convert][{split_name}] episode done ep={ep_idx} rows={wrote} "
                        f"episodes={completed_episodes}/{len(episodes)} "
                        f"rows={rows_written_pixels}/{total_rows} pending={len(pending)} "
                        f"rate={rate:.1f} rows/s backends={result.backend_used}",
                        flush=True,
                    )
                    last_heartbeat = now

                    should_log = (
                        rows_written_pixels == total_rows
                        or (
                            cfg.progress_every > 0
                            and rows_written_pixels % cfg.progress_every == 0
                            and (time.perf_counter() - last_log) > 0.5
                        )
                    )
                    if should_log:
                        now = time.perf_counter()
                        elapsed = now - started
                        rate = rows_written_pixels / elapsed if elapsed > 0 else 0.0
                        remaining = total_rows - rows_written_pixels
                        eta = remaining / rate if rate > 0 else float("inf")
                        pct = (100.0 * rows_written_pixels) / total_rows
                        print(
                            f"[convert][{split_name}] {rows_written_pixels}/{total_rows} "
                            f"({pct:.1f}%) rate={rate:.1f} rows/s eta={eta:.1f}s",
                            flush=True,
                        )
                        last_log = now

            for batch_start in range(0, len(episodes), episode_batch_size):
                batch_end = min(len(episodes), batch_start + episode_batch_size)
                batch_eps = episodes[batch_start:batch_end]
                print(
                    f"[convert][{split_name}] submitting batch "
                    f"{batch_start}:{batch_end} ({len(batch_eps)} episodes)",
                    flush=True,
                )
                for local_pos, ep_idx in enumerate(batch_eps):
                    ep_global_pos = batch_start + local_pos
                    ep_count = int(ep_len[ep_global_pos])
                    row_start = row_cursor
                    row_stop = row_cursor + ep_count
                    row_slice = slice(row_start, row_stop)
                    row_cursor = row_stop

                    tabular = _read_episode_tabular_slice(cols, row_start, row_stop)
                    _write_tabular_slice_to_all_files(file_handles, row_slice, tabular)

                    ep_row = _get_episode_row(ds.meta.episodes, ep_idx)
                    video_paths = {
                        camera: str(root / ds.meta.get_video_file_path(ep_idx, camera))
                        for camera in camera_keys
                    }
                    from_ts = {
                        camera: float(ep_row[f"videos/{camera}/from_timestamp"])
                        for camera in camera_keys
                    }
                    item = EpisodeWorkItem(
                        episode_index=int(ep_idx),
                        row_start=row_start,
                        row_stop=row_stop,
                        camera_keys=list(camera_keys),
                        video_paths=video_paths,
                        from_timestamps_s=from_ts,
                        timestamps_s=tabular["timestamp"].astype(np.float64, copy=False),
                        image_size=cfg.image_size,
                        preferred_backend=cfg.video_backend,
                        tolerance_s=tolerance_s,
                    )

                    if executor is None:
                        result = process_episode_worker(item)
                        wrote = _write_pixels_result_to_files(file_handles, result)
                        rows_written_pixels += wrote
                        completed_episodes += 1
                        for used in result.backend_used.values():
                            fallback_counts[used] = fallback_counts.get(used, 0) + 1
                        now = time.perf_counter()
                        elapsed = now - started
                        rate = rows_written_pixels / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[convert][{split_name}] episode done ep={ep_idx} rows={wrote} "
                            f"episodes={completed_episodes}/{len(episodes)} "
                            f"rows={rows_written_pixels}/{total_rows} rate={rate:.1f} rows/s "
                            f"backends={result.backend_used}",
                            flush=True,
                        )
                    else:
                        fut = executor.submit(process_episode_worker, item)
                        pending[fut] = (int(ep_idx), row_start, row_stop)
                        submitted_episodes += 1
                        print(
                            f"[convert][{split_name}] submitted ep={ep_idx} rows={ep_count} "
                            f"submitted={submitted_episodes}/{len(episodes)} pending={len(pending)}",
                            flush=True,
                        )
                        if len(pending) >= max_pending:
                            flush_ready(block_until_one=True)

                    rows_submitted += ep_count

                while pending:
                    flush_ready(block_until_one=True)

            if rows_submitted != total_rows or rows_written_pixels != total_rows:
                raise RuntimeError(
                    f"Split '{split_name}' write mismatch: submitted={rows_submitted} "
                    f"written_pixels={rows_written_pixels} total_rows={total_rows}"
                )

        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        print(
            f"[convert][{split_name}] decode_backends_used={fallback_counts}",
            flush=True,
        )
    finally:
        for handle in file_handles.values():
            handle.close()

    print(f"[convert][{split_name}] finished. files={produced_files}", flush=True)
    return produced_files


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    if cfg.dirty_episode_policy not in VALID_DIRTY_POLICY:
        raise ValueError(
            f"Invalid dirty_episode_policy '{cfg.dirty_episode_policy}'. "
            f"Expected one of {VALID_DIRTY_POLICY}."
        )
    if cfg.compression not in VALID_COMPRESSION:
        raise ValueError(
            f"Invalid compression '{cfg.compression}'. Expected one of {VALID_COMPRESSION}."
        )
    if cfg.image_size is not None and cfg.image_size <= 0:
        raise ValueError("image_size must be > 0.")
    if cfg.progress_every < 0:
        raise ValueError("progress_every must be >= 0.")
    if cfg.heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be > 0.")
    if cfg.auto_max_workers <= 0:
        raise ValueError("auto_max_workers must be > 0.")
    if cfg.episode_batch_size < 0:
        raise ValueError("episode_batch_size must be >= 0.")
    if cfg.max_pending_tasks < 0:
        raise ValueError("max_pending_tasks must be >= 0.")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if cfg.memory_guard_mode not in VALID_MEMORY_GUARD_MODE:
        raise ValueError(
            f"Invalid memory_guard_mode '{cfg.memory_guard_mode}'. "
            f"Expected one of {VALID_MEMORY_GUARD_MODE}."
        )
    if cfg.max_inflight_memory_ratio <= 0 or cfg.max_inflight_memory_ratio > 1:
        raise ValueError("max_inflight_memory_ratio must be in (0, 1].")
    if cfg.worker_memory_buffer_mb < 0:
        raise ValueError("worker_memory_buffer_mb must be >= 0.")

    if cfg.image_size is not None:
        _ensure_torch_installed(auto_install=cfg.auto_install_torch)

    raw_root = Path(cfg.datasets_dir) / cfg.raw_subdir / cfg.repo_id
    output_root = (
        Path(cfg.datasets_dir) / cfg.hdf5_subdir / sanitize_repo_id(cfg.repo_id)
    )
    report_path = output_root / cfg.report_filename

    meta_ds = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=raw_root,
        download_videos=False,
    )
    all_split_names = list(meta_ds.meta.info["splits"].keys())
    split_names = cfg.splits or all_split_names
    for split_name in split_names:
        if split_name not in meta_ds.meta.info["splits"]:
            raise ValueError(
                f"Unknown split '{split_name}'. Available splits: {all_split_names}"
            )

    all_camera_keys = list(meta_ds.meta.camera_keys)
    selected_camera_keys = cfg.camera_keys or all_camera_keys
    for camera_key in selected_camera_keys:
        if camera_key not in all_camera_keys:
            raise ValueError(
                f"Unknown camera key '{camera_key}'. Available: {all_camera_keys}"
            )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": cfg.repo_id,
        "config": asdict(cfg),
        "splits": {},
        "status": "ok",
    }

    print(
        "[convert] starting "
        f"repo_id={cfg.repo_id} splits={split_names} "
        f"video_backend={cfg.video_backend or 'auto'}",
        flush=True,
    )
    print(
        "[convert] note: macOS objc AVF* duplicate warnings are typically non-fatal; "
        "conversion should continue if progress logs keep updating.",
        flush=True,
    )

    for split_name in split_names:
        requested_episodes = _split_episodes(meta_ds.meta, split_name)
        print(
            f"[convert][{split_name}] pre-validating {len(requested_episodes)} episodes...",
            flush=True,
        )
        issues = _prevalidate_source_split(
            repo_id=cfg.repo_id,
            root=raw_root,
            episodes=requested_episodes,
            require_terminal_done=cfg.require_terminal_done,
        )
        invalid_episodes = sorted(issues.keys())
        kept_episodes = requested_episodes
        action_taken = "kept_all"

        if invalid_episodes:
            print(
                f"[convert][{split_name}] invalid_episodes={len(invalid_episodes)} "
                f"policy={cfg.dirty_episode_policy}",
                flush=True,
            )
            if cfg.dirty_episode_policy == "fail":
                action_taken = "fail"
                kept_episodes = requested_episodes
            elif cfg.dirty_episode_policy == "drop":
                action_taken = "drop_invalid"
                invalid_set = set(invalid_episodes)
                kept_episodes = [
                    ep for ep in requested_episodes if ep not in invalid_set
                ]
            else:
                action_taken = "warn_only"
                kept_episodes = requested_episodes

        split_report: dict[str, Any] = {
            "split": split_name,
            "requested_episodes": requested_episodes,
            "requested_episode_count": len(requested_episodes),
            "invalid_episode_count": len(invalid_episodes),
            "invalid_episodes": [
                {"episode_index": int(ep), "issues": issues[ep]}
                for ep in invalid_episodes
            ],
            "action_taken": action_taken,
            "kept_episode_count": len(kept_episodes),
            "kept_episodes": kept_episodes,
            "produced_files": [],
        }
        report["splits"][split_name] = split_report

        if invalid_episodes and cfg.dirty_episode_policy == "fail":
            report["status"] = "failed"
            _write_report(report_path, report)
            raise RuntimeError(
                f"Split '{split_name}' contains {len(invalid_episodes)} invalid episodes "
                "and dirty_episode_policy='fail'. See conversion_report.json."
            )

        if not kept_episodes:
            report["status"] = "failed"
            _write_report(report_path, report)
            raise RuntimeError(
                f"Split '{split_name}' has no episodes left after filtering."
            )

        print(
            f"[convert][{split_name}] converting with kept_episodes={len(kept_episodes)}",
            flush=True,
        )
        produced = _convert_split(
            cfg=cfg,
            split_name=split_name,
            episodes=kept_episodes,
            root=raw_root,
            output_dir=output_root,
            camera_keys=selected_camera_keys,
        )
        split_report["produced_files"] = produced

    _write_report(report_path, report)
    print(f"[convert] done repo_id={cfg.repo_id}", flush=True)
    print(f"[convert] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
