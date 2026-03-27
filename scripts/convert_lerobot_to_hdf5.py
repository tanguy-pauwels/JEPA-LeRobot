#!/usr/bin/env python3
"""Convert LeRobot datasets (parquet+mp4) to LeWM-compatible HDF5 files."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import draccus
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    import h5py
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]

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
VALID_DECODE_BACKEND = ("pyav", "opencv")
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

    decode_backend: str = "pyav"
    micro_batch_size: int = 64
    stall_timeout_seconds: float = 120.0

    progress_every: int = 500
    heartbeat_seconds: float = 10.0

    # Legacy options kept for CLI compatibility (deprecated).
    num_workers: int = 1
    video_backend: str | None = None
    auto_max_workers: int = 2
    episode_batch_size: int = 16
    max_pending_tasks: int = 0
    memory_guard_mode: str = "warn"
    max_inflight_memory_ratio: float = 0.40
    worker_memory_buffer_mb: int = 256
    auto_install_torch: bool = True


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


def _write_tabular_slice_to_all_files(
    camera_files: dict[str, Any], row_slice: slice, tabular: dict[str, np.ndarray]
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


def _require_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "Missing dependency 'h5py'. Install with: pip install h5py"
        )


def _require_pyav() -> Any:
    if importlib.util.find_spec("av") is None:
        raise RuntimeError("Missing dependency 'av'. Install with: pip install av")
    import av

    return av


def _require_cv2() -> Any:
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError(
            "Missing dependency 'opencv-python'. Install with: pip install opencv-python"
        )
    import cv2

    return cv2


def _assert_runtime_dependencies() -> None:
    _require_h5py()
    _require_pyav()
    _require_cv2()


def _normalize_frame_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected frame with 3 dims, got shape={arr.shape}.")

    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max(initial=0) <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"Frame must be HWC with C=1/3. Got {arr.shape}.")
    return np.ascontiguousarray(arr)


def _resize_batch_cv2(frames: list[np.ndarray], image_size: int | None) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot resize an empty batch.")

    if image_size is None:
        out = [_normalize_frame_hwc_uint8(f) for f in frames]
        return np.stack(out, axis=0)

    cv2 = _require_cv2()
    resized: list[np.ndarray] = []
    for frame in frames:
        arr = _normalize_frame_hwc_uint8(frame)
        resized_frame = cv2.resize(
            arr,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if resized_frame.ndim == 2:
            resized_frame = resized_frame[:, :, None]
        resized.append(_normalize_frame_hwc_uint8(resized_frame))
    return np.stack(resized, axis=0)


def _iter_frames_pyav(video_path: Path) -> Iterator[np.ndarray]:
    av = _require_pyav()
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _iter_frames_opencv(video_path: Path) -> Iterator[np.ndarray]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield rgb
    finally:
        cap.release()


def _backend_order(decode_backend: str) -> list[str]:
    if decode_backend == "pyav":
        return ["pyav", "opencv"]
    return ["opencv", "pyav"]


def _open_linear_frame_iterator(
    video_path: Path,
    decode_backend: str,
) -> tuple[str, Iterator[np.ndarray], np.ndarray]:
    errors: list[str] = []
    for backend in _backend_order(decode_backend):
        frame_iter: Iterator[np.ndarray] | None = None
        opened = False
        try:
            if backend == "pyav":
                frame_iter = _iter_frames_pyav(video_path)
            else:
                frame_iter = _iter_frames_opencv(video_path)
            first = next(frame_iter)
            opened = True
            return backend, frame_iter, _normalize_frame_hwc_uint8(first)
        except StopIteration:
            errors.append(f"{backend}: empty stream")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
        finally:
            if frame_iter is not None and not opened:
                close_fn = getattr(frame_iter, "close", None)
                if callable(close_fn):
                    close_fn()

    raise RuntimeError(
        f"Unable to open {video_path} with backends {_backend_order(decode_backend)}. "
        f"Errors={errors}"
    )


def _check_stall_or_raise(
    *,
    now: float,
    last_progress: float,
    stall_timeout_seconds: float,
    context: str,
) -> None:
    if now - last_progress > stall_timeout_seconds:
        raise TimeoutError(
            f"Stall watchdog triggered ({stall_timeout_seconds:.1f}s without progress). "
            f"Context={context}"
        )


def _assert_frame_count_or_raise(
    *,
    decoded_frames: int,
    expected_frames: int,
    split_name: str,
    episode_index: int,
    camera_key: str,
) -> None:
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch on split={split_name}, episode={episode_index}, "
            f"camera={camera_key}. decoded={decoded_frames}, expected={expected_frames}."
        )


def _compute_chunk_rows(total_rows: int, micro_batch_size: int) -> int:
    return max(1, min(total_rows, micro_batch_size))


def _estimate_micro_batch_ram_mb(
    micro_batch_size: int,
    pixel_shape: tuple[int, int, int],
    camera_count: int,
) -> float:
    bytes_per_frame_all_cams = int(np.prod(np.asarray(pixel_shape, dtype=np.int64))) * camera_count
    return (bytes_per_frame_all_cams * micro_batch_size) / (1024.0 * 1024.0)


def _resolve_decode_backend(cfg: ConvertConfig) -> str:
    backend = cfg.decode_backend.strip().lower()
    if cfg.video_backend:
        legacy = cfg.video_backend.strip().lower()
        if legacy in ("pyav", "av") and cfg.decode_backend == "pyav":
            print(
                "[convert][warn] --video_backend is deprecated; mapped to --decode_backend=pyav.",
                flush=True,
            )
            backend = "pyav"
        elif legacy in ("opencv", "cv2") and cfg.decode_backend == "pyav":
            print(
                "[convert][warn] --video_backend is deprecated; mapped to --decode_backend=opencv.",
                flush=True,
            )
            backend = "opencv"
        else:
            print(
                "[convert][warn] --video_backend is deprecated and ignored by the new linear pipeline.",
                flush=True,
            )
    return backend


def _warn_deprecated_options(cfg: ConvertConfig) -> None:
    if cfg.num_workers != 1:
        print(
            f"[convert][warn] --num_workers={cfg.num_workers} is deprecated in stable mode; forcing 1.",
            flush=True,
        )
    if cfg.auto_max_workers != 2:
        print("[convert][warn] --auto_max_workers is deprecated and ignored.", flush=True)
    if cfg.episode_batch_size != 16:
        print("[convert][warn] --episode_batch_size is deprecated and ignored.", flush=True)
    if cfg.max_pending_tasks != 0:
        print("[convert][warn] --max_pending_tasks is deprecated and ignored.", flush=True)
    if cfg.memory_guard_mode != "warn":
        print("[convert][warn] --memory_guard_mode is deprecated and ignored.", flush=True)
    if cfg.max_inflight_memory_ratio != 0.40:
        print("[convert][warn] --max_inflight_memory_ratio is deprecated and ignored.", flush=True)
    if cfg.worker_memory_buffer_mb != 256:
        print("[convert][warn] --worker_memory_buffer_mb is deprecated and ignored.", flush=True)
    if cfg.auto_install_torch is not True:
        print("[convert][warn] --auto_install_torch is deprecated and ignored.", flush=True)


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


def _decode_and_write_episode_camera(
    *,
    split_name: str,
    episode_index: int,
    camera: str,
    video_path: Path,
    pixels_ds: Any,
    row_start: int,
    row_stop: int,
    expected_rows: int,
    image_size: int | None,
    micro_batch_size: int,
    decode_backend: str,
    stall_timeout_seconds: float,
    heartbeat_seconds: float,
) -> tuple[str, int, int]:
    if expected_rows <= 0:
        return "none", 0, 0

    backend_used, frame_iter, first_frame = _open_linear_frame_iterator(
        video_path=video_path,
        decode_backend=decode_backend,
    )

    write_cursor = row_start
    decoded_frames = 1
    flush_count = 0
    buffer: list[np.ndarray] = [first_frame]

    now = time.perf_counter()
    last_progress = now
    last_heartbeat = now

    context = f"split={split_name} ep={episode_index} cam={camera}"

    try:
        while decoded_frames < expected_rows:
            now = time.perf_counter()
            _check_stall_or_raise(
                now=now,
                last_progress=last_progress,
                stall_timeout_seconds=stall_timeout_seconds,
                context=context,
            )
            try:
                frame = next(frame_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Video ended early ({context}). decoded={decoded_frames}, expected={expected_rows}"
                ) from exc

            buffer.append(frame)
            decoded_frames += 1
            last_progress = now

            if len(buffer) >= micro_batch_size:
                batch = _resize_batch_cv2(buffer, image_size)
                n = int(batch.shape[0])
                pixels_ds[write_cursor : write_cursor + n] = batch
                write_cursor += n
                flush_count += 1
                print(
                    f"[convert][{split_name}] flush ep={episode_index} cam={camera} "
                    f"batch={flush_count} rows={n} written={write_cursor - row_start}/{expected_rows}",
                    flush=True,
                )
                buffer.clear()
                last_progress = time.perf_counter()

            now = time.perf_counter()
            if now - last_heartbeat >= heartbeat_seconds:
                print(
                    f"[convert][{split_name}] heartbeat ep={episode_index} cam={camera} "
                    f"decoded={decoded_frames}/{expected_rows} "
                    f"written={write_cursor - row_start}/{expected_rows}",
                    flush=True,
                )
                last_heartbeat = now

        if buffer:
            batch = _resize_batch_cv2(buffer, image_size)
            n = int(batch.shape[0])
            pixels_ds[write_cursor : write_cursor + n] = batch
            write_cursor += n
            flush_count += 1
            print(
                f"[convert][{split_name}] flush ep={episode_index} cam={camera} "
                f"batch={flush_count} rows={n} written={write_cursor - row_start}/{expected_rows} (final)",
                flush=True,
            )
            buffer.clear()
            last_progress = time.perf_counter()

        _assert_frame_count_or_raise(
            decoded_frames=decoded_frames,
            expected_frames=expected_rows,
            split_name=split_name,
            episode_index=episode_index,
            camera_key=camera,
        )
        if write_cursor != row_stop:
            raise RuntimeError(
                f"Write cursor mismatch ({context}). write_cursor={write_cursor} row_stop={row_stop}"
            )

        return backend_used, decoded_frames, flush_count
    finally:
        close_fn = getattr(frame_iter, "close", None)
        if callable(close_fn):
            close_fn()


def _convert_split(
    cfg: ConvertConfig,
    split_name: str,
    episodes: list[int],
    root: Path,
    output_dir: Path,
    camera_keys: list[str],
    decode_backend: str,
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

    batch_ram_mb = _estimate_micro_batch_ram_mb(
        micro_batch_size=cfg.micro_batch_size,
        pixel_shape=pixel_shape,
        camera_count=len(camera_keys),
    )
    print(
        f"[convert][{split_name}] total_rows={total_rows}, fps={int(ds.meta.fps)}, "
        f"image_size={cfg.image_size or 'native'}, micro_batch={cfg.micro_batch_size}, "
        f"decode_backend={decode_backend}, est_batch_ram={batch_ram_mb:.1f}MB",
        flush=True,
    )

    file_handles: dict[str, Any] = {}
    produced_files: list[str] = []
    try:
        compression = _get_compression(cfg.compression)
        chunk_rows = _compute_chunk_rows(total_rows=total_rows, micro_batch_size=cfg.micro_batch_size)
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

            h5f = h5py.File(out_path, "w")  # type: ignore[union-attr]
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

        started = time.perf_counter()
        last_log = started
        cursor = 0
        rows_written = 0
        backend_counts: dict[str, int] = {}

        for ep_pos, ep_idx in enumerate(episodes):
            ep_count = int(ep_len[ep_pos])
            row_start = cursor
            row_stop = cursor + ep_count
            row_slice = slice(row_start, row_stop)

            ep_started = time.perf_counter()
            print(
                f"[convert][{split_name}] episode start ep={ep_idx} rows={ep_count} "
                f"({ep_pos + 1}/{len(episodes)})",
                flush=True,
            )

            tabular = _read_episode_tabular_slice(cols, row_start, row_stop)
            _write_tabular_slice_to_all_files(file_handles, row_slice, tabular)

            for camera in camera_keys:
                video_path = root / ds.meta.get_video_file_path(ep_idx, camera)
                backend_used, decoded, flushes = _decode_and_write_episode_camera(
                    split_name=split_name,
                    episode_index=int(ep_idx),
                    camera=camera,
                    video_path=video_path,
                    pixels_ds=file_handles[camera]["pixels"],
                    row_start=row_start,
                    row_stop=row_stop,
                    expected_rows=ep_count,
                    image_size=cfg.image_size,
                    micro_batch_size=cfg.micro_batch_size,
                    decode_backend=decode_backend,
                    stall_timeout_seconds=cfg.stall_timeout_seconds,
                    heartbeat_seconds=cfg.heartbeat_seconds,
                )
                backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1
                print(
                    f"[convert][{split_name}] episode camera done ep={ep_idx} cam={camera} "
                    f"decoded={decoded} backend={backend_used} flushes={flushes}",
                    flush=True,
                )

            cursor = row_stop
            rows_written = cursor

            ep_elapsed = time.perf_counter() - ep_started
            print(
                f"[convert][{split_name}] episode done ep={ep_idx} rows={ep_count} "
                f"elapsed={ep_elapsed:.2f}s",
                flush=True,
            )

            should_log = (
                rows_written == total_rows
                or (
                    cfg.progress_every > 0
                    and rows_written % cfg.progress_every == 0
                    and (time.perf_counter() - last_log) > 0.5
                )
                or (time.perf_counter() - last_log) >= cfg.heartbeat_seconds
            )
            if should_log:
                now = time.perf_counter()
                elapsed = now - started
                rate = rows_written / elapsed if elapsed > 0 else 0.0
                remaining = total_rows - rows_written
                eta = remaining / rate if rate > 0 else float("inf")
                pct = (100.0 * rows_written) / total_rows
                print(
                    f"[convert][{split_name}] {rows_written}/{total_rows} "
                    f"({pct:.1f}%) rate={rate:.1f} rows/s eta={eta:.1f}s",
                    flush=True,
                )
                last_log = now

        if rows_written != total_rows:
            raise RuntimeError(
                f"Split '{split_name}' write mismatch: written={rows_written} total_rows={total_rows}"
            )

        print(
            f"[convert][{split_name}] decode_backends_used={backend_counts}",
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
    if cfg.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be > 0.")
    if cfg.stall_timeout_seconds <= 0:
        raise ValueError("stall_timeout_seconds must be > 0.")

    decode_backend = _resolve_decode_backend(cfg)
    if decode_backend not in VALID_DECODE_BACKEND:
        raise ValueError(
            f"Invalid decode_backend '{decode_backend}'. Expected one of {VALID_DECODE_BACKEND}."
        )

    _assert_runtime_dependencies()
    _warn_deprecated_options(cfg)

    raw_root = Path(cfg.datasets_dir) / cfg.raw_subdir / cfg.repo_id
    output_root = Path(cfg.datasets_dir) / cfg.hdf5_subdir / sanitize_repo_id(cfg.repo_id)
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
        "resolved_decode_backend": decode_backend,
        "splits": {},
        "status": "ok",
    }

    print(
        "[convert] starting "
        f"repo_id={cfg.repo_id} splits={split_names} "
        f"decode_backend={decode_backend} micro_batch={cfg.micro_batch_size}",
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
                kept_episodes = [ep for ep in requested_episodes if ep not in invalid_set]
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
            decode_backend=decode_backend,
        )
        split_report["produced_files"] = produced

    _write_report(report_path, report)
    print(f"[convert] done repo_id={cfg.repo_id}", flush=True)
    print(f"[convert] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
