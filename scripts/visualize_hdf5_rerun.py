#!/usr/bin/env python3
"""Visualize LeWM HDF5 episodes in Rerun."""

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import draccus
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_dataset_utils import list_hdf5_files  # noqa: E402

REQUIRED_KEYS = (
    "pixels",
    "action",
    "state",
    "done",
    "episode_idx",
    "step_idx",
    "ep_len",
    "ep_offset",
)


@dataclass
class VisualizeHDF5Config:
    target_path: str = "datasets/hdf5"
    split: str | None = None
    camera_key: str | None = None
    episode_index: int = 0
    start_step: int = 0
    end_step: int | None = None
    step_stride: int = 1
    playback_hz: float = 0.0
    spawn_viewer: bool = True
    save_rrd: bool = True
    rrd_path: str | None = None
    progress_every: int = 100


def _require_h5py() -> Any:
    if importlib.util.find_spec("h5py") is None:
        raise RuntimeError("Missing dependency 'h5py'. Install with: pip install h5py")
    import h5py

    return h5py


def _require_rerun() -> Any:
    if importlib.util.find_spec("rerun") is None:
        raise RuntimeError(
            "Missing dependency 'rerun-sdk'. Install with: pip install rerun-sdk"
        )
    import rerun as rr

    return rr


def _camera_slug(camera_key: str) -> str:
    return camera_key.replace(".", "_")


def _parse_file_name(path: Path) -> tuple[str | None, str | None]:
    name = path.stem
    if "__" not in name:
        return None, None
    split, camera_slug = name.split("__", 1)
    return split, camera_slug


def _match_camera_slug(candidate_slug: str, requested_camera_key: str) -> bool:
    req_slug = _camera_slug(requested_camera_key)
    return candidate_slug == req_slug or candidate_slug == requested_camera_key


def _resolve_hdf5_file(target_path: Path, split: str | None, camera_key: str | None) -> Path:
    if target_path.is_file():
        if target_path.suffix != ".h5":
            raise ValueError(f"target_path must be a .h5 file, got: {target_path}")
        return target_path

    candidates = list_hdf5_files(target_path)
    if not candidates:
        raise FileNotFoundError(f"No .h5 files found under: {target_path}")

    filtered = candidates
    if split:
        filtered = [p for p in filtered if _parse_file_name(p)[0] == split]
    if camera_key:
        filtered = [
            p
            for p in filtered
            if _parse_file_name(p)[1] is not None
            and _match_camera_slug(_parse_file_name(p)[1] or "", camera_key)
        ]

    if len(filtered) == 1:
        return filtered[0]
    if not filtered:
        raise FileNotFoundError(
            f"No .h5 file matched filters under {target_path} "
            f"(split={split}, camera_key={camera_key})."
        )
    raise ValueError(
        "Ambiguous selection: multiple .h5 files matched filters. "
        f"Please refine split/camera_key. Matches={[str(p) for p in filtered]}"
    )


def _default_rrd_path(h5_path: Path, episode_index: int) -> Path:
    return h5_path.with_name(f"{h5_path.stem}__ep{episode_index}.rrd")


def _validate_hdf5_schema(h5f: Any) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in h5f]
    if missing:
        raise KeyError(f"Missing required HDF5 keys: {missing}")

    pixels = h5f["pixels"]
    if pixels.ndim != 4:
        raise ValueError(f"'pixels' must be rank-4, got rank-{pixels.ndim}.")
    if pixels.dtype != np.uint8:
        raise ValueError(f"'pixels' must be uint8, got dtype={pixels.dtype}.")
    channels = int(pixels.shape[-1])
    if channels not in (1, 3):
        raise ValueError(f"'pixels' channels must be 1 or 3, got {channels}.")

    if h5f["action"].shape[0] != pixels.shape[0] or h5f["state"].shape[0] != pixels.shape[0]:
        raise ValueError("Per-step arrays have inconsistent row counts.")


def _episode_bounds(h5f: Any, episode_index: int) -> tuple[int, int, int]:
    ep_len = np.asarray(h5f["ep_len"][:], dtype=np.int64)
    ep_offset = np.asarray(h5f["ep_offset"][:], dtype=np.int64)

    if ep_len.ndim != 1 or ep_offset.ndim != 1:
        raise ValueError("ep_len/ep_offset must be rank-1.")
    if len(ep_len) != len(ep_offset):
        raise ValueError("ep_len and ep_offset length mismatch.")
    if episode_index < 0 or episode_index >= len(ep_len):
        raise IndexError(
            f"episode_index={episode_index} out of range [0, {len(ep_len) - 1}]"
        )

    start = int(ep_offset[episode_index])
    length = int(ep_len[episode_index])
    end = start + length
    if length <= 0:
        raise ValueError(f"Episode {episode_index} has non-positive length={length}.")
    return start, end, length


def _resolve_step_window(
    *,
    episode_length: int,
    start_step: int,
    end_step: int | None,
    step_stride: int,
) -> range:
    if step_stride <= 0:
        raise ValueError("step_stride must be > 0.")
    if start_step < 0:
        raise ValueError("start_step must be >= 0.")

    max_step = episode_length - 1
    resolved_end = max_step if end_step is None else end_step
    if resolved_end < 0:
        raise ValueError("end_step must be >= 0 when provided.")
    if start_step > max_step:
        raise ValueError(f"start_step={start_step} exceeds episode max step={max_step}.")
    if resolved_end > max_step:
        raise ValueError(f"end_step={resolved_end} exceeds episode max step={max_step}.")
    if resolved_end < start_step:
        raise ValueError(
            f"Invalid step window: end_step={resolved_end} < start_step={start_step}."
        )
    return range(start_step, resolved_end + 1, step_stride)


def _extract_source_metadata(
    h5f: Any,
    file_path: Path,
    requested_split: str | None,
    requested_camera_key: str | None,
    episode_index: int,
    step_window: range,
) -> dict[str, Any]:
    split_from_name, camera_slug_from_name = _parse_file_name(file_path)
    split_name = (
        requested_split
        or h5f.attrs.get("source_split")
        or split_from_name
        or "unknown"
    )
    camera_name = (
        requested_camera_key
        or h5f.attrs.get("source_camera_key")
        or camera_slug_from_name
        or "unknown"
    )
    return {
        "source_file": str(file_path),
        "source_repo_id": str(h5f.attrs.get("source_repo_id", "unknown")),
        "source_split": str(split_name),
        "source_camera_key": str(camera_name),
        "episode_index": int(episode_index),
        "start_step": int(step_window.start),
        "end_step": int(step_window.stop - 1),
        "step_stride": int(step_window.step),
    }

def _rr_scalar(rr: Any, value: float | int | bool) -> Any:
    scalar = float(value)
    if hasattr(rr, "Scalar"):
        return rr.Scalar(scalar)
    if hasattr(rr, "Scalars"):
        return rr.Scalars(scalar)
    if hasattr(rr, "TimeSeriesScalar"):
        return rr.TimeSeriesScalar(scalar)
    raise AttributeError(
        "Rerun API mismatch: no Scalar/Scalars/TimeSeriesScalar found."
    )


def _as_recording_list(recordings: Any) -> list[Any]:
    if isinstance(recordings, (list, tuple)):
        return list(recordings)
    return [recordings]


def _log_to_recordings(recordings: Any, path: str, entity: Any) -> None:
    for rec in _as_recording_list(recordings):
        rec.log(path, entity)


def _set_time_on_recordings(recordings: Any, step_value: int) -> None:
    now_s = time.time()
    for rec in _as_recording_list(recordings):
        rec.set_time_sequence("step", step_value)
        if hasattr(rec, "set_time_seconds"):
            rec.set_time_seconds("log_time", now_s)


def _build_default_blueprint(rr: Any) -> Any:
    bp = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            rr.blueprint.Spatial2DView(
                name="Episode Camera",
                origin="/",
                contents=["episode/pixels"],
            ),
            rr.blueprint.Tabs(
                rr.blueprint.TimeSeriesView(
                    name="Action",
                    origin="/",
                    contents=["signals/action/**"],
                ),
                rr.blueprint.TimeSeriesView(
                    name="State",
                    origin="/",
                    contents=["signals/state/**"],
                ),
                rr.blueprint.TimeSeriesView(
                    name="Episode Signals",
                    origin="/",
                    contents=["signals/done", "signals/step_idx", "signals/episode_idx"],
                ),
                rr.blueprint.TextLogView(
                    name="Metadata",
                    origin="/",
                    contents=["metadata/**"],
                ),
            ),
            row_shares=[0.68, 0.32],
        )
    )
    return bp


def _log_metadata_with_recordings(rr: Any, recordings: Any, metadata: dict[str, Any]) -> None:
    text = json.dumps(metadata, indent=2)
    if hasattr(rr, "TextDocument"):
        _log_to_recordings(
            recordings,
            "metadata/session",
            rr.TextDocument(text, media_type="text/plain"),
        )
    elif hasattr(rr, "TextLog"):
        _log_to_recordings(recordings, "metadata/session", rr.TextLog(text))
    _log_to_recordings(
        recordings, "metadata/episode_index", _rr_scalar(rr, metadata["episode_index"])
    )
    _log_to_recordings(
        recordings, "metadata/start_step", _rr_scalar(rr, metadata["start_step"])
    )
    _log_to_recordings(
        recordings, "metadata/end_step", _rr_scalar(rr, metadata["end_step"])
    )


def _normalize_frame_for_rerun(frame_value: Any) -> np.ndarray:
    frame = np.asarray(frame_value)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame rank-3, got shape={frame.shape}.")

    # Support CHW tensors by converting to HWC.
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if frame.max(initial=0) <= 1.0:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    if frame.shape[-1] not in (1, 3):
        raise ValueError(
            f"Expected HWC image with channels 1 or 3 after normalization, got {frame.shape}."
        )
    return np.ascontiguousarray(frame)


def _log_episode_steps(
    *,
    rr: Any,
    recordings: Any | None = None,
    h5f: Any,
    episode_row_start: int,
    step_window: range,
    playback_hz: float,
    progress_every: int,
) -> int:
    pixels = h5f["pixels"]
    actions = h5f["action"]
    states = h5f["state"]
    done = h5f["done"]
    step_idx = h5f["step_idx"]
    episode_idx = h5f["episode_idx"]

    total = len(step_window)
    logged = 0
    rec_targets = recordings if recordings is not None else rr
    for rel_step in step_window:
        loop_started = time.perf_counter()
        row = episode_row_start + rel_step
        step_value = int(step_idx[row])
        _set_time_on_recordings(rec_targets, step_value)

        frame = _normalize_frame_for_rerun(pixels[row])
        if frame.shape[-1] == 1:
            image = frame[:, :, 0]  # grayscale image in Rerun
        else:
            image = frame
        _log_to_recordings(rec_targets, "episode/pixels", rr.Image(image))

        _log_to_recordings(rec_targets, "signals/done", _rr_scalar(rr, int(bool(done[row]))))
        _log_to_recordings(rec_targets, "signals/episode_idx", _rr_scalar(rr, int(episode_idx[row])))
        _log_to_recordings(rec_targets, "signals/step_idx", _rr_scalar(rr, step_value))

        action_row = np.asarray(actions[row], dtype=np.float32).reshape(-1)
        state_row = np.asarray(states[row], dtype=np.float32).reshape(-1)
        for i, value in enumerate(action_row):
            _log_to_recordings(
                rec_targets, f"signals/action/dim_{i}", _rr_scalar(rr, float(value))
            )
        for i, value in enumerate(state_row):
            _log_to_recordings(
                rec_targets, f"signals/state/dim_{i}", _rr_scalar(rr, float(value))
            )

        logged += 1
        if progress_every > 0 and (logged % progress_every == 0 or logged == total):
            print(f"[viz] progress {logged}/{total}", flush=True)

        if playback_hz > 0:
            step_budget = 1.0 / playback_hz
            elapsed = time.perf_counter() - loop_started
            if elapsed < step_budget:
                time.sleep(step_budget - elapsed)
    return logged


@draccus.wrap()
def main(cfg: VisualizeHDF5Config) -> None:
    if cfg.episode_index < 0:
        raise ValueError("episode_index must be >= 0.")
    if cfg.progress_every < 0:
        raise ValueError("progress_every must be >= 0.")
    if cfg.playback_hz < 0:
        raise ValueError("playback_hz must be >= 0.")

    h5py = _require_h5py()
    rr = _require_rerun()

    target = Path(cfg.target_path)
    h5_path = _resolve_hdf5_file(
        target_path=target,
        split=cfg.split,
        camera_key=cfg.camera_key,
    )
    rrd_path = Path(cfg.rrd_path) if cfg.rrd_path else _default_rrd_path(
        h5_path, cfg.episode_index
    )

    print(f"[viz] selected file: {h5_path}", flush=True)
    print(
        f"[viz] options: episode={cfg.episode_index}, start_step={cfg.start_step}, "
        f"end_step={cfg.end_step}, stride={cfg.step_stride}, playback_hz={cfg.playback_hz}",
        flush=True,
    )

    with h5py.File(h5_path, "r") as h5f:
        _validate_hdf5_schema(h5f)
        ep_row_start, ep_row_end, ep_length = _episode_bounds(
            h5f=h5f,
            episode_index=cfg.episode_index,
        )
        step_window = _resolve_step_window(
            episode_length=ep_length,
            start_step=cfg.start_step,
            end_step=cfg.end_step,
            step_stride=cfg.step_stride,
        )

        metadata = _extract_source_metadata(
            h5f=h5f,
            file_path=h5_path,
            requested_split=cfg.split,
            requested_camera_key=cfg.camera_key,
            episode_index=cfg.episode_index,
            step_window=step_window,
        )

        live_rec = rr.RecordingStream("lewm_hdf5_viewer")
        if cfg.spawn_viewer:
            live_rec.spawn()
        recordings: list[Any] = [live_rec]
        file_rec = None
        if cfg.save_rrd:
            rrd_path.parent.mkdir(parents=True, exist_ok=True)
            file_rec = rr.RecordingStream("lewm_hdf5_viewer")
            file_rec.save(str(rrd_path))
            recordings.append(file_rec)

        blueprint = _build_default_blueprint(rr)
        for rec in recordings:
            rec.send_blueprint(blueprint, make_active=True, make_default=True)

        _log_metadata_with_recordings(rr, recordings, metadata)
        print(
            f"[viz] episode bounds: rows=[{ep_row_start},{ep_row_end}) "
            f"window=[{step_window.start},{step_window.stop - 1}] "
            f"steps={len(step_window)}",
            flush=True,
        )

        started = time.perf_counter()
        logged_steps = _log_episode_steps(
            rr=rr,
            recordings=recordings,
            h5f=h5f,
            episode_row_start=ep_row_start,
            step_window=step_window,
            playback_hz=cfg.playback_hz,
            progress_every=cfg.progress_every,
        )
        elapsed = time.perf_counter() - started
        rate = logged_steps / elapsed if elapsed > 0 else 0.0
        print(
            f"[viz] done logged_steps={logged_steps} elapsed={elapsed:.2f}s "
            f"rate={rate:.1f} steps/s",
            flush=True,
        )
        if cfg.save_rrd:
            print(f"[viz] saved rrd: {rrd_path}", flush=True)
        print(
            "[viz] tip: in Rerun, select timeline 'step' to scrub sequence data.",
            flush=True,
        )
        for rec in recordings:
            if hasattr(rec, "disconnect"):
                rec.disconnect()


if __name__ == "__main__":
    main()
