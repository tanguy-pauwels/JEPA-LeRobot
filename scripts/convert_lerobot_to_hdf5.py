#!/usr/bin/env python3
"""Convert LeRobot datasets (parquet+mp4) to LeWM-compatible HDF5 files."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator

import draccus
import numpy as np
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

try:
    import h5py
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_dataset_utils import (  # noqa: E402
    FORMAT_VERSION,
    atomic_write_json,
    camera_slug,
    collect_source_episode_issues,
    load_manifest,
    load_state,
    manifest_path,
    parse_episode_range,
    remote_file_size_map,
    sanitize_repo_id,
    shard_camera_relpath,
    shard_id_str,
    state_path,
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
    "timestamp",
    "index",
    "task_index",
]
DONE_COLUMN_CANDIDATES = ("next.done", "is_terminal", "is_last")


class LocalLeRobotMetadata:
    """Minimal local metadata reader that avoids Hugging Face dataset cache use."""

    def __init__(self, root: Path):
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.episodes = self._load_episodes()
        self.total_episodes = len(self.episodes)
        self.total_frames = sum(
            int(row.get("length", int(row["dataset_to_index"]) - int(row["dataset_from_index"])))
            for row in self.episodes
        )
        self.fps = self.info["fps"]
        self.features = self.info["features"]

    def _load_episodes(self) -> list[dict[str, Any]]:
        try:
            import pyarrow.dataset as pa_ds
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local-only metadata loading requires pyarrow. "
                "Install the project requirements or disable --local_only."
            ) from exc

        paths = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Local metadata is missing episode parquet files under {self.root / 'meta' / 'episodes'}."
            )
        rows = pa_ds.dataset([str(path) for path in paths], format="parquet").to_table().to_pylist()
        rows.sort(key=lambda row: int(row["episode_index"]))
        return rows

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature.get("dtype") in ("video", "image")
        ]

    def get_data_file_path(self, ep_index: int) -> Path:
        row = self.episodes[int(ep_index)]
        return Path(
            self.info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        row = self.episodes[int(ep_index)]
        return Path(
            self.info["video_path"].format(
                video_key=vid_key,
                chunk_index=int(row[f"videos/{vid_key}/chunk_index"]),
                file_index=int(row[f"videos/{vid_key}/file_index"]),
            )
        )


@dataclass
class ConvertConfig:
    repo_id: str = DEFAULT_DATASET
    datasets_dir: str = "datasets"
    raw_subdir: str = "raw"
    hdf5_subdir: str = "hdf5"
    splits: list[str] = field(default_factory=list)
    camera_keys: list[str] = field(default_factory=list)
    episode_indices: list[int] = field(default_factory=list)
    episode_start: int | None = None
    episode_count: int | None = None
    local_only: bool = False
    image_size: int | None = None
    compression: str = "lzf"
    overwrite: bool = False
    prevalidate_source: bool = True
    require_terminal_done: bool = True
    dirty_episode_policy: str = "fail"
    report_filename: str = "conversion_report.json"
    publish_repo_id: str | None = None
    publish_revision: str | None = None
    hf_private: bool = False
    shard_episode_count: int | None = None
    resume: bool = True
    cleanup_local_shards_after_upload: bool = True
    upload_manifest_every_shard: bool = True

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


@dataclass(frozen=True)
class VideoEpisodeSlice:
    episode_index: int
    episode_length: int
    global_row_start: int
    global_row_stop: int
    video_frame_start: int
    video_frame_stop: int


@dataclass(frozen=True)
class VideoDecodePlan:
    camera_key: str
    video_path: Path
    slices: list[VideoEpisodeSlice]
    expected_total_frames: int


_CONFIG_PATH_KEYS_TO_REDACT = {
    "datasets_dir",
    "raw_subdir",
    "hdf5_subdir",
}


def _split_episodes(meta: Any, split_name: str) -> list[int]:
    split_spec = meta.info["splits"][split_name]
    return parse_episode_range(split_spec, total_episodes=meta.total_episodes)


def _config_for_summary(cfg: ConvertConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key in _CONFIG_PATH_KEYS_TO_REDACT:
        payload.pop(key, None)
    return payload


def _resolve_done_column(columns: Any) -> str:
    available = set(columns)
    for candidate in DONE_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate
    raise KeyError(
        f"Could not find an episode termination column. "
        f"Expected one of {DONE_COLUMN_CANDIDATES}; available columns={sorted(available)}"
    )


def _select_episode_indices(
    available_episodes: list[int],
    *,
    episode_indices: list[int],
    episode_start: int | None,
    episode_count: int | None,
) -> list[int]:
    has_explicit = len(episode_indices) > 0
    has_range = episode_start is not None or episode_count is not None
    if has_explicit and has_range:
        raise ValueError("Use either episode_indices or episode_start/episode_count, not both.")

    if has_explicit:
        selected = [int(ep) for ep in episode_indices]
        available_set = set(available_episodes)
        invalid = [ep for ep in selected if ep < 0 or ep not in available_set]
        if invalid:
            raise ValueError(f"Episode indices {invalid} are not in the selected split.")
        return selected

    if episode_start is None and episode_count is None:
        return available_episodes

    start = 0 if episode_start is None else int(episode_start)
    if start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}.")
    if episode_count is not None and int(episode_count) <= 0:
        raise ValueError(f"episode_count must be > 0, got {episode_count}.")
    if start > len(available_episodes):
        raise ValueError(
            f"episode_start={start} exceeds selected split episode count ({len(available_episodes)})."
        )

    stop = len(available_episodes) if episode_count is None else start + int(episode_count)
    if stop > len(available_episodes):
        raise ValueError(
            f"Requested episode range [{start}, {stop}) exceeds "
            f"selected split episode count ({len(available_episodes)})."
        )
    return available_episodes[start:stop]


def _assert_local_metadata_exists(root: Path) -> None:
    required = [root / "meta", root / "meta" / "info.json"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Local-only conversion requested, but local LeRobot metadata is missing: "
            f"{missing}. Expected a downloaded dataset under {root}."
        )


def _assert_local_episode_files(
    root: Path,
    meta: Any,
    episodes: list[int],
    camera_keys: list[str],
) -> None:
    missing: dict[Path, int] = {}
    for ep_idx in episodes:
        data_path = root / meta.get_data_file_path(ep_idx)
        if not data_path.exists():
            missing[data_path] = missing.get(data_path, 0) + 1
        for camera_key in camera_keys:
            video_path = root / meta.get_video_file_path(ep_idx, camera_key)
            if not video_path.exists():
                missing[video_path] = missing.get(video_path, 0) + 1

    if missing:
        missing_items = sorted(missing.items(), key=lambda item: str(item[0]))
        preview = [
            f"{path} (referenced by {count} episode(s))"
            for path, count in missing_items[:20]
        ]
        suffix = "" if len(missing_items) <= len(preview) else f" ... and {len(missing_items) - len(preview)} more unique file(s)"
        raise FileNotFoundError(
            "Local-only conversion requested, but required local dataset files are missing: "
            f"{preview}{suffix}"
        )


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
    """Return one episode metadata row for local, pandas and HF datasets backends."""
    if isinstance(episodes_table, list):
        row = episodes_table[int(ep_idx)]
        if isinstance(row, dict):
            return row
        return dict(row)
    if hasattr(episodes_table, "iloc"):
        row = episodes_table.iloc[ep_idx]
        if hasattr(row, "to_dict"):
            return row.to_dict()
        return dict(row)
    row = episodes_table[int(ep_idx)]
    if isinstance(row, dict):
        return row
    return dict(row)


def _iter_episode_rows(episodes_table: Any) -> Iterator[dict[str, Any]]:
    if isinstance(episodes_table, list):
        for row in episodes_table:
            yield row if isinstance(row, dict) else dict(row)
        return
    if hasattr(episodes_table, "iloc"):
        for idx in range(len(episodes_table)):
            yield _get_episode_row(episodes_table, idx)
        return
    for row in episodes_table:
        yield row if isinstance(row, dict) else dict(row)


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


def _episode_length_from_row(row: dict[str, Any]) -> int:
    if "dataset_from_index" in row and "dataset_to_index" in row:
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        return end - start
    if "length" in row:
        return int(row["length"])
    raise KeyError(
        "Episode metadata row is missing both "
        "'dataset_from_index/dataset_to_index' and 'length'."
    )


def _episode_source_start_from_row(row: dict[str, Any]) -> int | None:
    if "dataset_from_index" in row:
        return int(row["dataset_from_index"])
    return None


def _build_video_decode_plan(
    meta: Any,
    episodes: list[int],
    camera_keys: list[str],
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
) -> dict[str, list[VideoDecodePlan]]:
    selected_episode_set = set(int(ep) for ep in episodes)
    episode_rows = {
        int(ep_idx): _get_episode_row(meta.episodes, int(ep_idx))
        for ep_idx in episodes
    }
    local_row_map = {
        int(ep_idx): (
            int(ep_len[pos]),
            int(ep_offset[pos]),
            int(ep_offset[pos] + ep_len[pos]),
        )
        for pos, ep_idx in enumerate(episodes)
    }

    # Compute a source-relative frame origin for every shared MP4 from the full metadata,
    # not just the selected subset. This keeps sparse episode subsets aligned.
    video_base_start: dict[tuple[str, str], int] = {}
    for row_dict in _iter_episode_rows(meta.episodes):
        source_start = _episode_source_start_from_row(row_dict)
        if source_start is None:
            continue
        for camera_key in camera_keys:
            key = (
                camera_key,
                str(
                    Path(
                        meta.info["video_path"].format(
                            video_key=camera_key,
                            chunk_index=int(row_dict[f"videos/{camera_key}/chunk_index"]),
                            file_index=int(row_dict[f"videos/{camera_key}/file_index"]),
                        )
                    )
                ),
            )
            current = video_base_start.get(key)
            if current is None or source_start < current:
                video_base_start[key] = source_start

    plan_by_camera: dict[str, list[VideoDecodePlan]] = {camera: [] for camera in camera_keys}
    for camera_key in camera_keys:
        grouped: dict[str, list[VideoEpisodeSlice]] = {}
        for ep_idx in episodes:
            row = episode_rows[int(ep_idx)]
            episode_length, global_row_start, global_row_stop = local_row_map[int(ep_idx)]
            source_start = _episode_source_start_from_row(row)
            video_path = meta.get_video_file_path(int(ep_idx), camera_key)
            group_key = str(video_path)
            if source_start is None:
                selected_before = [
                    other
                    for other in episodes
                    if other in selected_episode_set
                    and str(meta.get_video_file_path(int(other), camera_key)) == group_key
                    and int(other) < int(ep_idx)
                ]
                video_frame_start = sum(local_row_map[int(other)][0] for other in selected_before)
            else:
                base_start = video_base_start.get((camera_key, group_key), source_start)
                video_frame_start = int(source_start - base_start)
            video_frame_stop = video_frame_start + episode_length
            grouped.setdefault(group_key, []).append(
                VideoEpisodeSlice(
                    episode_index=int(ep_idx),
                    episode_length=episode_length,
                    global_row_start=global_row_start,
                    global_row_stop=global_row_stop,
                    video_frame_start=video_frame_start,
                    video_frame_stop=video_frame_stop,
                )
            )

        plans: list[VideoDecodePlan] = []
        for group_key, slices in grouped.items():
            ordered_slices = sorted(
                slices,
                key=lambda item: (item.video_frame_start, item.episode_index),
            )
            previous_stop = 0
            for slice_info in ordered_slices:
                if slice_info.global_row_stop - slice_info.global_row_start != slice_info.episode_length:
                    raise RuntimeError(
                        "Invalid HDF5 row bounds in video decode plan: "
                        f"episode={slice_info.episode_index} "
                        f"rows=[{slice_info.global_row_start}, {slice_info.global_row_stop}) "
                        f"len={slice_info.episode_length}."
                    )
                if slice_info.video_frame_stop - slice_info.video_frame_start != slice_info.episode_length:
                    raise RuntimeError(
                        "Invalid video frame bounds in decode plan: "
                        f"episode={slice_info.episode_index} "
                        f"frames=[{slice_info.video_frame_start}, {slice_info.video_frame_stop}) "
                        f"len={slice_info.episode_length}."
                    )
                if slice_info.video_frame_start < previous_stop:
                    raise RuntimeError(
                        "Video frame ranges overlap or are not monotonic within the same MP4: "
                        f"camera={camera_key} video={group_key} episode={slice_info.episode_index}."
                    )
                previous_stop = slice_info.video_frame_stop
            plans.append(
                VideoDecodePlan(
                    camera_key=camera_key,
                    video_path=Path(group_key),
                    slices=ordered_slices,
                    expected_total_frames=ordered_slices[-1].video_frame_stop,
                )
            )
        plan_by_camera[camera_key] = sorted(plans, key=lambda item: str(item.video_path))

    return plan_by_camera


def _get_compression(compression: str) -> str | None:
    if compression == "none":
        return None
    return compression


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _file_size(path: Path) -> int:
    return int(path.stat().st_size)


def _build_shard_plan(meta: Any, episodes: list[int], shard_episode_count: int) -> list[dict[str, Any]]:
    if shard_episode_count <= 0:
        raise ValueError("shard_episode_count must be > 0.")

    plan: list[dict[str, Any]] = []
    episode_offset = 0
    frame_offset = 0
    shard_index = 0
    for start in range(0, len(episodes), shard_episode_count):
        shard_episodes = episodes[start : start + shard_episode_count]
        ep_len, _ = _episodes_lengths(meta, shard_episodes)
        frame_count = int(ep_len.sum())
        plan.append(
            {
                "shard_id": shard_id_str(shard_index),
                "episode_indices": [int(ep) for ep in shard_episodes],
                "episode_count": len(shard_episodes),
                "episode_offset": episode_offset,
                "frame_offset": frame_offset,
                "frame_count": frame_count,
            }
        )
        episode_offset += len(shard_episodes)
        frame_offset += frame_count
        shard_index += 1
    return plan


def _build_manifest(
    *,
    cfg: ConvertConfig,
    split_plans: dict[str, list[dict[str, Any]]],
    selected_camera_keys: list[str],
    split_done_keys: dict[str, str],
    split_fps: dict[str, int],
) -> dict[str, Any]:
    manifest_splits: dict[str, Any] = {}
    for split_name, shard_plan in split_plans.items():
        shard_entries = []
        episode_to_shard: dict[str, str] = {}
        for shard in shard_plan:
            camera_files = {
                camera: str(shard_camera_relpath(split_name, shard["shard_id"], camera))
                for camera in selected_camera_keys
            }
            shard_entry = {
                "shard_id": shard["shard_id"],
                "episode_indices": shard["episode_indices"],
                "episode_count": shard["episode_count"],
                "episode_offset": shard["episode_offset"],
                "frame_offset": shard["frame_offset"],
                "frame_count": shard["frame_count"],
                "camera_files": camera_files,
                "expected_sizes": {},
            }
            for episode_index in shard["episode_indices"]:
                episode_to_shard[str(int(episode_index))] = shard["shard_id"]
            shard_entries.append(shard_entry)

        manifest_splits[split_name] = {
            "fps": split_fps[split_name],
            "camera_keys": selected_camera_keys,
            "done_key": split_done_keys[split_name],
            "shards": shard_entries,
            "episode_to_shard": episode_to_shard,
        }

    return {
        "format_version": FORMAT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_repo_id": cfg.repo_id,
        "published_repo_id": cfg.publish_repo_id,
        "config": _config_for_summary(cfg),
        "splits": manifest_splits,
    }


def _build_initial_publish_state(
    *,
    cfg: ConvertConfig,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    state = {
        "format_version": FORMAT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_repo_id": cfg.repo_id,
        "published_repo_id": cfg.publish_repo_id,
        "config": _config_for_summary(cfg),
        "splits": {},
    }
    for split_name, split_payload in manifest["splits"].items():
        split_state = {"shards": {}}
        for shard in split_payload["shards"]:
            split_state["shards"][shard["shard_id"]] = {
                "status": "pending",
                "episode_indices": shard["episode_indices"],
                "episode_count": shard["episode_count"],
                "episode_offset": shard["episode_offset"],
                "frame_offset": shard["frame_offset"],
                "frame_count": shard["frame_count"],
                "expected_sizes": {},
                "uploaded_sizes": {},
            }
        state["splits"][split_name] = split_state
    return state


def _merge_existing_state(base_state: dict[str, Any], existing_state: dict[str, Any] | None) -> dict[str, Any]:
    if not existing_state:
        return base_state
    merged = json.loads(json.dumps(base_state))
    for split_name, split_payload in merged["splits"].items():
        old_split = existing_state.get("splits", {}).get(split_name, {})
        old_shards = old_split.get("shards", {})
        for shard_id, shard_state in split_payload["shards"].items():
            if shard_id in old_shards:
                shard_state.update(old_shards[shard_id])
    return merged


def _merge_existing_manifest(base_manifest: dict[str, Any], existing_manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not existing_manifest:
        return base_manifest
    merged = json.loads(json.dumps(base_manifest))
    for split_name, split_payload in merged["splits"].items():
        old_split = existing_manifest.get("splits", {}).get(split_name, {})
        old_by_id = {
            shard["shard_id"]: shard
            for shard in old_split.get("shards", [])
        }
        for shard in split_payload["shards"]:
            old_shard = old_by_id.get(shard["shard_id"])
            if not old_shard:
                continue
            if old_shard.get("expected_sizes"):
                shard["expected_sizes"] = old_shard["expected_sizes"]
    return merged


def _set_manifest_expected_sizes(
    manifest: dict[str, Any],
    split_name: str,
    shard_id: str,
    size_map: dict[str, int],
) -> None:
    for shard in manifest["splits"][split_name]["shards"]:
        if shard["shard_id"] == shard_id:
            shard["expected_sizes"] = {path: int(size) for path, size in size_map.items()}
            return
    raise KeyError(f"Unknown shard_id={shard_id} in split={split_name}.")


def _remote_repo_size_map(api: HfApi, repo_id: str, revision: str | None) -> dict[str, int]:
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        files_metadata=True,
    )
    return remote_file_size_map(info)


def _expected_remote_shard_sizes(
    manifest: dict[str, Any],
    split_name: str,
    shard_id: str,
) -> dict[str, int]:
    for shard in manifest["splits"][split_name]["shards"]:
        if shard["shard_id"] == shard_id:
            return {str(path): int(size) for path, size in shard.get("expected_sizes", {}).items()}
    raise KeyError(f"Unknown shard_id={shard_id} in split={split_name}.")


def _remote_shard_verified(
    manifest: dict[str, Any],
    split_name: str,
    shard_id: str,
    remote_size_map: dict[str, int],
) -> tuple[bool, dict[str, int]]:
    expected_sizes = _expected_remote_shard_sizes(manifest, split_name, shard_id)
    if not expected_sizes:
        return False, {}
    uploaded_sizes: dict[str, int] = {}
    for relpath, expected_size in expected_sizes.items():
        actual_size = remote_size_map.get(relpath)
        if actual_size != expected_size:
            return False, {}
        uploaded_sizes[relpath] = int(actual_size)
    return True, uploaded_sizes


def _ensure_dataset_repo(api: HfApi, repo_id: str, private: bool) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )


def _upload_file_sync(
    api: HfApi,
    *,
    repo_id: str,
    revision: str | None,
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
) -> None:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
        run_as_future=False,
    )


def _delete_local_shard_dir(output_root: Path, split_name: str, shard_id: str) -> None:
    shard_dir = output_root / split_name / shard_id
    if shard_dir.is_dir():
        shutil.rmtree(shard_dir)


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
    columns_ds: Any, row_start: int, row_stop: int, done_key: str
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
            _slice_column_values(columns_ds, done_key, row_start, row_stop),
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
            "Missing dependency 'opencv-python-headless'. "
            "Install with: pip install opencv-python-headless"
        )
    import cv2

    return cv2


def _assert_runtime_dependencies(decode_backend: str, image_size: int | None) -> None:
    _require_h5py()

    available_backends = [
        backend
        for backend in _backend_order(decode_backend)
        if importlib.util.find_spec("av" if backend == "pyav" else "cv2") is not None
    ]
    if not available_backends:
        raise RuntimeError(
            "Missing video decode dependency. Install at least one supported backend: "
            "av for --decode_backend=pyav or opencv-python-headless for "
            "--decode_backend=opencv."
        )

    if image_size is not None:
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
    camera_key: str,
    video_path: Path,
) -> None:
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch on split={split_name}, camera={camera_key}, "
            f"video={video_path}. decoded={decoded_frames}, expected={expected_frames}."
        )


def _assert_slice_bounds_or_raise(
    *,
    split_name: str,
    camera_key: str,
    video_path: Path,
    slice_info: VideoEpisodeSlice,
    total_rows: int,
) -> None:
    if slice_info.global_row_start < 0 or slice_info.global_row_stop > total_rows:
        raise RuntimeError(
            f"Video slice would write outside HDF5 bounds on split={split_name}, "
            f"camera={camera_key}, video={video_path}, episode={slice_info.episode_index}, "
            f"rows=[{slice_info.global_row_start}, {slice_info.global_row_stop}), total_rows={total_rows}."
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
    done_key = _resolve_done_column(source_dataset.features)
    tiny = _select_columns_compat(
        source_dataset, ["episode_index", "frame_index", done_key]
    )
    episode_idx = _column_to_numpy(tiny, "episode_index")
    step_idx = _column_to_numpy(tiny, "frame_index")
    done = _column_to_numpy(tiny, done_key)

    return collect_source_episode_issues(
        episode_idx=episode_idx,
        step_idx=step_idx,
        done=done,
        require_terminal_done=require_terminal_done,
        require_no_early_done=True,
        require_contiguous_steps=True,
    )


def _prevalidate_source_split_local(
    meta: Any,
    root: Path,
    episodes: list[int],
    require_terminal_done: bool,
) -> dict[int, list[str]]:
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as pa_ds
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local-only source prevalidation requires pyarrow. "
            "Install the project requirements or disable --local_only."
        ) from exc

    data_paths = sorted({root / meta.get_data_file_path(ep_idx) for ep_idx in episodes})
    print(
        f"[convert] local prevalidation reading {len(data_paths)} parquet file(s)",
        flush=True,
    )
    dataset = pa_ds.dataset([str(path) for path in data_paths], format="parquet")
    done_key = _resolve_done_column(dataset.schema.names)
    table = dataset.to_table(
        columns=["episode_index", "frame_index", done_key],
        filter=pc.field("episode_index").isin(episodes),
    )

    episode_idx = table["episode_index"].to_numpy(zero_copy_only=False)
    step_idx = table["frame_index"].to_numpy(zero_copy_only=False)
    done = table[done_key].to_numpy(zero_copy_only=False)

    return collect_source_episode_issues(
        episode_idx=episode_idx,
        step_idx=step_idx,
        done=done,
        require_terminal_done=require_terminal_done,
        require_no_early_done=True,
        require_contiguous_steps=True,
    )


def _decode_and_write_video_plan(
    *,
    split_name: str,
    plan: VideoDecodePlan,
    pixels_ds: Any,
    total_rows: int,
    image_size: int | None,
    micro_batch_size: int,
    decode_backend: str,
    stall_timeout_seconds: float,
    heartbeat_seconds: float,
) -> tuple[str, int, int]:
    if plan.expected_total_frames <= 0:
        return "none", 0, 0

    backend_used, frame_iter, first_frame = _open_linear_frame_iterator(
        video_path=plan.video_path,
        decode_backend=decode_backend,
    )
    decoded_frames = 0
    flush_count = 0
    written_frames = 0
    pending_frames: list[np.ndarray] = []
    pending_row_start: int | None = None

    now = time.perf_counter()
    last_progress = now
    last_heartbeat = now

    context = (
        f"split={split_name} cam={plan.camera_key} video={plan.video_path} "
        f"slices={len(plan.slices)}"
    )

    try:
        def _flush_pending() -> None:
            nonlocal flush_count, written_frames, pending_frames, pending_row_start, last_progress
            if not pending_frames:
                return
            if pending_row_start is None:
                raise RuntimeError(f"Missing row start for pending frame batch ({context}).")
            batch = _resize_batch_cv2(pending_frames, image_size)
            row_stop = pending_row_start + int(batch.shape[0])
            if row_stop > total_rows:
                raise RuntimeError(
                    f"Pending video frame batch exceeds HDF5 bounds ({context}). "
                    f"rows=[{pending_row_start}, {row_stop}) total_rows={total_rows}"
                )
            pixels_ds[pending_row_start:row_stop] = batch
            written_frames += int(batch.shape[0])
            flush_count += 1
            pending_frames = []
            pending_row_start = None
            last_progress = time.perf_counter()

        frame_buffer: list[np.ndarray] = [first_frame]
        frame_index = 0
        slice_index = 0

        while frame_index < plan.expected_total_frames:
            now = time.perf_counter()
            _check_stall_or_raise(
                now=now,
                last_progress=last_progress,
                stall_timeout_seconds=stall_timeout_seconds,
                context=context,
            )
            if frame_index > 0:
                try:
                    frame = next(frame_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Video ended early ({context}). decoded={decoded_frames}, "
                        f"expected={plan.expected_total_frames}"
                    ) from exc
                frame_buffer = [frame]

            current_frame = frame_buffer[0]
            decoded_frames += 1
            last_progress = now

            while (
                slice_index < len(plan.slices)
                and frame_index >= plan.slices[slice_index].video_frame_stop
            ):
                slice_index += 1

            if slice_index < len(plan.slices):
                slice_info = plan.slices[slice_index]
                _assert_slice_bounds_or_raise(
                    split_name=split_name,
                    camera_key=plan.camera_key,
                    video_path=plan.video_path,
                    slice_info=slice_info,
                    total_rows=total_rows,
                )
                if slice_info.video_frame_start <= frame_index < slice_info.video_frame_stop:
                    row_index = slice_info.global_row_start + (frame_index - slice_info.video_frame_start)
                    if row_index >= slice_info.global_row_stop:
                        raise RuntimeError(
                            f"Computed row_index exceeds episode slice bounds on split={split_name}, "
                            f"camera={plan.camera_key}, video={plan.video_path}, "
                            f"episode={slice_info.episode_index}, row_index={row_index}."
                        )
                    expected_next_row = (
                        pending_row_start + len(pending_frames)
                        if pending_row_start is not None
                        else row_index
                    )
                    if pending_row_start is not None and row_index != expected_next_row:
                        _flush_pending()
                    if pending_row_start is None:
                        pending_row_start = row_index
                    pending_frames.append(current_frame)
                    if len(pending_frames) >= micro_batch_size:
                        _flush_pending()
                elif pending_frames:
                    _flush_pending()
            elif pending_frames:
                _flush_pending()

            now = time.perf_counter()
            if now - last_heartbeat >= heartbeat_seconds:
                print(
                    f"[convert][{split_name}] heartbeat cam={plan.camera_key} "
                    f"video={plan.video_path.name} decoded={decoded_frames}/{plan.expected_total_frames} "
                    f"written={written_frames}",
                    flush=True,
                )
                last_heartbeat = now
            frame_index += 1

        _flush_pending()

        _assert_frame_count_or_raise(
            decoded_frames=decoded_frames,
            expected_frames=plan.expected_total_frames,
            split_name=split_name,
            camera_key=plan.camera_key,
            video_path=plan.video_path,
        )
        if written_frames != sum(slice_info.episode_length for slice_info in plan.slices):
            raise RuntimeError(
                f"Video write mismatch ({context}). written={written_frames}, "
                f"expected={sum(slice_info.episode_length for slice_info in plan.slices)}"
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
    file_layout: str = "legacy",
    shard_metadata: dict[str, Any] | None = None,
) -> tuple[list[str], int, str, int, dict[str, Any]]:
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
    video_decode_plan = _build_video_decode_plan(
        ds.meta,
        episodes,
        camera_keys,
        ep_len,
        ep_offset,
    )

    done_key = _resolve_done_column(ds.features)
    cols = _select_columns_compat(ds, [*TABULAR_COLUMNS, done_key])

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

        action_probe = _read_episode_tabular_slice(cols, 0, min(1, total_rows), done_key)["action"]
        state_probe = _read_episode_tabular_slice(cols, 0, min(1, total_rows), done_key)["state"]
        action_dim = int(action_probe.shape[1])
        state_dim = int(state_probe.shape[1])

        for camera in camera_keys:
            camera_slug = camera.replace(".", "_")
            if file_layout == "legacy":
                out_path = output_dir / f"{split_name}__{camera_slug}.h5"
            elif file_layout == "shard":
                out_path = output_dir / f"{camera_slug}.h5"
            else:
                raise ValueError(f"Unsupported file_layout '{file_layout}'.")
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
            h5f.attrs["source_done_key"] = done_key
            h5f.attrs["fps"] = int(ds.meta.fps)
            h5f.attrs["generated_by"] = "convert_lerobot_to_hdf5.py"
            if shard_metadata is not None:
                h5f.attrs["shard_id"] = shard_metadata["shard_id"]
                h5f.attrs["shard_episode_offset"] = int(shard_metadata["episode_offset"])
                h5f.attrs["shard_frame_offset"] = int(shard_metadata["frame_offset"])
            h5f.attrs["video_file_count"] = int(len(video_decode_plan[camera]))
            h5f.attrs["video_segment_count"] = int(len(video_decode_plan[camera]))
            h5f.attrs["video_frame_write_count"] = 0

        started = time.perf_counter()
        last_log = started
        cursor = 0
        rows_written = 0
        backend_counts: dict[str, int] = {}
        camera_decode_stats = {
            camera: {
                "video_file_count": len(video_decode_plan[camera]),
                "video_segment_count": len(video_decode_plan[camera]),
                "decoded_frames": 0,
                "written_frames": 0,
                "backend_counts": {},
            }
            for camera in camera_keys
        }

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

            tabular = _read_episode_tabular_slice(cols, row_start, row_stop, done_key)
            _write_tabular_slice_to_all_files(file_handles, row_slice, tabular)

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

        for camera in camera_keys:
            for plan in video_decode_plan[camera]:
                backend_used, decoded, flushes = _decode_and_write_video_plan(
                    split_name=split_name,
                    plan=VideoDecodePlan(
                        camera_key=plan.camera_key,
                        video_path=root / plan.video_path,
                        slices=plan.slices,
                        expected_total_frames=plan.expected_total_frames,
                    ),
                    pixels_ds=file_handles[camera]["pixels"],
                    total_rows=total_rows,
                    image_size=cfg.image_size,
                    micro_batch_size=cfg.micro_batch_size,
                    decode_backend=decode_backend,
                    stall_timeout_seconds=cfg.stall_timeout_seconds,
                    heartbeat_seconds=cfg.heartbeat_seconds,
                )
                backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1
                camera_stats = camera_decode_stats[camera]
                camera_stats["decoded_frames"] += decoded
                camera_stats["written_frames"] += sum(slice_info.episode_length for slice_info in plan.slices)
                camera_stats["backend_counts"][backend_used] = (
                    camera_stats["backend_counts"].get(backend_used, 0) + 1
                )
                print(
                    f"[convert][{split_name}] video plan done cam={camera} "
                    f"video={plan.video_path} decoded={decoded} "
                    f"slices={len(plan.slices)} flushes={flushes}",
                    flush=True,
                )

            file_handles[camera].attrs["video_frame_write_count"] = int(
                camera_decode_stats[camera]["written_frames"]
            )
            file_handles[camera].attrs["video_backend_counts"] = json.dumps(
                camera_decode_stats[camera]["backend_counts"],
                sort_keys=True,
            )

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

    for camera in camera_keys:
        expected_written = int(total_rows)
        actual_written = int(camera_decode_stats[camera]["written_frames"])
        if actual_written != expected_written:
            raise RuntimeError(
                f"Pixel write mismatch for split={split_name}, camera={camera}. "
                f"written={actual_written}, expected={expected_written}."
            )

    video_plan_summary = {
        camera: {
            "video_file_count": int(camera_decode_stats[camera]["video_file_count"]),
            "video_segment_count": int(camera_decode_stats[camera]["video_segment_count"]),
            "decoded_frames": int(camera_decode_stats[camera]["decoded_frames"]),
            "written_frames": int(camera_decode_stats[camera]["written_frames"]),
            "backend_counts": dict(camera_decode_stats[camera]["backend_counts"]),
        }
        for camera in camera_keys
    }

    print(f"[convert][{split_name}] finished. files={produced_files}", flush=True)
    return produced_files, total_rows, done_key, int(ds.meta.fps), video_plan_summary


def _process_sharded_split(
    *,
    cfg: ConvertConfig,
    split_name: str,
    shard_plan: list[dict[str, Any]],
    raw_root: Path,
    output_root: Path,
    camera_keys: list[str],
    decode_backend: str,
    manifest: dict[str, Any],
    state: dict[str, Any],
    report: dict[str, Any],
    api: HfApi | None,
    report_path: Path,
) -> list[str]:
    published_files: list[str] = []
    split_report = report["splits"][split_name]
    split_state = state["splits"][split_name]["shards"]
    manifest_local_path = manifest_path(output_root)
    state_local_path = state_path(output_root)

    remote_sizes: dict[str, int] = {}
    if cfg.publish_repo_id and api is not None:
        _ensure_dataset_repo(api, cfg.publish_repo_id, cfg.hf_private)
        remote_sizes = _remote_repo_size_map(api, cfg.publish_repo_id, cfg.publish_revision)

    for shard in shard_plan:
        shard_id = shard["shard_id"]
        shard_state = split_state[shard_id]
        verified_remote, uploaded_sizes = _remote_shard_verified(
            manifest,
            split_name,
            shard_id,
            remote_sizes,
        )
        if cfg.resume and verified_remote:
            shard_state["status"] = "uploaded"
            shard_state["uploaded_sizes"] = uploaded_sizes
            published_files.extend(uploaded_sizes.keys())
            print(
                f"[convert][{split_name}] resume skip shard={shard_id} "
                f"(already uploaded and size-verified)",
                flush=True,
            )
            continue

        shard_output_dir = output_root / split_name / shard_id
        if shard_output_dir.exists():
            shutil.rmtree(shard_output_dir)
        produced_files, frame_count, done_key, fps, video_plan_summary = _convert_split(
            cfg=cfg,
            split_name=split_name,
            episodes=shard["episode_indices"],
            root=raw_root,
            output_dir=shard_output_dir,
            camera_keys=camera_keys,
            decode_backend=decode_backend,
            file_layout="shard",
            shard_metadata=shard,
        )
        rel_files = {
            camera: str(shard_camera_relpath(split_name, shard_id, camera))
            for camera in camera_keys
        }
        expected_sizes = {
            rel_files[camera]: _file_size(Path(local_path))
            for camera, local_path in zip(camera_keys, produced_files, strict=True)
        }
        _set_manifest_expected_sizes(manifest, split_name, shard_id, expected_sizes)
        shard_state["status"] = "converted"
        shard_state["expected_sizes"] = expected_sizes
        shard_state["done_key"] = done_key
        shard_state["fps"] = fps
        shard_state["video_decode_plan_summary"] = video_plan_summary
        split_report.setdefault("shards", []).append(
            {
                "shard_id": shard_id,
                "episode_indices": shard["episode_indices"],
                "episode_count": shard["episode_count"],
                "frame_offset": shard["frame_offset"],
                "frame_count": frame_count,
                "produced_files": list(expected_sizes.keys()),
                "video_decode_plan_summary": video_plan_summary,
            }
        )

        _write_json_atomic(manifest_local_path, manifest)
        _write_json_atomic(state_local_path, state)
        _write_report(report_path, report)

        if cfg.publish_repo_id and api is not None:
            for camera_key, local_path in zip(camera_keys, produced_files, strict=True):
                relpath = rel_files[camera_key]
                _upload_file_sync(
                    api,
                    repo_id=cfg.publish_repo_id,
                    revision=cfg.publish_revision,
                    local_path=Path(local_path),
                    path_in_repo=relpath,
                    commit_message=f"Upload {split_name} {shard_id} {camera_slug(camera_key)}",
                )

            if cfg.upload_manifest_every_shard:
                _upload_file_sync(
                    api,
                    repo_id=cfg.publish_repo_id,
                    revision=cfg.publish_revision,
                    local_path=manifest_local_path,
                    path_in_repo=manifest_local_path.name,
                    commit_message=f"Update manifest after {split_name} {shard_id}",
                )
                _upload_file_sync(
                    api,
                    repo_id=cfg.publish_repo_id,
                    revision=cfg.publish_revision,
                    local_path=report_path,
                    path_in_repo=report_path.name,
                    commit_message=f"Update report after {split_name} {shard_id}",
                )

            remote_sizes = _remote_repo_size_map(api, cfg.publish_repo_id, cfg.publish_revision)
            verified_remote, uploaded_sizes = _remote_shard_verified(
                manifest,
                split_name,
                shard_id,
                remote_sizes,
            )
            if not verified_remote:
                raise RuntimeError(
                    f"Remote verification failed for split={split_name} shard={shard_id}. "
                    "At least one uploaded file is missing or has the wrong size."
                )
            shard_state["status"] = "uploaded"
            shard_state["uploaded_sizes"] = uploaded_sizes
            published_files.extend(uploaded_sizes.keys())
            _write_json_atomic(state_local_path, state)
            _write_report(report_path, report)

            if cfg.cleanup_local_shards_after_upload:
                _delete_local_shard_dir(output_root, split_name, shard_id)
        else:
            published_files.extend(expected_sizes.keys())

    if cfg.publish_repo_id and api is not None and not cfg.upload_manifest_every_shard:
        _upload_file_sync(
            api,
            repo_id=cfg.publish_repo_id,
            revision=cfg.publish_revision,
            local_path=manifest_local_path,
            path_in_repo=manifest_local_path.name,
            commit_message=f"Finalize manifest for {split_name}",
        )
        _upload_file_sync(
            api,
            repo_id=cfg.publish_repo_id,
            revision=cfg.publish_revision,
            local_path=report_path,
            path_in_repo=report_path.name,
            commit_message=f"Finalize report for {split_name}",
        )

    return published_files


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
    if cfg.shard_episode_count is not None and cfg.shard_episode_count <= 0:
        raise ValueError("shard_episode_count must be > 0 when provided.")
    if cfg.publish_repo_id and cfg.shard_episode_count is None:
        raise ValueError("publish_repo_id currently requires shard_episode_count.")

    decode_backend = _resolve_decode_backend(cfg)
    if decode_backend not in VALID_DECODE_BACKEND:
        raise ValueError(
            f"Invalid decode_backend '{decode_backend}'. Expected one of {VALID_DECODE_BACKEND}."
        )

    _assert_runtime_dependencies(decode_backend=decode_backend, image_size=cfg.image_size)
    _warn_deprecated_options(cfg)

    raw_root = Path(cfg.datasets_dir) / cfg.raw_subdir / cfg.repo_id
    output_root = Path(cfg.datasets_dir) / cfg.hdf5_subdir / sanitize_repo_id(cfg.repo_id)
    report_path = output_root / cfg.report_filename

    print(
        f"[convert] resolved paths raw_root={raw_root} output_root={output_root}",
        flush=True,
    )

    if cfg.local_only:
        print(f"[convert] checking local metadata under {raw_root}", flush=True)
        _assert_local_metadata_exists(raw_root)

    print(
        f"[convert] loading dataset metadata source={'local' if cfg.local_only else 'lerobot'} "
        f"repo_id={cfg.repo_id}",
        flush=True,
    )
    if cfg.local_only:
        meta = LocalLeRobotMetadata(raw_root)
    else:
        meta = LeRobotDatasetMetadata(
            repo_id=cfg.repo_id,
            root=raw_root,
        )
    print(
        f"[convert] metadata loaded total_episodes={meta.total_episodes} "
        f"total_frames={meta.total_frames} fps={int(meta.fps)}",
        flush=True,
    )
    all_split_names = list(meta.info["splits"].keys())
    split_names = cfg.splits or all_split_names
    print(f"[convert] selected splits={split_names}", flush=True)
    for split_name in split_names:
        if split_name not in meta.info["splits"]:
            raise ValueError(
                f"Unknown split '{split_name}'. Available splits: {all_split_names}"
            )

    all_camera_keys = list(meta.camera_keys)
    selected_camera_keys = cfg.camera_keys or all_camera_keys
    print(
        f"[convert] selected cameras={selected_camera_keys} "
        f"(available={len(all_camera_keys)})",
        flush=True,
    )
    for camera_key in selected_camera_keys:
        if camera_key not in all_camera_keys:
            raise ValueError(
                f"Unknown camera key '{camera_key}'. Available: {all_camera_keys}"
            )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": cfg.repo_id,
        "config": _config_for_summary(cfg),
        "resolved_decode_backend": decode_backend,
        "splits": {},
        "status": "ok",
    }

    sharded_mode = cfg.shard_episode_count is not None
    split_plans: dict[str, list[dict[str, Any]]] = {}
    split_done_keys: dict[str, str] = {}
    split_fps: dict[str, int] = {}
    manifest: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    api: HfApi | None = HfApi() if cfg.publish_repo_id else None

    print(
        "[convert] starting "
        f"repo_id={cfg.repo_id} splits={split_names} "
        f"decode_backend={decode_backend} micro_batch={cfg.micro_batch_size}",
        flush=True,
    )

    for split_name in split_names:
        split_episodes = _split_episodes(meta, split_name)
        print(
            f"[convert][{split_name}] split resolved source_episode_count={len(split_episodes)}",
            flush=True,
        )
        requested_episodes = _select_episode_indices(
            split_episodes,
            episode_indices=cfg.episode_indices,
            episode_start=cfg.episode_start,
            episode_count=cfg.episode_count,
        )
        print(
            f"[convert][{split_name}] requested episodes after CLI filters={len(requested_episodes)}",
            flush=True,
        )
        if cfg.local_only:
            print(
                f"[convert][{split_name}] checking local parquet/mp4 presence "
                f"for {len(requested_episodes)} episodes and {len(selected_camera_keys)} cameras",
                flush=True,
            )
            _assert_local_episode_files(
                root=raw_root,
                meta=meta,
                episodes=requested_episodes,
                camera_keys=selected_camera_keys,
            )
            print(f"[convert][{split_name}] local file presence check passed", flush=True)
        if cfg.prevalidate_source:
            print(
                f"[convert][{split_name}] pre-validating {len(requested_episodes)} episodes...",
                flush=True,
            )
            if cfg.local_only:
                issues = _prevalidate_source_split_local(
                    meta=meta,
                    root=raw_root,
                    episodes=requested_episodes,
                    require_terminal_done=cfg.require_terminal_done,
                )
            else:
                issues = _prevalidate_source_split(
                    repo_id=cfg.repo_id,
                    root=raw_root,
                    episodes=requested_episodes,
                    require_terminal_done=cfg.require_terminal_done,
                )
        else:
            print(
                f"[convert][{split_name}] source prevalidation skipped "
                f"(episodes={len(requested_episodes)})",
                flush=True,
            )
            issues = {}
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
            f"[convert][{split_name}] kept_episodes={len(kept_episodes)} "
            f"dirty_policy={cfg.dirty_episode_policy}",
            flush=True,
        )
        if sharded_mode:
            split_plans[split_name] = _build_shard_plan(meta, kept_episodes, cfg.shard_episode_count or 0)
            split_done_keys[split_name] = _resolve_done_column(meta.features)
            split_fps[split_name] = int(meta.fps)
            split_report["shards"] = []
            print(
                f"[convert][{split_name}] shard plan built shards={len(split_plans[split_name])} "
                f"shard_episode_count={cfg.shard_episode_count}",
                flush=True,
            )
        else:
            print(
                f"[convert][{split_name}] converting with kept_episodes={len(kept_episodes)}",
                flush=True,
            )
            produced, _, _, _, video_plan_summary = _convert_split(
                cfg=cfg,
                split_name=split_name,
                episodes=kept_episodes,
                root=raw_root,
                output_dir=output_root,
                camera_keys=selected_camera_keys,
                decode_backend=decode_backend,
            )
            split_report["produced_files"] = produced
            split_report["video_decode_plan_summary"] = video_plan_summary

    if sharded_mode:
        print("[convert] building manifest/state for shard mode", flush=True)
        manifest = _build_manifest(
            cfg=cfg,
            split_plans=split_plans,
            selected_camera_keys=selected_camera_keys,
            split_done_keys=split_done_keys,
            split_fps=split_fps,
        )
        if cfg.resume:
            print("[convert] loading previous manifest/state for resume if present", flush=True)
        manifest = _merge_existing_manifest(
            manifest,
            load_manifest(output_root) if cfg.resume and manifest_path(output_root).is_file() else None,
        )
        state = _merge_existing_state(
            _build_initial_publish_state(cfg=cfg, manifest=manifest),
            load_state(output_root) if cfg.resume else None,
        )
        _write_json_atomic(manifest_path(output_root), manifest)
        _write_json_atomic(state_path(output_root), state)
        _write_report(report_path, report)
        print(
            f"[convert] shard bootstrap written manifest={manifest_path(output_root)} "
            f"state={state_path(output_root)}",
            flush=True,
        )

        for split_name in split_names:
            print(
                f"[convert][{split_name}] converting shard plan "
                f"(shards={len(split_plans[split_name])}, episodes={len(report['splits'][split_name]['kept_episodes'])})",
                flush=True,
            )
            published = _process_sharded_split(
                cfg=cfg,
                split_name=split_name,
                shard_plan=split_plans[split_name],
                raw_root=raw_root,
                output_root=output_root,
                camera_keys=selected_camera_keys,
                decode_backend=decode_backend,
                manifest=manifest,
                state=state,
                report=report,
                api=api,
                report_path=report_path,
            )
            report["splits"][split_name]["produced_files"] = published

    _write_report(report_path, report)
    if sharded_mode and manifest is not None and state is not None:
        _write_json_atomic(manifest_path(output_root), manifest)
        _write_json_atomic(state_path(output_root), state)
    print(f"[convert] done repo_id={cfg.repo_id}", flush=True)
    print(f"[convert] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
