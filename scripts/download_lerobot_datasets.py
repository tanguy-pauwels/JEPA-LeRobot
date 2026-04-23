#!/usr/bin/env python3
"""Download LeRobot datasets from Hugging Face into datasets/raw/."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@dataclass
class DownloadConfig:
    repo_ids: list[str] = field(
        default_factory=lambda: ["lerobot/koch_pick_place_1_lego"]
    )
    datasets_dir: str = "datasets"
    raw_subdir: str = "raw"
    download_videos: bool = True
    camera_keys: list[str] = field(default_factory=list)
    episode_indices: list[int] = field(default_factory=list)
    episode_start: int | None = None
    episode_count: int | None = None
    force_cache_sync: bool = False
    revision: str | None = None


def _get_episode_row(episodes_table: Any, ep_idx: int) -> dict[str, Any]:
    if hasattr(episodes_table, "iloc"):
        row = episodes_table.iloc[ep_idx]
        if hasattr(row, "to_dict"):
            return row.to_dict()
        return dict(row)
    row = episodes_table[int(ep_idx)]
    if isinstance(row, dict):
        return row
    return dict(row)


def _select_episode_indices(
    total_episodes: int,
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
        invalid = [ep for ep in selected if ep < 0 or ep >= total_episodes]
        if invalid:
            raise ValueError(
                f"Episode indices out of range: {invalid}. "
                f"Valid range is 0..{total_episodes - 1}."
            )
        return selected

    start = 0 if episode_start is None else int(episode_start)
    if start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}.")
    if episode_count is not None and int(episode_count) <= 0:
        raise ValueError(f"episode_count must be > 0, got {episode_count}.")
    if start > total_episodes:
        raise ValueError(
            f"episode_start={start} exceeds total episodes ({total_episodes})."
        )

    stop = total_episodes if episode_count is None else start + int(episode_count)
    if stop > total_episodes:
        raise ValueError(
            f"Requested episode range [{start}, {stop}) exceeds "
            f"total episodes ({total_episodes})."
        )
    return list(range(start, stop))


def _resolve_camera_keys(meta: Any, camera_keys: list[str], download_videos: bool) -> list[str]:
    if not download_videos:
        return []

    available = list(meta.camera_keys)
    selected = camera_keys or available
    unknown = [key for key in selected if key not in available]
    if unknown:
        raise ValueError(f"Unknown camera key(s): {unknown}. Available: {available}")
    return selected


def _path_pattern(path: Path | str) -> str:
    return Path(path).as_posix()


def _build_selective_allow_patterns(
    meta: Any,
    *,
    episodes: list[int],
    camera_keys: list[str],
) -> list[str]:
    patterns: set[str] = {"meta/*", "meta/**"}
    for ep_idx in episodes:
        patterns.add(_path_pattern(meta.get_data_file_path(ep_idx)))
        for camera_key in camera_keys:
            patterns.add(_path_pattern(meta.get_video_file_path(ep_idx, camera_key)))
    return sorted(patterns)


def _episode_frame_count(meta: Any, episodes: list[int]) -> int:
    total = 0
    for ep_idx in episodes:
        row = _get_episode_row(meta.episodes, ep_idx)
        if "dataset_from_index" in row and "dataset_to_index" in row:
            total += int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        elif "length" in row:
            total += int(row["length"])
    return total


def _has_selective_filters(cfg: DownloadConfig) -> bool:
    return bool(cfg.camera_keys or cfg.episode_indices or cfg.episode_start is not None or cfg.episode_count is not None)


def _download_selective(repo_id: str, local_root: Path, cfg: DownloadConfig) -> None:
    meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=local_root,
        revision=cfg.revision,
        force_cache_sync=cfg.force_cache_sync,
    )
    episodes = _select_episode_indices(
        int(meta.total_episodes),
        episode_indices=cfg.episode_indices,
        episode_start=cfg.episode_start,
        episode_count=cfg.episode_count,
    )
    cameras = _resolve_camera_keys(meta, cfg.camera_keys, cfg.download_videos)
    allow_patterns = _build_selective_allow_patterns(
        meta,
        episodes=episodes,
        camera_keys=cameras,
    )

    print(
        f"[download] selective episodes={len(episodes)} cameras={cameras or 'none'} "
        f"files={len(allow_patterns)}",
        flush=True,
    )
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=meta.revision,
        local_dir=local_root,
        allow_patterns=allow_patterns,
        force_download=cfg.force_cache_sync,
    )
    print(
        f"[download] done repo_id={repo_id} "
        f"(episodes={len(episodes)}, frames={_episode_frame_count(meta, episodes)})"
    )


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    base_raw_dir = Path(cfg.datasets_dir) / cfg.raw_subdir
    base_raw_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in cfg.repo_ids:
        local_root = base_raw_dir / repo_id
        local_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] repo_id={repo_id} -> {local_root}")

        if _has_selective_filters(cfg):
            _download_selective(repo_id, local_root, cfg)
            continue

        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=local_root,
            download_videos=cfg.download_videos,
            force_cache_sync=cfg.force_cache_sync,
            revision=cfg.revision,
        )
        print(
            f"[download] done repo_id={repo_id} "
            f"(episodes={dataset.num_episodes}, frames={dataset.num_frames})"
        )


if __name__ == "__main__":
    main()
