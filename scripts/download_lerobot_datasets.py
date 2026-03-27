#!/usr/bin/env python3
"""Download LeRobot datasets from Hugging Face into datasets/raw/."""

from dataclasses import dataclass, field
from pathlib import Path

import draccus
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class DownloadConfig:
    repo_ids: list[str] = field(
        default_factory=lambda: ["lerobot/koch_pick_place_1_lego"]
    )
    datasets_dir: str = "datasets"
    raw_subdir: str = "raw"
    download_videos: bool = True
    force_cache_sync: bool = False
    revision: str | None = None


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    base_raw_dir = Path(cfg.datasets_dir) / cfg.raw_subdir
    base_raw_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in cfg.repo_ids:
        local_root = base_raw_dir / repo_id
        local_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] repo_id={repo_id} -> {local_root}")

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
