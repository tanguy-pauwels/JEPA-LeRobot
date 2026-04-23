from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

MODULE = importlib.import_module("scripts.convert_lerobot_to_hdf5")


class FakeMeta:
    def get_data_file_path(self, ep_idx: int) -> Path:
        return Path(f"data/chunk-000/file-{ep_idx:03d}.parquet")

    def get_video_file_path(self, ep_idx: int, camera_key: str) -> Path:
        return Path(f"videos/{camera_key}/chunk-000/file-{ep_idx:03d}.mp4")


class ConvertLocalOnlyTests(unittest.TestCase):
    def test_assert_local_episode_files_passes_when_required_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "videos/observation.images.left/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000/file-000.parquet").touch()
            (root / "videos/observation.images.left/chunk-000/file-000.mp4").touch()

            MODULE._assert_local_episode_files(
                root=root,
                meta=FakeMeta(),
                episodes=[0],
                camera_keys=["observation.images.left"],
            )

    def test_assert_local_episode_files_fails_when_camera_video_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000/file-000.parquet").touch()

            with self.assertRaises(FileNotFoundError):
                MODULE._assert_local_episode_files(
                    root=root,
                    meta=FakeMeta(),
                    episodes=[0],
                    camera_keys=["observation.images.left"],
                )


if __name__ == "__main__":
    unittest.main()
