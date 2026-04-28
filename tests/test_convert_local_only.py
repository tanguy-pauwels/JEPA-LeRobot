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
        file_index = 0 if int(ep_idx) < 2 else int(ep_idx)
        return Path(f"videos/{camera_key}/chunk-000/file-{file_index:03d}.mp4")


class ConvertLocalOnlyTests(unittest.TestCase):
    def test_select_episode_start_and_count(self) -> None:
        selected = MODULE._select_episode_indices(
            list(range(100)),
            episode_indices=[],
            episode_start=0,
            episode_count=10,
        )
        self.assertEqual(list(range(10)), selected)

    def test_select_episode_indices_preserves_order(self) -> None:
        selected = MODULE._select_episode_indices(
            list(range(100)),
            episode_indices=[2, 5, 9],
            episode_start=None,
            episode_count=None,
        )
        self.assertEqual([2, 5, 9], selected)

    def test_resolve_done_column_supports_droid_terminal_key(self) -> None:
        done_key = MODULE._resolve_done_column(
            ["episode_index", "frame_index", "is_terminal", "is_last"]
        )
        self.assertEqual("is_terminal", done_key)

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

    def test_assert_local_episode_files_reports_shared_missing_video_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000/file-000.parquet").touch()
            (root / "data/chunk-000/file-001.parquet").touch()

            with self.assertRaises(FileNotFoundError) as ctx:
                MODULE._assert_local_episode_files(
                    root=root,
                    meta=FakeMeta(),
                    episodes=[0, 1],
                    camera_keys=["observation.images.left"],
                )
            self.assertIn("referenced by 2 episode(s)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
