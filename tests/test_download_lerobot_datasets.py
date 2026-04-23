from __future__ import annotations

from pathlib import Path
import unittest

MODULE_PATH = "scripts.download_lerobot_datasets"


class FakeMeta:
    total_episodes = 100
    camera_keys = [
        "observation.images.left",
        "observation.images.right",
        "observation.images.wrist",
    ]

    def get_data_file_path(self, ep_index: int) -> Path:
        return Path(f"data/chunk-000/file-{ep_index // 10:03d}.parquet")

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        camera_slug = vid_key
        return Path(f"videos/{camera_slug}/chunk-000/file-{ep_index:03d}.mp4")


class DownloadSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        import importlib

        self.module = importlib.import_module(MODULE_PATH)

    def test_episode_start_and_count_selects_range(self) -> None:
        episodes = self.module._select_episode_indices(
            100,
            episode_indices=[],
            episode_start=0,
            episode_count=10,
        )
        self.assertEqual(list(range(10)), episodes)

    def test_episode_indices_preserve_order(self) -> None:
        episodes = self.module._select_episode_indices(
            100,
            episode_indices=[2, 5, 9],
            episode_start=None,
            episode_count=None,
        )
        self.assertEqual([2, 5, 9], episodes)

    def test_episode_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self.module._select_episode_indices(
                100,
                episode_indices=[],
                episode_start=0,
                episode_count=0,
            )

    def test_rejects_unknown_camera(self) -> None:
        with self.assertRaises(ValueError):
            self.module._resolve_camera_keys(
                FakeMeta(),
                ["observation.images.missing"],
                download_videos=True,
            )

    def test_allow_patterns_keep_only_selected_camera_videos(self) -> None:
        patterns = self.module._build_selective_allow_patterns(
            FakeMeta(),
            episodes=[0, 1],
            camera_keys=["observation.images.left"],
        )

        self.assertIn("meta/*", patterns)
        self.assertIn("meta/**", patterns)
        self.assertIn("data/chunk-000/file-000.parquet", patterns)
        self.assertIn(
            "videos/observation.images.left/chunk-000/file-000.mp4",
            patterns,
        )
        self.assertNotIn(
            "videos/observation.images.right/chunk-000/file-000.mp4",
            patterns,
        )
        self.assertNotIn(
            "videos/observation.images.wrist/chunk-000/file-000.mp4",
            patterns,
        )

    def test_download_videos_false_selects_no_cameras(self) -> None:
        cameras = self.module._resolve_camera_keys(
            FakeMeta(),
            ["observation.images.left"],
            download_videos=False,
        )
        self.assertEqual([], cameras)


if __name__ == "__main__":
    unittest.main()
