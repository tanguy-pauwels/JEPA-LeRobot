from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

from lewm_dataset_utils.sharded_hdf5 import (
    atomic_write_json,
    remote_file_size_map,
    resolve_manifest_hdf5_files,
    shard_camera_relpath,
)

CONVERT = importlib.import_module("scripts.convert_lerobot_to_hdf5")


class _FakeSibling:
    def __init__(self, rfilename: str, size: int | None = None) -> None:
        self.rfilename = rfilename
        self.size = size
        self.lfs = None


class _FakeRepoInfo:
    def __init__(self, siblings: list[_FakeSibling]) -> None:
        self.siblings = siblings


class _FakeMeta:
    def __init__(self) -> None:
        self.episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 2},
            {"dataset_from_index": 2, "dataset_to_index": 5},
            {"dataset_from_index": 5, "dataset_to_index": 9},
            {"dataset_from_index": 9, "dataset_to_index": 10},
        ]


class ShardedConvertHelperTests(unittest.TestCase):
    def test_config_for_summary_redacts_local_path_fields(self) -> None:
        cfg = CONVERT.ConvertConfig(
            repo_id="lerobot/droid_1.0.1",
            datasets_dir="/private/tmp/datasets",
            raw_subdir="raw",
            hdf5_subdir="hdf5",
        )
        payload = CONVERT._config_for_summary(cfg)
        self.assertNotIn("datasets_dir", payload)
        self.assertNotIn("raw_subdir", payload)
        self.assertNotIn("hdf5_subdir", payload)
        self.assertEqual("lerobot/droid_1.0.1", payload["repo_id"])

    def test_build_shard_plan_tracks_offsets(self) -> None:
        plan = CONVERT._build_shard_plan(_FakeMeta(), [0, 1, 2, 3], 2)
        self.assertEqual(2, len(plan))
        self.assertEqual("shard-00000", plan[0]["shard_id"])
        self.assertEqual([0, 1], plan[0]["episode_indices"])
        self.assertEqual(0, plan[0]["episode_offset"])
        self.assertEqual(0, plan[0]["frame_offset"])
        self.assertEqual(5, plan[0]["frame_count"])
        self.assertEqual("shard-00001", plan[1]["shard_id"])
        self.assertEqual([2, 3], plan[1]["episode_indices"])
        self.assertEqual(2, plan[1]["episode_offset"])
        self.assertEqual(5, plan[1]["frame_offset"])
        self.assertEqual(5, plan[1]["frame_count"])

    def test_remote_shard_verified_requires_exact_size_match(self) -> None:
        manifest = {
            "splits": {
                "train": {
                    "shards": [
                        {
                            "shard_id": "shard-00000",
                            "expected_sizes": {
                                "train/shard-00000/observation_images_front.h5": 123
                            },
                        }
                    ]
                }
            }
        }
        ok, uploaded = CONVERT._remote_shard_verified(
            manifest,
            "train",
            "shard-00000",
            {"train/shard-00000/observation_images_front.h5": 123},
        )
        self.assertTrue(ok)
        self.assertEqual({"train/shard-00000/observation_images_front.h5": 123}, uploaded)

        bad, _ = CONVERT._remote_shard_verified(
            manifest,
            "train",
            "shard-00000",
            {"train/shard-00000/observation_images_front.h5": 122},
        )
        self.assertFalse(bad)

    def test_merge_existing_manifest_preserves_expected_sizes(self) -> None:
        base = {
            "splits": {
                "train": {
                    "shards": [{"shard_id": "shard-00000", "expected_sizes": {}}]
                }
            }
        }
        existing = {
            "splits": {
                "train": {
                    "shards": [
                        {
                            "shard_id": "shard-00000",
                            "expected_sizes": {"train/shard-00000/front.h5": 42},
                        }
                    ]
                }
            }
        }
        merged = CONVERT._merge_existing_manifest(base, existing)
        self.assertEqual(
            {"train/shard-00000/front.h5": 42},
            merged["splits"]["train"]["shards"][0]["expected_sizes"],
        )


class ShardedManifestHelperTests(unittest.TestCase):
    def test_atomic_write_json_replaces_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            atomic_write_json(path, {"a": 1})
            self.assertEqual({"a": 1}, json.loads(path.read_text(encoding="utf-8")))

    def test_resolve_manifest_hdf5_files_filters_split_camera_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = {
                "splits": {
                    "train": {
                        "shards": [
                            {
                                "shard_id": "shard-00000",
                                "camera_files": {
                                    "observation.images.front": str(
                                        shard_camera_relpath(
                                            "train",
                                            "shard-00000",
                                            "observation.images.front",
                                        )
                                    ),
                                    "observation.images.wrist": str(
                                        shard_camera_relpath(
                                            "train",
                                            "shard-00000",
                                            "observation.images.wrist",
                                        )
                                    ),
                                },
                            },
                            {
                                "shard_id": "shard-00001",
                                "camera_files": {
                                    "observation.images.front": str(
                                        shard_camera_relpath(
                                            "train",
                                            "shard-00001",
                                            "observation.images.front",
                                        )
                                    ),
                                },
                            },
                        ]
                    }
                }
            }
            atomic_write_json(root / "manifest.json", manifest)
            files = resolve_manifest_hdf5_files(
                root,
                split="train",
                camera_key="observation.images.front",
                shard_ids=["shard-00001"],
            )
            self.assertEqual(
                [root / "train" / "shard-00001" / "observation_images_front.h5"],
                files,
            )

    def test_remote_file_size_map_uses_sibling_size(self) -> None:
        info = _FakeRepoInfo([_FakeSibling("foo/bar.h5", 456)])
        self.assertEqual({"foo/bar.h5": 456}, remote_file_size_map(info))


if __name__ == "__main__":
    unittest.main()
