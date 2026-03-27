from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = importlib.import_module("scripts.visualize_hdf5_rerun")
HAS_H5PY = importlib.util.find_spec("h5py") is not None


@unittest.skipUnless(HAS_H5PY, "h5py is not installed")
class VisualizePathResolutionTests(unittest.TestCase):
    def test_resolve_single_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train__observation_images_front.h5").touch()
            out = MODULE._resolve_hdf5_file(root, split="train", camera_key="observation.images.front")
            self.assertEqual(root / "train__observation_images_front.h5", out)

    def test_resolve_ambiguous_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train__observation_images_front.h5").touch()
            (root / "train__observation_images_wrist.h5").touch()
            with self.assertRaises(ValueError):
                MODULE._resolve_hdf5_file(root, split="train", camera_key=None)


@unittest.skipUnless(HAS_H5PY, "h5py is not installed")
class VisualizeValidationTests(unittest.TestCase):
    def _create_valid_h5(self, path: Path) -> None:
        import h5py

        with h5py.File(path, "w") as h5f:
            h5f.create_dataset("pixels", data=np.zeros((6, 8, 8, 3), dtype=np.uint8))
            h5f.create_dataset("action", data=np.zeros((6, 4), dtype=np.float32))
            h5f.create_dataset("state", data=np.zeros((6, 5), dtype=np.float32))
            h5f.create_dataset("done", data=np.array([0, 0, 1, 0, 0, 1], dtype=bool))
            h5f.create_dataset("episode_idx", data=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64))
            h5f.create_dataset("step_idx", data=np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
            h5f.create_dataset("ep_len", data=np.array([3, 3], dtype=np.int64))
            h5f.create_dataset("ep_offset", data=np.array([0, 3], dtype=np.int64))
            h5f.attrs["source_repo_id"] = "lerobot/koch_pick_place_1_lego"
            h5f.attrs["source_split"] = "train"
            h5f.attrs["source_camera_key"] = "observation.images.front"

    def test_validate_missing_keys_raises(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.h5"
            with h5py.File(path, "w") as h5f:
                h5f.create_dataset("pixels", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
            with h5py.File(path, "r") as h5f:
                with self.assertRaises(KeyError):
                    MODULE._validate_hdf5_schema(h5f)

    def test_window_validation_raises(self) -> None:
        with self.assertRaises(ValueError):
            MODULE._resolve_step_window(
                episode_length=10,
                start_step=8,
                end_step=2,
                step_stride=1,
            )

    def test_default_rrd_path(self) -> None:
        base = Path("/tmp/train__observation_images_front.h5")
        out = MODULE._default_rrd_path(base, episode_index=7)
        self.assertEqual(Path("/tmp/train__observation_images_front__ep7.rrd"), out)

    def test_log_loop_smoke_with_fake_rerun(self) -> None:
        import h5py

        class _FakeRR:
            def __init__(self) -> None:
                self.logs: list[tuple[str, object]] = []
                self.times: list[int] = []

            class Image:
                def __init__(self, value: object) -> None:
                    self.value = value

            class Scalar:
                def __init__(self, value: float) -> None:
                    self.value = value

            def set_time_sequence(self, name: str, value: int) -> None:
                self.times.append(int(value))

            def log(self, path: str, value: object) -> None:
                self.logs.append((path, value))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train__observation_images_front.h5"
            self._create_valid_h5(path)
            fake_rr = _FakeRR()
            with h5py.File(path, "r") as h5f:
                MODULE._validate_hdf5_schema(h5f)
                ep_start, _, ep_len = MODULE._episode_bounds(h5f, episode_index=0)
                step_window = MODULE._resolve_step_window(
                    episode_length=ep_len,
                    start_step=0,
                    end_step=2,
                    step_stride=1,
                )
                logged = MODULE._log_episode_steps(
                    rr=fake_rr,
                    h5f=h5f,
                    episode_row_start=ep_start,
                    step_window=step_window,
                    playback_hz=0.0,
                    progress_every=1,
                )
            self.assertEqual(3, logged)
            self.assertEqual([0, 1, 2], fake_rr.times)
            self.assertTrue(any(p == "episode/pixels" for p, _ in fake_rr.logs))


if __name__ == "__main__":
    unittest.main()
