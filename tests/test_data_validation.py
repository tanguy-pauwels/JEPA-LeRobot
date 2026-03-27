from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lewm_dataset_utils.data_validation import (
    collect_source_episode_issues,
    validate_hdf5_file,
)

HAS_H5PY = importlib.util.find_spec("h5py") is not None


class SourceValidationTests(unittest.TestCase):
    def test_collect_source_episode_issues_clean(self) -> None:
        episode_idx = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        step_idx = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        done = np.array([False, False, True, False, True], dtype=bool)

        issues = collect_source_episode_issues(episode_idx, step_idx, done)
        self.assertEqual(issues, {})

    def test_collect_source_episode_issues_missing_terminal_done(self) -> None:
        episode_idx = np.array([0, 0, 1, 1], dtype=np.int64)
        step_idx = np.array([0, 1, 0, 1], dtype=np.int64)
        done = np.array([False, False, False, False], dtype=bool)

        issues = collect_source_episode_issues(episode_idx, step_idx, done)
        self.assertIn(0, issues)
        self.assertIn(1, issues)
        self.assertTrue(
            any("not terminated cleanly" in msg for msg in issues[0]),
            issues[0],
        )

    def test_collect_source_episode_issues_early_done(self) -> None:
        episode_idx = np.array([0, 0, 0], dtype=np.int64)
        step_idx = np.array([0, 1, 2], dtype=np.int64)
        done = np.array([False, True, True], dtype=bool)

        issues = collect_source_episode_issues(episode_idx, step_idx, done)
        self.assertIn(0, issues)
        self.assertTrue(any("before the episode terminal step" in m for m in issues[0]))

    def test_collect_source_episode_issues_non_contiguous_step_idx(self) -> None:
        episode_idx = np.array([0, 0, 0], dtype=np.int64)
        step_idx = np.array([0, 2, 3], dtype=np.int64)
        done = np.array([False, False, True], dtype=bool)

        issues = collect_source_episode_issues(episode_idx, step_idx, done)
        self.assertIn(0, issues)
        self.assertTrue(any("not contiguous" in m for m in issues[0]))


@unittest.skipUnless(HAS_H5PY, "h5py is not installed")
class HDF5ValidationTests(unittest.TestCase):
    def _create_base_h5(self, file_path: Path, done_values: np.ndarray) -> None:
        import h5py

        n_rows = len(done_values)
        with h5py.File(file_path, "w") as h5f:
            h5f.create_dataset("pixels", data=np.zeros((n_rows, 4, 4, 3), dtype=np.uint8))
            h5f.create_dataset("action", data=np.zeros((n_rows, 2), dtype=np.float32))
            h5f.create_dataset("proprio", data=np.zeros((n_rows, 2), dtype=np.float32))
            h5f.create_dataset("state", data=np.zeros((n_rows, 2), dtype=np.float32))
            h5f.create_dataset("episode_idx", data=np.array([0, 0, 1, 1], dtype=np.int64))
            h5f.create_dataset("step_idx", data=np.array([0, 1, 0, 1], dtype=np.int64))
            h5f.create_dataset("done", data=done_values.astype(bool))
            h5f.create_dataset("timestamp", data=np.arange(n_rows, dtype=np.float32))
            h5f.create_dataset("index", data=np.arange(n_rows, dtype=np.int64))
            h5f.create_dataset("task_index", data=np.zeros((n_rows,), dtype=np.int64))
            h5f.create_dataset("ep_len", data=np.array([2, 2], dtype=np.int64))
            h5f.create_dataset("ep_offset", data=np.array([0, 2], dtype=np.int64))

    def test_validate_hdf5_file_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "clean.h5"
            self._create_base_h5(h5_path, done_values=np.array([False, True, False, True]))
            errors = validate_hdf5_file(h5_path, strict_done=True)
            self.assertEqual([], errors)

    def test_validate_hdf5_file_terminal_done_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "bad_terminal.h5"
            self._create_base_h5(
                h5_path, done_values=np.array([False, False, False, True])
            )
            errors = validate_hdf5_file(h5_path, strict_done=True)
            self.assertTrue(any("terminal step has done=False" in err for err in errors), errors)

    def test_validate_hdf5_file_early_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "bad_early.h5"
            self._create_base_h5(h5_path, done_values=np.array([True, True, False, True]))
            errors = validate_hdf5_file(h5_path, strict_done=True)
            self.assertTrue(any("before terminal step" in err for err in errors), errors)


if __name__ == "__main__":
    unittest.main()

