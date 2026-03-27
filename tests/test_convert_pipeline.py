from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = importlib.import_module("scripts.convert_lerobot_to_hdf5")
HAS_CV2 = importlib.util.find_spec("cv2") is not None


class ConvertPipelineUnitTests(unittest.TestCase):
    def test_backend_order(self) -> None:
        self.assertEqual(["pyav", "opencv"], MODULE._backend_order("pyav"))
        self.assertEqual(["opencv", "pyav"], MODULE._backend_order("opencv"))

    def test_chunk_rows_aligns_micro_batch(self) -> None:
        self.assertEqual(64, MODULE._compute_chunk_rows(total_rows=300, micro_batch_size=64))
        self.assertEqual(10, MODULE._compute_chunk_rows(total_rows=10, micro_batch_size=64))

    def test_stall_watchdog_raises(self) -> None:
        with self.assertRaises(TimeoutError):
            MODULE._check_stall_or_raise(
                now=20.0,
                last_progress=0.0,
                stall_timeout_seconds=5.0,
                context="unit-test",
            )

    def test_frame_count_fail_fast(self) -> None:
        with self.assertRaises(RuntimeError):
            MODULE._assert_frame_count_or_raise(
                decoded_frames=5,
                expected_frames=6,
                split_name="train",
                episode_index=1,
                camera_key="observation.images.front",
            )


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class ConvertPipelineOpenCVTests(unittest.TestCase):
    def test_resize_batch_cv2_keeps_shape(self) -> None:
        frames = [
            np.zeros((16, 20, 3), dtype=np.uint8),
            np.ones((16, 20, 3), dtype=np.uint8) * 200,
        ]
        out = MODULE._resize_batch_cv2(frames, image_size=8)
        self.assertEqual((2, 8, 8, 3), out.shape)
        self.assertEqual(np.uint8, out.dtype)

    def test_open_linear_frame_iterator_fallback_to_opencv(self) -> None:
        cv2 = importlib.import_module("cv2")
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "tiny.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (16, 16),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter not available in this environment")

            for i in range(4):
                frame = np.zeros((16, 16, 3), dtype=np.uint8)
                frame[:, :, 2] = i * 40  # BGR red channel for deterministic gradient
                writer.write(frame)
            writer.release()

            backend, iterator, first = MODULE._open_linear_frame_iterator(
                video_path=video_path,
                decode_backend="opencv",
            )
            self.assertEqual("opencv", backend)
            self.assertEqual((16, 16, 3), first.shape)
            self.assertEqual(np.uint8, first.dtype)
            close_fn = getattr(iterator, "close", None)
            if callable(close_fn):
                close_fn()


if __name__ == "__main__":
    unittest.main()
