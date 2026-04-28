from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = importlib.import_module("scripts.convert_lerobot_to_hdf5")
HAS_CV2 = importlib.util.find_spec("cv2") is not None


class _FakeMeta:
    def __init__(self) -> None:
        self.info = {
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        }
        self.episodes = [
            {
                "episode_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": 5,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
            },
            {
                "episode_index": 1,
                "dataset_from_index": 5,
                "dataset_to_index": 12,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
            },
            {
                "episode_index": 2,
                "dataset_from_index": 12,
                "dataset_to_index": 16,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
            },
        ]

    def get_video_file_path(self, ep_idx: int, camera_key: str) -> Path:
        row = self.episodes[int(ep_idx)]
        return Path(
            self.info["video_path"].format(
                video_key=camera_key,
                chunk_index=int(row[f"videos/{camera_key}/chunk_index"]),
                file_index=int(row[f"videos/{camera_key}/file_index"]),
            )
        )


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
                camera_key="observation.images.front",
                video_path=Path("videos/front.mp4"),
            )

    def test_build_video_decode_plan_for_shared_mp4(self) -> None:
        meta = _FakeMeta()
        ep_len = np.array([5, 7, 4], dtype=np.int64)
        ep_offset = np.array([0, 5, 12], dtype=np.int64)
        plan = MODULE._build_video_decode_plan(
            meta,
            episodes=[0, 1, 2],
            camera_keys=["observation.images.front"],
            ep_len=ep_len,
            ep_offset=ep_offset,
        )
        camera_plan = plan["observation.images.front"]
        self.assertEqual(1, len(camera_plan))
        file_plan = camera_plan[0]
        self.assertEqual(Path("videos/observation.images.front/chunk-000/file-000.mp4"), file_plan.video_path)
        self.assertEqual(16, file_plan.expected_total_frames)
        self.assertEqual(
            [(0, 5), (5, 12), (12, 16)],
            [(sl.video_frame_start, sl.video_frame_stop) for sl in file_plan.slices],
        )
        self.assertEqual(
            [(0, 5), (5, 12), (12, 16)],
            [(sl.global_row_start, sl.global_row_stop) for sl in file_plan.slices],
        )

    def test_build_video_decode_plan_keeps_global_offsets_for_subset(self) -> None:
        meta = _FakeMeta()
        ep_len = np.array([7, 4], dtype=np.int64)
        ep_offset = np.array([0, 7], dtype=np.int64)
        plan = MODULE._build_video_decode_plan(
            meta,
            episodes=[1, 2],
            camera_keys=["observation.images.front"],
            ep_len=ep_len,
            ep_offset=ep_offset,
        )
        file_plan = plan["observation.images.front"][0]
        self.assertEqual(16, file_plan.expected_total_frames)
        self.assertEqual(
            [(5, 12), (12, 16)],
            [(sl.video_frame_start, sl.video_frame_stop) for sl in file_plan.slices],
        )
        self.assertEqual(
            [(0, 7), (7, 11)],
            [(sl.global_row_start, sl.global_row_stop) for sl in file_plan.slices],
        )

    def test_decode_and_write_video_plan_routes_frames_to_each_episode(self) -> None:
        pixels = np.zeros((16, 2, 2, 3), dtype=np.uint8)
        plan = MODULE.VideoDecodePlan(
            camera_key="observation.images.front",
            video_path=Path("videos/front.mp4"),
            slices=[
                MODULE.VideoEpisodeSlice(0, 5, 0, 5, 0, 5),
                MODULE.VideoEpisodeSlice(1, 7, 5, 12, 5, 12),
                MODULE.VideoEpisodeSlice(2, 4, 12, 16, 12, 16),
            ],
            expected_total_frames=16,
        )

        class _FrameIter:
            def __init__(self) -> None:
                self.idx = 1

            def __iter__(self) -> "_FrameIter":
                return self

            def __next__(self) -> np.ndarray:
                if self.idx >= 16:
                    raise StopIteration
                value = self.idx
                self.idx += 1
                return np.full((2, 2, 3), value, dtype=np.uint8)

            def close(self) -> None:
                return None

        with patch.object(
            MODULE,
            "_open_linear_frame_iterator",
            return_value=("stub", _FrameIter(), np.full((2, 2, 3), 0, dtype=np.uint8)),
        ):
            backend, decoded, flushes = MODULE._decode_and_write_video_plan(
                split_name="train",
                plan=plan,
                pixels_ds=pixels,
                total_rows=16,
                image_size=None,
                micro_batch_size=4,
                decode_backend="pyav",
                stall_timeout_seconds=5.0,
                heartbeat_seconds=60.0,
            )
        self.assertEqual("stub", backend)
        self.assertEqual(16, decoded)
        self.assertEqual(4, flushes)
        self.assertTrue(np.all(pixels[0] == 0))
        self.assertTrue(np.all(pixels[4] == 4))
        self.assertTrue(np.all(pixels[5] == 5))
        self.assertTrue(np.all(pixels[11] == 11))
        self.assertTrue(np.all(pixels[12] == 12))
        self.assertTrue(np.all(pixels[15] == 15))

    def test_decode_and_write_video_plan_fails_when_video_is_too_short(self) -> None:
        pixels = np.zeros((6, 2, 2, 3), dtype=np.uint8)
        plan = MODULE.VideoDecodePlan(
            camera_key="observation.images.front",
            video_path=Path("videos/front.mp4"),
            slices=[MODULE.VideoEpisodeSlice(0, 6, 0, 6, 0, 6)],
            expected_total_frames=6,
        )

        class _ShortIter:
            def __iter__(self) -> "_ShortIter":
                return self

            def __next__(self) -> np.ndarray:
                raise StopIteration

            def close(self) -> None:
                return None

        with patch.object(
            MODULE,
            "_open_linear_frame_iterator",
            return_value=("stub", _ShortIter(), np.full((2, 2, 3), 0, dtype=np.uint8)),
        ):
            with self.assertRaises(RuntimeError):
                MODULE._decode_and_write_video_plan(
                    split_name="train",
                    plan=plan,
                    pixels_ds=pixels,
                    total_rows=6,
                    image_size=None,
                    micro_batch_size=4,
                    decode_backend="pyav",
                    stall_timeout_seconds=5.0,
                    heartbeat_seconds=60.0,
                )

    def test_decode_and_write_video_plan_fails_when_slice_exceeds_output(self) -> None:
        pixels = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        plan = MODULE.VideoDecodePlan(
            camera_key="observation.images.front",
            video_path=Path("videos/front.mp4"),
            slices=[MODULE.VideoEpisodeSlice(0, 4, 1, 5, 0, 4)],
            expected_total_frames=4,
        )

        class _FrameIter:
            def __iter__(self) -> "_FrameIter":
                return self

            def __next__(self) -> np.ndarray:
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def close(self) -> None:
                return None

        with patch.object(
            MODULE,
            "_open_linear_frame_iterator",
            return_value=("stub", _FrameIter(), np.zeros((2, 2, 3), dtype=np.uint8)),
        ):
            with self.assertRaises(RuntimeError):
                MODULE._decode_and_write_video_plan(
                    split_name="train",
                    plan=plan,
                    pixels_ds=pixels,
                    total_rows=4,
                    image_size=None,
                    micro_batch_size=4,
                    decode_backend="pyav",
                    stall_timeout_seconds=5.0,
                    heartbeat_seconds=60.0,
                )

    def test_dependency_check_allows_single_available_backend_without_resize(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name == "av":
                return object()
            if name == "cv2":
                return None
            return object()

        with (
            patch.object(MODULE.importlib.util, "find_spec", side_effect=fake_find_spec),
            patch.object(MODULE, "_require_h5py", return_value=None),
        ):
            MODULE._assert_runtime_dependencies(decode_backend="pyav", image_size=None)

    def test_dependency_check_requires_cv2_for_resize(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name == "av":
                return object()
            if name == "cv2":
                return None
            return object()

        with (
            patch.object(MODULE.importlib.util, "find_spec", side_effect=fake_find_spec),
            patch.object(MODULE, "_require_h5py", return_value=None),
        ):
            with self.assertRaises(RuntimeError):
                MODULE._assert_runtime_dependencies(decode_backend="pyav", image_size=224)


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
