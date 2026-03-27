# Dataset Utility Scripts

Utilities for the LeRobot -> LeWM data pipeline.

## 1) Download LeRobot datasets

```bash
python scripts/download_lerobot_datasets.py
```

Default dataset:

- `lerobot/koch_pick_place_1_lego`

Example with custom repo:

```bash
python scripts/download_lerobot_datasets.py \
  --repo_ids='["lerobot/koch_pick_place_1_lego","lerobot/koch_pick_place_5_lego"]'
```

## 2) Convert LeRobot to HDF5 (LeWM)

```bash
python scripts/convert_lerobot_to_hdf5.py
```

Output:

- `datasets/hdf5/lerobot__koch_pick_place_1_lego/<split>__<camera>.h5`

Strict clean-termination mode (default):

- `require_terminal_done=true`
- `dirty_episode_policy=fail`

Example with resize and drop policy:

```bash
python scripts/convert_lerobot_to_hdf5.py \
  --image_size=224 \
  --dirty_episode_policy=drop
```

Recommended for Mac 8GB / Kaggle stability:

```bash
python scripts/convert_lerobot_to_hdf5.py \
  --decode_backend=pyav \
  --micro_batch_size=64 \
  --stall_timeout_seconds=120
```

Notes:

- `decode_backend` supports `pyav` (default) and `opencv`; the converter automatically falls back to the other backend if initialization fails.
- `micro_batch_size` controls RAM and I/O granularity; `64` is the default and HDF5 chunks are aligned to this value.
- `stall_timeout_seconds` fails fast if no decode/write progress is observed for too long (default: `120`).
- `num_workers` and previous parallelism/torch flags are kept for CLI compatibility but are deprecated and ignored in stable linear mode.
- `heartbeat_seconds` controls periodic liveness logs (default: `10`).

A machine-readable report is written to:

- `datasets/hdf5/<repo_sanitized>/conversion_report.json`

## 3) Inspect / Validate HDF5

Inspect:

```bash
python scripts/inspect_validate_hdf5.py \
  --mode=inspect \
  --target_path=datasets/hdf5/lerobot__koch_pick_place_1_lego
```

Validate:

```bash
python scripts/inspect_validate_hdf5.py \
  --mode=validate \
  --target_path=datasets/hdf5/lerobot__koch_pick_place_1_lego
```

Validation includes JEPA-critical checks:

- terminal `done=True` on each episode
- no early `done=True`
- contiguous `step_idx` from `0` to `ep_len-1`
- consistency between `ep_len`, `ep_offset`, and per-step arrays
