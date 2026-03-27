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

Recommended on laptops (safer memory profile):

```bash
python scripts/convert_lerobot_to_hdf5.py \
  --num_workers=2 \
  --max_pending_tasks=1 \
  --memory_guard_mode=error
```

Notes:

- `num_workers=0` uses conservative auto mode capped by `--auto_max_workers` (default: `2`).
- `memory_guard_mode` can be `off`, `warn`, or `error`.
- `max_inflight_memory_ratio` controls the guard threshold (default: `0.40`).

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
