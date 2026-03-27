#!/usr/bin/env python3
"""Inspect and validate LeWM HDF5 datasets."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import draccus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_dataset_utils import (  # noqa: E402
    inspect_hdf5_file,
    list_hdf5_files,
    sanitize_repo_id,
    validate_hdf5_file,
)

DEFAULT_DATASET = "lerobot/koch_pick_place_1_lego"


@dataclass
class InspectValidateConfig:
    mode: str = "validate"  # inspect | validate
    target_path: str = (
        f"datasets/hdf5/{sanitize_repo_id(DEFAULT_DATASET)}"
    )
    output_json: str | None = None
    fail_on_error: bool = True
    strict_done: bool = True


@draccus.wrap()
def main(cfg: InspectValidateConfig) -> None:
    if cfg.mode not in ("inspect", "validate"):
        raise ValueError("mode must be 'inspect' or 'validate'.")

    target = Path(cfg.target_path)
    files = list_hdf5_files(target)
    if not files:
        raise FileNotFoundError(f"No .h5 file found under: {target}")

    payload: dict[str, Any] = {
        "mode": cfg.mode,
        "target_path": str(target),
        "files": {},
    }

    any_error = False
    for file_path in files:
        if cfg.mode == "inspect":
            summary = inspect_hdf5_file(file_path)
            payload["files"][str(file_path)] = summary
            print(
                f"[inspect] {file_path} "
                f"(episodes={summary.get('num_episodes')}, steps={summary.get('total_steps')})"
            )
        else:
            errors = validate_hdf5_file(file_path, strict_done=cfg.strict_done)
            valid = len(errors) == 0
            payload["files"][str(file_path)] = {"valid": valid, "errors": errors}
            if valid:
                print(f"[validate] OK {file_path}")
            else:
                any_error = True
                print(f"[validate] FAIL {file_path}")
                for err in errors:
                    print(f"  - {err}")

    if cfg.output_json:
        out_path = Path(cfg.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{cfg.mode}] json report: {out_path}")

    if cfg.mode == "validate" and any_error and cfg.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
