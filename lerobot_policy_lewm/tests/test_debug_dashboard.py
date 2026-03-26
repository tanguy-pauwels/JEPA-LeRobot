import json
from pathlib import Path

import pandas as pd

from lerobot_policy_lewm.debug_dashboard import build_app


def test_build_app_from_artifacts(tmp_path: Path):
    summary = {
        "policy_path": "/tmp/policy",
        "dataset_repo_id": "lerobot/aloha_sim_insertion_human_image",
        "device": "cpu",
        "num_samples": 4,
        "action_dim": 2,
        "global_mse": 0.1,
        "global_mae": 0.2,
        "avg_l2_per_sample": 0.3,
        "target_mean": 0.0,
        "target_std": 1.0,
        "pred_mean": 0.1,
        "pred_std": 0.9,
    }
    (tmp_path / "offline_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    pd.DataFrame([
        {"dim": 0, "mse": 0.1, "mae": 0.2, "corr": 0.9, "target_mean": 0, "target_std": 1, "pred_mean": 0.1, "pred_std": 0.8},
        {"dim": 1, "mse": 0.2, "mae": 0.3, "corr": 0.8, "target_mean": 0, "target_std": 1, "pred_mean": 0.2, "pred_std": 0.7},
    ]).to_csv(tmp_path / "per_dim_metrics.csv", index=False)

    pd.DataFrame([
        {"episode_index": 0, "num_samples": 2, "mse": 0.1, "mae": 0.2, "max_abs_error": 0.4},
        {"episode_index": 1, "num_samples": 2, "mse": 0.2, "mae": 0.3, "max_abs_error": 0.5},
    ]).to_csv(tmp_path / "per_episode_metrics.csv", index=False)

    pd.DataFrame([
        {"batch_index": 0, "sample_index_in_batch": 0, "episode_index": 0, "dim": 0, "target": 0.0, "pred": 0.1, "error": 0.1, "abs_error": 0.1},
        {"batch_index": 0, "sample_index_in_batch": 1, "episode_index": 0, "dim": 1, "target": 1.0, "pred": 0.9, "error": -0.1, "abs_error": 0.1},
    ]).to_csv(tmp_path / "samples_predictions.csv", index=False)

    app = build_app(tmp_path)
    assert app is not None
