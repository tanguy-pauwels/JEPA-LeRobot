import numpy as np

from lerobot_policy_lewm.eval_offline import (
    _resolve_policy_path,
    compute_per_dim_metrics,
    compute_per_episode_metrics,
)


def test_compute_per_dim_metrics_shapes_and_keys():
    target = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    pred = np.array([[0.1, 0.8], [0.9, 3.2], [2.2, 4.9]], dtype=np.float32)

    rows = compute_per_dim_metrics(target, pred)

    assert len(rows) == 2
    assert rows[0]["dim"] == 0
    assert "mse" in rows[0]
    assert "mae" in rows[0]
    assert "corr" in rows[0]


def test_compute_per_episode_metrics_aggregation():
    target = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]], dtype=np.float32)
    pred = np.array([[0.1, 0.8], [0.9, 3.2], [2.2, 4.9], [2.9, 7.1]], dtype=np.float32)
    episodes = np.array([10, 10, 11, 11], dtype=np.int64)

    rows = compute_per_episode_metrics(episodes, target, pred)

    assert len(rows) == 2
    assert rows[0]["episode_index"] == 10
    assert rows[0]["num_samples"] == 2
    assert rows[1]["episode_index"] == 11


def test_resolve_policy_path_from_run_dir(tmp_path):
    run_dir = tmp_path / "outputs" / "train" / "2026-03-26" / "12-19-51_lewm"
    ckpt = run_dir / "checkpoints" / "000100" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}", encoding="utf-8")

    resolved = _resolve_policy_path(str(run_dir))
    assert resolved == ckpt


def test_resolve_policy_path_from_missing_last_alias(tmp_path):
    run_dir = tmp_path / "outputs" / "train" / "2026-03-26" / "12-19-51_lewm"
    ckpt1 = run_dir / "checkpoints" / "000100" / "pretrained_model"
    ckpt2 = run_dir / "checkpoints" / "000200" / "pretrained_model"
    ckpt1.mkdir(parents=True)
    ckpt2.mkdir(parents=True)
    (ckpt1 / "config.json").write_text("{}", encoding="utf-8")
    (ckpt2 / "config.json").write_text("{}", encoding="utf-8")

    hinted = run_dir / "checkpoints" / "last" / "pretrained_model"
    resolved = _resolve_policy_path(str(hinted))
    assert resolved == ckpt2


def test_resolve_policy_path_from_checkpoints_last_dir(tmp_path):
    run_dir = tmp_path / "outputs" / "train" / "2026-03-26" / "12-19-51_lewm"
    ckpt1 = run_dir / "checkpoints" / "000100" / "pretrained_model"
    ckpt2 = run_dir / "checkpoints" / "000200" / "pretrained_model"
    ckpt1.mkdir(parents=True)
    ckpt2.mkdir(parents=True)
    (ckpt1 / "config.json").write_text("{}", encoding="utf-8")
    (ckpt2 / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoints" / "last").mkdir(parents=True, exist_ok=True)

    resolved = _resolve_policy_path(str(run_dir / "checkpoints" / "last"))
    assert resolved == ckpt2
