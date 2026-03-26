import torch

from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_lewm.configuration_lewm import LeWMConfig
from lerobot_policy_lewm.modeling_lewm import LeWMPolicy


def _make_policy() -> LeWMPolicy:
    cfg = LeWMConfig(
        input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
    )
    return LeWMPolicy(cfg)


def test_forward_returns_scalar_loss_and_log_dict():
    policy = _make_policy()
    batch = {
        "observation.image": torch.randint(0, 255, (2, 3, 3, 224, 224), dtype=torch.uint8),
        "action": torch.randn(2, 3, 7),
    }

    loss, logs = policy.forward(batch)

    assert loss.ndim == 0
    assert "mse_loss" in logs
    assert "mae_loss" in logs
    assert "loss_total" in logs
    assert "loss_action" in logs
    assert "loss_dynamics" in logs
    assert "loss_sigreg" in logs


def test_forward_supports_reduction_none_and_debug_metrics():
    cfg = LeWMConfig(
        input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        debug_metrics=True,
    )
    policy = LeWMPolicy(cfg)
    batch = {
        "observation.image": torch.randint(0, 255, (2, 3, 3, 224, 224), dtype=torch.uint8),
        "action": torch.randn(2, 3, 7),
    }

    loss, logs = policy.forward(batch, reduction="none")

    assert tuple(loss.shape) == (2,)
    for key in ["pred_mean", "pred_std", "target_mean", "target_std", "pred_target_l2"]:
        assert key in logs


def test_forward_raises_on_action_history_mismatch():
    policy = _make_policy()
    batch = {
        "observation.image": torch.randint(0, 255, (2, 3, 3, 224, 224), dtype=torch.uint8),
        "action": torch.randn(2, 7),
    }
    try:
        policy.forward(batch)
        assert False, "Expected ValueError for action history mismatch"
    except ValueError as e:
        assert "aligned with image history" in str(e)


def test_select_action_returns_expected_shape():
    policy = _make_policy()
    batch = {
        "observation.image": torch.randint(0, 255, (2, 3, 3, 224, 224), dtype=torch.uint8),
    }

    action = policy.select_action(batch)

    assert tuple(action.shape) == (2, 7)


def test_select_action_multistep_with_single_frames():
    policy = _make_policy()
    for _ in range(5):
        batch = {
            "observation.image": torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8),
        }
        action = policy.select_action(batch)
        assert tuple(action.shape) == (2, 7)
