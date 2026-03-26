import pytest

from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_lewm.configuration_lewm import LeWMConfig


def test_validate_features_fails_on_multiple_images_without_matching_preferred_key():
    cfg = LeWMConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
    )

    with pytest.raises(ValueError, match="preferred_image_key"):
        cfg.validate_features()


def test_validate_features_selects_laptop_camera_when_available():
    cfg = LeWMConfig(
        input_features={
            "observation.images.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
    )
    cfg.validate_features()
    assert cfg.selected_image_key == "observation.images.laptop"


def test_validate_features_fails_without_action():
    cfg = LeWMConfig(
        input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
        output_features={},
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires one ACTION output feature"):
        cfg.validate_features()


def test_observation_delta_indices_matches_n_obs_steps():
    cfg = LeWMConfig(
        input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=3,
        device="cpu",
    )
    assert cfg.observation_delta_indices == [-2, -1, 0]


def test_action_delta_indices_align_with_observation_when_world_model_enabled():
    cfg = LeWMConfig(
        input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=3,
        device="cpu",
    )
    assert cfg.action_delta_indices == [-2, -1, 0]


def test_world_model_requires_n_obs_steps_at_least_two():
    with pytest.raises(ValueError, match="n_obs_steps must be >= 2"):
        LeWMConfig(
            input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            n_obs_steps=1,
            device="cpu",
        )


def test_negative_loss_weight_is_rejected():
    with pytest.raises(ValueError, match="Loss weights must be >= 0"):
        LeWMConfig(
            input_features={"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
            loss_weight_dynamics=-1.0,
            device="cpu",
        )
