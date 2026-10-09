import json
from importlib.resources import files
import pytest
from cfact.collection.config import validate_config


def settings():
    return json.loads(files("cfact.resources").joinpath("collection.json").read_text())


def test_packaged_protocol_and_zero_fixed_horizon_are_supported():
    config = settings()
    validate_config(config)
    config["execution"]["continuation_steps"] = 0
    validate_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("candidate_pool", "max_selected_g0", 0),
        ("tree_limits", "maximum_decision_depth", 4),
        ("execution", "max_env_steps_per_root", -1),
        ("video", "fps", 10),
        ("video", "fps", 20.5),
        ("resource", "maximum_output_gb", float("nan")),
    ],
)
def test_invalid_or_unimplemented_settings_fail_before_collection(section, key, value):
    config = settings()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
