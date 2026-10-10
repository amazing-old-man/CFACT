import json
from importlib.resources import files

import pytest

from cfact.collection.config import validate_config
from cfact.collection.coordinator import _next_dispatch_index


def test_task_cap_dispatches_other_task_and_preserves_each_task_order():
    pending = [{"task": "a", "root_id": "a1"}, {"task": "b", "root_id": "b0"},
               {"task": "b", "root_id": "b1"}]
    active = [{"task": "a", "root_id": "a0"}]
    assert _next_dispatch_index(pending, active, 1) == 1
    active.append(pending.pop(1))
    assert _next_dispatch_index(pending, active, 1) is None
    assert _next_dispatch_index(pending, active[1:], 1) == 0
    assert _next_dispatch_index(pending, active[:1], 1) == 1
    assert _next_dispatch_index(pending, active, None) == 0
    assert _next_dispatch_index(pending, active, 2) == 0
    assert _next_dispatch_index([], active, 1) is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_invalid_task_cap_is_rejected(limit):
    config = json.loads(files("cfact.resources").joinpath("collection.json").read_text())
    config["scheduling"] = {"max_active_roots_per_task": limit}
    with pytest.raises(ValueError, match="max_active_roots_per_task"):
        validate_config(config)
