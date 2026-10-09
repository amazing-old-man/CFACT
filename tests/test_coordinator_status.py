import pytest
from cfact.collection.coordinator import _require_stage_success


@pytest.mark.parametrize("status", ["TECHNICAL_ERROR", "COLLECTION_BUDGET_CENSORED"])
def test_failed_worker_is_not_reported_as_success(status):
    with pytest.raises(RuntimeError):
        _require_stage_success(
            [{"status": status}], {}, {}, {"status": "DEFERRED_UNTIL_FULL_QUEUE"}
        )


@pytest.mark.parametrize("which", ["encoding", "lineages", "export"])
def test_failed_output_stage_is_not_reported_as_success(which):
    stages = {"encoding": {}, "lineages": {}, "export": {"status": "PASS"}}
    stages[which] = {"status": "EXPORT_ERROR"} if which == "export" else {"failures": ["failed"]}
    with pytest.raises(RuntimeError):
        _require_stage_success([{"status": "PHYSICS_COMPLETE"}], **stages)


def test_declared_source_replay_block_is_valid_recorded_outcome():
    _require_stage_success(
        [{"status": "BLOCKED_SOURCE_REPLAY"}, {"status": "PHYSICS_COMPLETE"}],
        {},
        {},
        {"status": "PASS"},
    )
