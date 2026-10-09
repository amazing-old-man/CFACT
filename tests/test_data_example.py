import json
import numpy as np
import pytest
from cfact import TreeDataset
from cfact.data.demo import make_demo
from cfact.data.example import summarize


def test_example_extracts_only_sibling_contrasts_and_preserves_censoring(tmp_path):
    run = make_demo(tmp_path / "run")
    report = summarize(run, output=tmp_path / "pairs.jsonl")
    assert report["physical_edges"] == 3
    assert report["outcome_pairs"] == 1
    assert report["synthetic_fixture"] is True
    pair = json.loads((tmp_path / "pairs.jsonl").read_text())
    assert {pair["left_outcome"], pair["right_outcome"]} == {"official_success", "horizon_censored"}
    assert pair["parent_edge"] is None
    assert "preference" not in pair
    assert report["first_transition"]["time_t_plus_1"] == 0.05


def test_tree_path_uses_actual_child_and_rejects_sibling_stitching(tmp_path):
    reader = TreeDataset(make_demo(tmp_path / "run"))
    assert len(reader.read_tree_path("ExampleTask", "root1", ["a", "c"])) == 2
    with pytest.raises(KeyError):
        reader.read_tree_path("ExampleTask", "root1", ["a", "b"])
    with pytest.raises(ValueError):
        reader.read_node("ExampleTask", "root1", "../../outside")


def test_ambiguous_candidate_id_requires_full_edge_path(tmp_path):
    run = make_demo(tmp_path / "run")
    path = next(run.glob("data/*/*/*/tree.json"))
    tree = json.loads(path.read_text())
    tree["child_pools"][0]["edges"][0]["candidate_id"] = "a"
    path.write_text(json.dumps(tree))
    reader = TreeDataset(run)
    with pytest.raises(KeyError, match="expected one edge"):
        reader.read_edge("ExampleTask", "root1", "a")
    assert reader.read_edge("ExampleTask", "root1", tree["edges"][0]["edge_path"])


def test_unknown_results_do_not_become_negative_examples(tmp_path):
    run = make_demo(tmp_path / "run")
    path = next(run.glob("data/*/*/*/tree.json"))
    tree = json.loads(path.read_text())
    tree["edges"][0]["Y_fixed"] = "technical_unknown"
    path.write_text(json.dumps(tree))
    assert summarize(run)["outcome_pairs"] == 0


def test_inconsistent_measured_state_lengths_are_rejected(tmp_path):
    reader = TreeDataset(make_demo(tmp_path / "run"))
    edge = reader.read_edge("ExampleTask", "root1", "a")
    path = reader.run_root / edge["edge_path"] / "execution.npz"
    values = reader.read_segment("ExampleTask", "root1", "a")
    values["robot_qpos"] = np.zeros((2, 7))
    np.savez(path, **values)
    with pytest.raises(ValueError, match="state alignment"):
        reader.read_segment("ExampleTask", "root1", "a")


def test_example_does_not_modify_or_overwrite_input(tmp_path):
    run = make_demo(tmp_path / "run")
    with pytest.raises(ValueError, match="outside"):
        summarize(run, output=run / "pairs.jsonl")
    output = tmp_path / "pairs.jsonl"
    output.write_text("previous result")
    with pytest.raises(FileExistsError):
        summarize(run, output=output)
    assert output.read_text() == "previous result"
