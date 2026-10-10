import argparse
import yaml
import pytest
from cfact.run import generate


@pytest.mark.parametrize("explicit", [False, True])
def test_generate_validates_the_configured_source_directory_before_any_physics(
    tmp_path, monkeypatch, explicit
):
    config = yaml.safe_load(
        (__import__("pathlib").Path("configs/pre_scale_native224.yaml")).read_text()
    )
    config["source_dir"] = str(tmp_path / "official-source")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    override = tmp_path / "explicit-prepared" if explicit else None
    seen = []

    class ValidationBoundary(Exception):
        pass

    def validate(path, **kwargs):
        seen.append((path, kwargs["required_episodes"]))
        raise ValidationBoundary

    monkeypatch.setattr("cfact.run.validate_prepared_identity", validate)
    args = argparse.Namespace(
        config=config_path,
        prepared_dir=override,
        task=None,
        root_demo=None,
        donor_demo=None,
        max_roots_per_task=None,
        seed=None,
        max_control_steps=None,
        workers=1,
        gpu_devices="0",
        export_python=None,
        dry_run=False,
    )
    with pytest.raises(ValidationBoundary):
        generate(args)
    expected = (override or tmp_path / "official-source/prepared") / "OpenDrawer.hdf5"
    assert seen == [(expected.resolve(), ["demo_0", "demo_3", "demo_4"])]
