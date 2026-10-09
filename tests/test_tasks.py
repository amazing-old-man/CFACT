from cfact.generation.primitives import default_operation_plugins
from cfact.tasks import SUPPORTED_TASK_NAMES, TARGET_TASKS, target_task


def test_exact_target_registry_has_twenty_tasks_and_nine_macro_families():
    assert len(TARGET_TASKS) == 20
    assert len(SUPPORTED_TASK_NAMES) == 20
    assert len(set(SUPPORTED_TASK_NAMES)) == 20
    assert len({task.operation_family for task in TARGET_TASKS}) == 9
    assert "NavigateKitchen" not in SUPPORTED_TASK_NAMES


def test_every_target_maps_to_an_existing_primitive_plugin():
    plugins = default_operation_plugins()
    assert all(task.primitive_family in plugins for task in TARGET_TASKS)


def test_unknown_task_is_not_silently_accepted():
    try:
        target_task("NavigateKitchen")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown task unexpectedly entered the whitelist")


def test_close_blender_uses_mobile_base_frame_adapter():
    assert target_task("CloseBlenderLid").adapter_interface == "MG_ModernCloseBlenderLid"
