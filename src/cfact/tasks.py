"""Registry of supported RoboCasa task interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetTask:
    name: str
    operation_family: str
    primitive_family: str
    legacy_env: str | None
    mimicgen_config: str
    mimicgen_interface: str
    adapter_interface: str | None = None


TARGET_TASKS: tuple[TargetTask, ...] = (
    TargetTask(
        "PickPlaceCounterToCabinet",
        "pick_place",
        "placement",
        "PnPCounterToCab",
        "PnPCounterToCabinet",
        "MG_PnPCounterToCabinet",
    ),
    TargetTask(
        "PickPlaceCounterToSink",
        "pick_place",
        "placement",
        "PnPCounterToSink",
        "PnPCounterToSink",
        "MG_PnPCounterToSink",
        "MG_ModernPnPCounterToSink",
    ),
    TargetTask("OpenCabinet", "open_door", "linear_joint", None, "OpenCabinet", "MG_OpenCabinet"),
    TargetTask(
        "OpenDrawer", "open_door", "linear_joint", "OpenDrawer", "OpenDrawer", "MG_OpenDrawer"
    ),
    TargetTask("OpenOven", "open_door", "rotary_joint", None, "OpenOven", "MG_OpenOven"),
    TargetTask(
        "CloseCabinet", "close_door", "linear_joint", None, "CloseCabinet", "MG_CloseCabinet"
    ),
    TargetTask(
        "CloseDrawer", "close_door", "linear_joint", "CloseDrawer", "CloseDrawer", "MG_CloseDrawer"
    ),
    TargetTask("CloseOven", "close_door", "rotary_joint", None, "CloseOven", "MG_CloseOven"),
    TargetTask(
        "OpenBlenderLid", "lid", "rotary_joint", None, "OpenBlenderLid", "MG_OpenBlenderLid"
    ),
    TargetTask(
        "CloseBlenderLid",
        "lid",
        "rotary_joint",
        None,
        "CloseBlenderLid",
        "MG_CloseBlenderLid",
        "MG_ModernCloseBlenderLid",
    ),
    TargetTask(
        "CoffeeSetupMug",
        "insertion",
        "placement",
        "CoffeeSetupMug",
        "CoffeeSetupMug",
        "MG_CoffeeSetupMug",
    ),
    TargetTask(
        "CoffeeServeMug",
        "insertion",
        "placement",
        "CoffeeServeMug",
        "CoffeeServeMug",
        "MG_CoffeeServeMug",
    ),
    TargetTask(
        "TurnOnMicrowave",
        "press_button",
        "button",
        "TurnOnMicrowave",
        "TurnOnMicrowave",
        "MG_TurnOnMicrowave",
        "MG_ModernTurnOnMicrowave",
    ),
    TargetTask(
        "StartCoffeeMachine",
        "press_button",
        "button",
        None,
        "StartCoffeeMachine",
        "MG_StartCoffeeMachine",
        "MG_ModernStartCoffeeMachine",
    ),
    TargetTask(
        "SlideDishwasherRack",
        "slide_rack",
        "linear_joint",
        None,
        "SlideDishwasherRack",
        "MG_SlideDishwasherRack",
    ),
    TargetTask(
        "SlideOvenRack", "slide_rack", "linear_joint", None, "SlideOvenRack", "MG_SlideOvenRack"
    ),
    TargetTask(
        "TurnOnSinkFaucet",
        "turn_lever",
        "lever",
        "TurnOnSinkFaucet",
        "TurnOnSinkFaucet",
        "MG_TurnOnSinkFaucet",
    ),
    TargetTask("TurnOnToaster", "turn_lever", "lever", None, "TurnOnToaster", "MG_TurnOnToaster"),
    TargetTask("TurnOnStove", "twist_knob", "knob", "TurnOnStove", "TurnOnStove", "MG_TurnOnStove"),
    TargetTask(
        "AdjustToasterOvenTemperature",
        "twist_knob",
        "knob",
        None,
        "AdjustToasterOvenTemperature",
        "MG_AdjustToasterOvenTemperature",
    ),
)


SUPPORTED_TASK_NAMES = tuple(task.name for task in TARGET_TASKS)


def target_task(name: str) -> TargetTask:
    for task in TARGET_TASKS:
        if task.name == name:
            return task
    raise KeyError(f"unsupported task: {name}")
