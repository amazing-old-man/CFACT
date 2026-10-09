"""Version-isolated MimicGen interface compatibility for RoboCasa365."""

from __future__ import annotations

import numpy as np

from mimicgen.env_interfaces.robocasa.single_stage.mg_blender import (
    MG_CloseBlenderLid,
)
from mimicgen.env_interfaces.robocasa.single_stage.mg_pnp import MG_PnPCounterToSink
from mimicgen.env_interfaces.robocasa.single_stage.mg_microwave import (
    MG_TurnOnMicrowave,
)
from mimicgen.env_interfaces.robocasa.single_stage.mg_coffee import (
    MG_StartCoffeeMachine,
)

from cfact.adapters.events import (
    coffee_machine_button_signals,
    select_coffee_machine_button,
    microwave_button_signals,
)
from cfact.adapters.action_codec import ActionCodec


class MG_ModernPnPCounterToSink(MG_PnPCounterToSink):
    """Use a stable RoboCasa365 basin frame without changing task semantics.

    The local legacy stack intentionally retains the older ``<sink>_bottom``
    fixture name. RoboCasa365 uses one reset-region geom for a single basin and
    two for a double basin.  Sampling one of those geoms on every query makes a
    double-basin reference jump by roughly one basin width inside a trajectory.
    We instead use the centroid of all basin-region frames.  This is attached
    to the same fixed fixture, remains constant within an episode, and treats
    the two sides symmetrically. Action, contact, and official success semantics
    remain inherited unchanged.
    """

    def _sink_reference_pose(self):
        regions = sorted(self.env.sink.get_reset_regions().keys())
        if not regions:
            raise ValueError("modern sink exposes no reset region")
        poses = np.asarray(
            [
                self.get_object_pose(
                    obj_name=f"{self.env.sink.name}_reg_{region}",
                    obj_type="geom",
                )
                for region in regions
            ]
        )
        rotations = poses[:, :3, :3]
        if not np.allclose(rotations, rotations[0], atol=1e-8, rtol=0.0):
            raise ValueError("modern sink basin regions do not share one orientation")
        reference = poses[0].copy()
        reference[:3, 3] = poses[:, :3, 3].mean(axis=0)
        return reference

    def get_object_poses(self):
        return {
            "obj": self.get_object_pose(
                obj_name=self.env.objects["obj"].root_body,
                obj_type="body",
                obj_pos_offset="bottom",
            ),
            "sink": self._sink_reference_pose(),
        }


class MG_ModernTurnOnMicrowave(MG_TurnOnMicrowave):
    """Expose physical button contact separately from terminal success.

    Upstream exposes only the success signal, which becomes true after the
    button has toggled and the gripper has withdrawn.  The adaptive tree needs
    the earlier observed contact event without changing the official task
    predicate.
    """

    def get_subtask_term_signals(self):
        return microwave_button_signals(self.env)


class MG_ModernStartCoffeeMachine(MG_StartCoffeeMachine):
    """Use one deterministic, physically relevant button pose and contact event.

    The upstream interface draws a random start-button geom on every pose query,
    which can make the object frame jump between buttons inside one trajectory.
    The modern adapter selects the currently contacted button, or otherwise the
    nearest button to the end effector with a stable name tie-break.
    """

    def get_object_poses(self):
        eef_position = self.get_robot_eef_pose()[:3, 3]
        button = select_coffee_machine_button(self.env, eef_position)
        return {"button": self.get_object_pose(obj_name=button, obj_type="geom")}

    def get_subtask_term_signals(self):
        return coffee_machine_button_signals(self.env)


class MG_ModernCloseBlenderLid(MG_CloseBlenderLid):
    """Compatibility name; action interpretation now uses the common codec."""

    def action_to_target_pose(self, action, relative=True):
        return ActionCodec(self).decode(action)
