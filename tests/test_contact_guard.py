"""Measured contact guard distinguishes arm penetration from gripper contact."""

from __future__ import annotations
import json
from types import SimpleNamespace

import mujoco

from cfact.execution.options import MeasuredArmContactGuard


def test_arm_penetration_triggers_with_body_names_but_gripper_contact_does_not() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <worldbody>
        <body name="robot0_base">
          <body name="robot0_link5"><geom size="0.01"/></body>
          <body name="gripper0_right_finger"><site name="tcp"/></body>
        </body>
        <body name="cabinet_door"><geom size="0.01"/></body>
      </worldbody>
    </mujoco>""")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_link5")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_finger")
    door = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet_door")
    env = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(model=SimpleNamespace(_model=model)),
            robots=[SimpleNamespace(eef_site_id={"right": site_id})],
        )
    )
    guard = MeasuredArmContactGuard(env, penetration_limit_m=0.001)
    trace = SimpleNamespace(
        robot_contacts_json=[
            json.dumps([{"body_ids": [finger, door], "distance_m": -0.004, "geom_ids": [1, 2]}])
        ],
        commanded_actions=[1],
        sim_time=[0.0, 0.05],
    )
    assert guard.check(trace) is None
    trace.robot_contacts_json[-1] = json.dumps(
        [
            {"body_ids": [link, door], "distance_m": -0.0008, "geom_ids": [0, 2]},
            {"body_ids": [link, door], "distance_m": -0.002, "geom_ids": [0, 2]},
        ]
    )
    report = guard.check(trace)
    assert report is not None
    assert report["contacts"][0]["body_names"] == ["robot0_link5", "cabinet_door"]
    assert report["contacts"][0]["distance_m"] == -0.002
    assert report["control_step"] == 1
