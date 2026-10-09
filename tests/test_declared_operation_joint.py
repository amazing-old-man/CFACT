from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
from cfact.execution.binding import declared_operation_joint, bind_joint
from test_execution_contract import Model


def interface(joint):
    return SimpleNamespace(env=SimpleNamespace(fixture=SimpleNamespace(
        flow_joint=ET.Element('joint',name=joint))))


def test_official_role_resolves_multiple_ancestor_joints():
    model=Model()
    model.jnt_bodyid[:]=1
    role={'fixture_attribute':'fixture','joint_property':'flow_joint'}
    name=declared_operation_joint(interface('cabinet_slide'),role)
    assert bind_joint(model,{'body_id':2},explicit_joint=name)['joint_name']=='cabinet_slide'
    # Lack of declaration retains the ambiguity guard.
    assert declared_operation_joint(interface('cabinet_slide'),None) is None
    with pytest.raises(ValueError,match='ambiguous'):
        bind_joint(model,{'body_id':2})


def test_official_role_cannot_select_unrelated_joint():
    name=declared_operation_joint(interface('unrelated_hinge'),
        {'fixture_attribute':'fixture','joint_property':'flow_joint'})
    with pytest.raises(ValueError,match='not_ancestor'):
        bind_joint(Model(),{'body_id':2},explicit_joint=name)


@pytest.mark.parametrize('declaration',[
    {'fixture_attribute':'fixture','joint_property':'_private'},
    {'fixture_attribute':'fixture.other','joint_property':'flow_joint'},
    {'fixture_attribute':'fixture','joint_property':'flow_joint','fallback':'closest'},
])
def test_invalid_declarations_rejected(declaration):
    with pytest.raises(ValueError,match='invalid_operation_joint_role'):
        declared_operation_joint(interface('cabinet_slide'),declaration)


def test_free_object_remains_explicitly_unsupported_by_articulation_binding():
    model=Model()
    model.jnt_type[:]=0
    with pytest.raises(ValueError,match=r'ambiguous:\[\]'):
        bind_joint(model,{'body_id':2})


def test_missing_official_property_fails_closed():
    env=SimpleNamespace(env=SimpleNamespace(fixture=SimpleNamespace(flow_joint=None)))
    with pytest.raises(ValueError,match='role_missing'):
        declared_operation_joint(env,{'fixture_attribute':'fixture','joint_property':'flow_joint'})
