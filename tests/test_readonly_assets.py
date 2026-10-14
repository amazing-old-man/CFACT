import os
from pathlib import Path
from cfact.runtime.readonly_assets import _GeneratedPath, absolute_asset_paths


def test_generated_path_redirects_only_current_process_temporary_xml(tmp_path):
    proxy=_GeneratedPath(tmp_path)
    name=f"1791954000_12345_{os.getpid()}.xml"
    assert proxy.join('/readonly/assets',name)==str(tmp_path/name)
    assert proxy.join('/readonly/assets','model.xml')=='/readonly/assets/model.xml'


def test_relocating_xml_preserves_physical_values_and_original_asset_resolution(tmp_path):
    from xml.etree import ElementTree as E
    xml='<mujoco><asset><mesh name="m" file="../mesh.stl"/></asset><worldbody><geom mass="2.0" friction="1 0.1 0.1" mesh="m"/></worldbody></mujoco>'
    result=E.fromstring(absolute_asset_paths(xml,tmp_path/'object'))
    assert result.find('asset/mesh').get('file')==str(tmp_path/'mesh.stl')
    assert result.find('worldbody/geom').attrib==E.fromstring(xml).find('worldbody/geom').attrib
