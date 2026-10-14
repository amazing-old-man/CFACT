"""Keep upstream generated MJCF files inside the explicit writable workspace.

RoboCasa creates a timestamp XML beside input assets. Only that temporary path
is redirected. Mesh/texture paths are made absolute to their original asset
directory; all physical MJCF values and the upstream constructor stay intact.
"""
import os
from pathlib import Path
import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET


class _GeneratedPath:
    def __init__(self, directory):
        self.directory=directory

    def __getattr__(self,name):return getattr(os.path,name)

    def join(self,*parts):
        if parts and re.fullmatch(r"\d+_\d+_"+str(os.getpid())+r"\.xml",str(parts[-1])):
            return str(self.directory/str(parts[-1]))
        return os.path.join(*parts)


def absolute_asset_paths(xml, original_directory):
    tree=ET.fromstring(xml)
    for element in tree.iter():
        value=element.get("file")
        if value and not os.path.isabs(value):
            element.set("file",str((original_directory/value).resolve()))
    return ET.tostring(tree,encoding="unicode")


def install_generated_xml_redirect(workspace):
    import robocasa.models.objects.objects as objects
    if getattr(objects,"_cfact_xml_redirect",False):return
    directory=Path(workspace).resolve()/"tmp/generated_mjcf"/str(os.getpid())
    directory.mkdir(parents=True,exist_ok=True)
    # Module-local proxy: do not change the global os module or other jobs.
    proxy=SimpleNamespace(**{key:getattr(os,key) for key in dir(os) if key!="path"})
    proxy.path=_GeneratedPath(directory)
    objects.os=proxy
    original=objects.MJCFObject.postprocess_model_xml
    def postprocess(self,xml):
        return absolute_asset_paths(original(self,xml),Path(self.mjcf_path).parent)
    objects.MJCFObject.postprocess_model_xml=postprocess
    objects._cfact_xml_redirect=True


def install_write_boundary(workspace):
    """Fail before an unexpected external write; applies only to this process."""
    import sys
    root=Path(workspace).resolve()
    def inside(path):
        if isinstance(path,int):return
        p=Path(os.fsdecode(path)).resolve()
        if p in {Path('/dev/null'),Path('/dev/tty')}:return
        if not p.is_relative_to(root):raise PermissionError("CFACT_EXTERNAL_WRITE_BLOCKED:"+str(p))
    def audit(event,args):
        if event=="open":
            path,mode,flags=args
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):inside(path)
        elif event in {"os.remove","os.rmdir","os.mkdir","os.chmod","os.chown","os.truncate"}:inside(args[0])
        elif event in {"os.rename","os.link"}:
            inside(args[0]);inside(args[1])
        elif event=="os.symlink":inside(args[1])
    sys.addaudithook(audit)
