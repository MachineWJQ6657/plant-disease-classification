"""Portable disk names, with unchanged canonical upstream paths in manifests."""
from pathlib import Path, PurePosixPath
import re
import os
from urllib.parse import unquote

def storage_root(root):
    """Use the Windows extended-length namespace for long upstream filenames."""
    value=os.path.abspath(os.path.normpath(os.fspath(root)))
    if os.name=='nt' and not value.startswith('\\\\?\\'):
        value='\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
    return Path(value)

def portable_component(name):
    # Encode literal percent first, making the mapping reversible and collision-free.
    value=re.sub(r'[%<>:"\\|?*\x00-\x1f]',lambda m:'%%%02X'%ord(m[0]),name)
    value=re.sub(r'[. ]+$',lambda m:''.join('%%%02X'%ord(c) for c in m[0]),value)
    if re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)',value,re.I):
        value='%%%02X'%ord(value[0])+value[1:]
    return value

def disk_path(root,canonical):
    parts=PurePosixPath(canonical).parts
    if not parts or '..' in parts or PurePosixPath(canonical).is_absolute():
        raise ValueError('Expected safe relative upstream path')
    return storage_root(root).joinpath(*(portable_component(p) for p in parts))

def canonical_path(path,root):
    return '/'.join(unquote(p) for p in storage_root(path).relative_to(storage_root(root)).parts)
