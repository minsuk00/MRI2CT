"""MC-IDDPM baseline package.

The cloned author code uses non-package-relative imports
(e.g. `import diffusion.GaussianDiffusion as gd` inside `diffusion/Create_diffusion.py`).
We add this directory to `sys.path` so those imports resolve without editing the
upstream files.
"""
import sys as _sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
