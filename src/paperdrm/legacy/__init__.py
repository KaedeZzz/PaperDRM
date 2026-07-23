"""
Legacy / experimental modules kept for reference but not part of the active pipeline.

Anything in here is either:
- superseded by a current module (Hough -> Gabor, ImageParam -> DRPConfig),
- experimental and currently unused (spectral TV decomposition),
- or a deprecation shim from an old API (config.py).

Do not import from here in production code. Move things out of legacy/ if you
decide to revive them.
"""

import warnings

warnings.warn(
    "paperdrm.legacy contains code that is not part of the active pipeline.",
    DeprecationWarning,
    stacklevel=2,
)
