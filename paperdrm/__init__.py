"""
PaperDRM: laid-line detection in historical paper from DRP image stacks.

Pipeline stages (each is a subpackage):
- loader:      DRP stack loading, caching, and configuration
- drp:         operations on the DRP stack (slicing, building, masking)
- features:    per-pixel features from DRP (direction map, spherical descriptors)
- enhance:     orientation -> laid-line likelihood grayscale (trig masks)
- detect:      laid-line period and grid estimation (Gabor)
- viz:         plotting helpers for each stage
- evaluation:  evaluating detection results (consistency, phantoms, ground truth)

Top-level entry: ImagePack + Settings.
"""

from paperdrm.loader import ImagePack, Settings

__all__ = ["ImagePack", "Settings"]
