"""
PaperDRM: laid-line detection in historical paper from DRP image stacks.

Pipeline stages (each is a subpackage):
- stage0_loader:      DRP stack loading, caching, and configuration
- stage0_drp:         operations on the DRP stack (slicing, building, masking)
- stage1_features:    per-pixel features from DRP (direction map, spherical descriptors)
- stage2_enhance:     orientation -> laid-line likelihood grayscale (trig masks)
- stage3_detect:      laid-line period and grid estimation (Gabor)
- stage4_viz:         plotting helpers for each stage
- stage5_evaluation:  evaluating detection results (consistency, phantoms, ground truth)
- legacy:             quarantined unused code (not part of the active pipeline)

Top-level entry: ImagePack + Settings.
"""

from paperdrm.stage0_loader import ImagePack, Settings

__all__ = ["ImagePack", "Settings"]
