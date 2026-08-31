""" Bridging helpers between the public dataclasses and the C++ backend.

The dataclasses in ``inlier.core.Dataclasses`` remain the user-facing
source of truth; the pybind structs mirror their field names, so the
conversion is a mechanical field copy.
"""   

from __future__ import annotations

from dataclasses import fields

import numpy as np

from inlier import _inlier_pybind as _ip
from inlier.core.Dataclasses import InLiER_Config

_INLIER_FIELD_NAMES = (
    "N_h", "z_min", "z_max", "r_max", "N_r", "N_a", "N_s", "cell_size",
    "xy_max", "window", "max_kp_per_slice", "ransac_iters",
    "ransac_dist_thresh", "ransac_min_inliers", "point_mode",
    "shape_radius", "shape_min_neighbors",
)


def to_cpp_inlier_config(cfg: InLiER_Config) -> "_ip.InLiERConfig":
    out = _ip.InLiERConfig()
    for name in _INLIER_FIELD_NAMES:
        setattr(out, name, getattr(cfg, name))
    out.max_kp_total = int(cfg.max_kp_total)
    return out

def to_cpp_stage_config(cpp_cls, dataclass_cfg):
    """Copy a stage config dataclass (Shortlist/BEAMScore/Rerank/Verify)
    into its C++ mirror class."""
    out = cpp_cls()
    for f in fields(dataclass_cfg):
        value = getattr(dataclass_cfg, f.name)
        if f.name == "topk_pct":
            value = -1.0 if value is None else float(value)
        setattr(out, f.name, value)
    return out


def plane_dict_to_cpp(plane: dict) -> "_ip.Plane":
      """Plane dict (normal/d/point/inliers) -> C++ Plane."""
      out = _ip.Plane()
      out.normal = np.asarray(plane["normal"], dtype=np.float64)
      d = np.asarray(plane["d"], dtype=np.float64).reshape(-1)
      out.d = float(d[0]) if d.size else 0.0
      out.point = np.asarray(plane["point"], dtype=np.float64)
      return out
     
    
def plane_cpp_to_dict(plane: "_ip.Plane") -> dict:
    """C++ Plane -> the dict shape returned by InLiER._ransac_plane."""
    return {
        "normal": np.asarray(plane.normal, dtype=np.float64),
        "d": np.array([plane.d], dtype=np.float64),
        "point": np.asarray(plane.point, dtype=np.float64),
        "inliers": np.asarray(plane.inliers, dtype=bool),
    }   
