# embree_bypass.py
import pyvista as pv
import numpy as np
from typing import Tuple, Any

def _vtk_fallback_multi_ray_trace(
    self: pv.PolyData,
    origins: np.ndarray,
    direction_vectors: np.ndarray,
    first_point: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    VTK-only implementation of multi_ray_trace.
    Works on arm64, no Embree required.
    """
    origins = np.asarray(origins, dtype=float)
    direction_vectors = np.asarray(direction_vectors, dtype=float)

    # Normalise directions and create far points
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_dirs = direction_vectors / norms
    far_points = origins + unit_dirs * self.length  # mesh diagonal = ray length

    hit_points = []
    hit_indices = []

    for origin, end in zip(origins, far_points):
        try:
            point, ind = self.ray_trace(origin, end, first_point=first_point)
            if len(point) > 0:
                hit_points.append(point[0])
                hit_indices.append(ind[0])
            else:
                hit_points.append(np.full(3, np.nan))
                hit_indices.append(-1)
        except Exception:
            hit_points.append(np.full(3, np.nan))
            hit_indices.append(-1)

    return (
        np.array(hit_points),
        np.array(hit_indices, dtype=int),
        []  # cells – not used by the project
    )

# ------------------------------------------------------------------
# Apply the patch **only if Embree is not available**
# ------------------------------------------------------------------
if not getattr(pv, "embree_available", False):
    # Bind the fallback to the *instance* method
    pv.PolyData.multi_ray_trace = _vtk_fallback_multi_ray_trace # type: ignore[attr-defined]
    print("Embree not available → using VTK-only ray tracing (arm64 native)")

