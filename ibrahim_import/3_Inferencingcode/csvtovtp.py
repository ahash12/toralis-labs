#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV -> VTP exporter (ParaView/VTK PolyData)

Two modes:
1) Point-cloud VTP (no mesh): writes points from CSV + attached point-data arrays.
2) Mesh-attached VTP (recommended if you have the wall surface mesh):
   reads an existing surface mesh (.vtp/.stl/.ply/.obj/.vtk etc), maps CSV rows onto mesh points
   via nearest-neighbor, and writes a NEW .vtp with the CSV fields attached to the mesh.

This script is intentionally consistent with the robust column-picking + unit-scale sanity
approach used in your inference code. :contentReference[oaicite:0]{index=0}

Dependencies:
  pip install numpy pandas pyvista
Optional (faster NN mapping):
  pip install scipy
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS (EDIT THESE)
# =============================================================================

# Input CSV (your Fluent wall_data_Re*.csv)
CSV_PATH = Path(r"/mnt/data/wall_data_Re150.csv")  # <-- change

# OPTIONAL: existing surface mesh to attach data to (recommended).
# If you leave this as None, you'll get a point-cloud VTP (no connectivity).
MESH_PATH: Optional[Path] = None
# Example (Windows):
# MESH_PATH = Path(r"C:\Users\radie\Desktop\1_TrainingSIMS\realvesselSIM1\coronary_extracted_vessel.vtp")
# Or if you only have Fluent .msh, export a wall surface to .stl/.vtp first.

# Output VTP
OUT_VTP_PATH = Path(r"/mnt/data/wall_data_Re150.vtp")  # <-- change

# Nearest-neighbor mapping controls (only used if MESH_PATH is provided)
AUTO_SCALE_COORDS = True   # attempts mm<->m scaling if CSV coords mismatch mesh coords
NN_DISTANCE_REPORT = True  # prints basic p95/p99 mapping distance diagnostics

# If True, and if your CSV does NOT have WSS vector columns, we still export whatever scalar columns exist.
EXPORT_ALL_NUMERIC_COLUMNS = True

# =============================================================================
# Helpers (column parsing + NN mapping + VTP writing)
# =============================================================================

def _lower_map_cols(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in df.columns}

def _pick_col(cols_map: Dict[str, str], *names: str) -> str:
    """
    Pick a column by exact normalized name or by substring fallback.
    """
    # exact match
    for n in names:
        k = n.strip().lower()
        if k in cols_map:
            return cols_map[k]

    # substring fallback
    wants = [n.strip().lower() for n in names]
    for want in wants:
        for k_norm, orig in cols_map.items():
            if want in k_norm:
                return orig

    raise KeyError(f"Missing column. Tried={names}. Available={list(cols_map.values())}")

def try_read_csv_table(path: Path) -> pd.DataFrame:
    """
    Fluent exports are usually comma-separated, but some are whitespace-delimited.
    """
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return pd.read_csv(path)

def extract_xyz_and_optional_wss(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      xyz: (N,3)
      data_fields: dict of arrays to attach (each shape (N,) or (N,3) or (N,k))
    """
    cols = _lower_map_cols(df)

    # XYZ (try common Fluent + generic conventions)
    xcol = _pick_col(cols, "x-coordinate", "x", "x_coordinate")
    ycol = _pick_col(cols, "y-coordinate", "y", "y_coordinate")
    zcol = _pick_col(cols, "z-coordinate", "z", "z_coordinate")
    xyz = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32)

    data_fields: Dict[str, np.ndarray] = {}

    # WSS vector (common Fluent naming patterns + your pipeline patterns)
    def _try_vec3() -> Optional[np.ndarray]:
        try:
            wx = _pick_col(cols, "x-wall-shear", "x_wall_shear", "wss_x", "x-wss", "wssx")
            wy = _pick_col(cols, "y-wall-shear", "y_wall_shear", "wss_y", "y-wss", "wssy")
            wz = _pick_col(cols, "z-wall-shear", "z_wall_shear", "wss_z", "z-wss", "wssz")
            return df[[wx, wy, wz]].to_numpy(dtype=np.float32)
        except Exception:
            return None

    wss_vec = _try_vec3()
    if wss_vec is not None:
        data_fields["WSS_vec"] = wss_vec
        data_fields["WSS_mag"] = np.linalg.norm(wss_vec.astype(np.float64), axis=1).astype(np.float32)

    # Also grab common scalar columns if present (pressure, wall-shear magnitude, etc.)
    # If EXPORT_ALL_NUMERIC_COLUMNS is True, we’ll attach all numeric columns (except xyz + wss components duplicated).
    if EXPORT_ALL_NUMERIC_COLUMNS:
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        # Remove xyz columns from export (we store xyz as geometry, not point data)
        for c in [xcol, ycol, zcol]:
            if c in numeric_df.columns:
                numeric_df.drop(columns=[c], inplace=True)

        # If we already exported WSS_vec from 3 cols, remove those cols to avoid duplicates
        if wss_vec is not None:
            for k in ["x-wall-shear", "x_wall_shear", "wss_x", "x-wss", "wssx",
                      "y-wall-shear", "y_wall_shear", "wss_y", "y-wss", "wssy",
                      "z-wall-shear", "z_wall_shear", "wss_z", "z-wss", "wssz"]:
                k_norm = k.lower()
                if k_norm in cols:
                    orig = cols[k_norm]
                    if orig in numeric_df.columns:
                        numeric_df.drop(columns=[orig], inplace=True)

        # Attach remaining numeric columns as scalars
        for c in numeric_df.columns:
            arr = numeric_df[c].to_numpy()
            # ensure 1D scalar
            if arr.ndim == 1 and arr.shape[0] == xyz.shape[0]:
                data_fields[str(c)] = arr.astype(np.float32)

    return xyz, data_fields

def bbox_diag(xyz: np.ndarray) -> float:
    mn = np.min(xyz, axis=0)
    mx = np.max(xyz, axis=0)
    return float(np.linalg.norm(mx - mn))

def nearest_neighbor_map(query_xyz: np.ndarray, ref_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      idx: (M,) indices into ref_xyz
      dist: (M,) Euclidean distances
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(ref_xyz)
        dist, idx = tree.query(query_xyz, k=1)
        return idx.astype(np.int64), dist.astype(np.float32)
    except Exception:
        # fallback (slower)
        M = int(query_xyz.shape[0])
        idx_out = np.empty(M, dtype=np.int64)
        dist_out = np.empty(M, dtype=np.float32)
        chunk = 1500
        for i0 in range(0, M, chunk):
            i1 = min(i0 + chunk, M)
            q = query_xyz[i0:i1]
            d2 = ((q[:, None, :] - ref_xyz[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2, axis=1)
            dist = np.sqrt(d2[np.arange(idx.size), idx])
            idx_out[i0:i1] = idx
            dist_out[i0:i1] = dist.astype(np.float32)
        return idx_out, dist_out

def choose_csv_scale(csv_xyz: np.ndarray, mesh_xyz: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Heuristic: if bbox sizes differ by ~1000x, scale CSV to match mesh.
    Mirrors your inference validation logic. :contentReference[oaicite:1]{index=1}
    """
    diag_csv = bbox_diag(csv_xyz)
    diag_mesh = bbox_diag(mesh_xyz)
    ratio = (diag_mesh / diag_csv) if (diag_csv > 1e-12 and diag_mesh > 1e-12) else 1.0

    candidates = [1.0]
    # common mm<->m mismatch (~1000x)
    if 0.0005 <= ratio <= 0.002 or 500.0 <= ratio <= 2000.0:
        candidates.append(float(ratio))

    best_scale = 1.0
    best_p99 = float("inf")
    for sc in candidates:
        xyz2 = (csv_xyz * sc).astype(np.float32)
        _, dist = nearest_neighbor_map(xyz2, mesh_xyz.astype(np.float32))
        p99 = float(np.quantile(dist, 0.99)) if dist.size else float("inf")
        if p99 < best_p99:
            best_p99 = p99
            best_scale = sc

    out = (csv_xyz * best_scale).astype(np.float32)
    info = {
        "auto_scale": float(best_scale),
        "diag_csv": float(diag_csv),
        "diag_mesh": float(diag_mesh),
        "diag_ratio_mesh_over_csv": float(ratio),
        "best_p99_for_scale_choice": float(best_p99),
    }
    return out, info

def write_vtp_with_pyvista(
    out_vtp: Path,
    points_xyz: np.ndarray,
    faces: Optional[np.ndarray],
    point_data: Dict[str, np.ndarray],
) -> None:
    """
    faces:
      - None -> point cloud
      - (F,3) int -> triangle connectivity (0-based)
    """
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("PyVista not available. Install with: pip install pyvista") from e

    out_vtp = Path(out_vtp)
    out_vtp.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {pts.shape}")

    if faces is None:
        mesh = pv.PolyData(pts)
    else:
        tri = np.asarray(faces, dtype=np.int64)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError(f"faces must be (F,3), got {tri.shape}")
        if tri.size and (tri.min() < 0 or tri.max() >= pts.shape[0]):
            raise ValueError("faces contain indices out of range for points_xyz")
        # pyvista face array format: [3,i0,i1,i2, 3,j0,j1,j2, ...]
        faces_pv = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
        mesh = pv.PolyData(pts, faces_pv)

    # attach arrays
    for k, arr in point_data.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            if a.shape[0] != pts.shape[0]:
                raise ValueError(f"Point array '{k}' length {a.shape[0]} != N {pts.shape[0]}")
            mesh.point_data[str(k)] = a.astype(np.float32)
        elif a.ndim == 2:
            if a.shape[0] != pts.shape[0]:
                raise ValueError(f"Point array '{k}' first dim {a.shape[0]} != N {pts.shape[0]}")
            mesh.point_data[str(k)] = a.astype(np.float32)
        else:
            raise ValueError(f"Point array '{k}' must be 1D or 2D, got {a.shape}")

    mesh.save(str(out_vtp))

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

    df = try_read_csv_table(CSV_PATH)
    csv_xyz, csv_fields = extract_xyz_and_optional_wss(df)

    # -------------------------
    # Mode A: No mesh -> point cloud VTP
    # -------------------------
    if MESH_PATH is None:
        print("[MODE] No mesh provided -> exporting point-cloud VTP (no triangles).")
        print(f"[IN ] CSV : {CSV_PATH}")
        print(f"[OUT] VTP : {OUT_VTP_PATH}")
        write_vtp_with_pyvista(
            out_vtp=OUT_VTP_PATH,
            points_xyz=csv_xyz,
            faces=None,
            point_data=csv_fields,
        )
        print("[OK] Wrote point-cloud VTP.")
        return

    # -------------------------
    # Mode B: Mesh provided -> attach CSV data to mesh points
    # -------------------------
    if not Path(MESH_PATH).exists():
        raise FileNotFoundError(f"MESH_PATH not found: {MESH_PATH}")

    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("PyVista not available. Install with: pip install pyvista") from e

    print("[MODE] Mesh provided -> mapping CSV rows onto mesh points (nearest-neighbor).")
    print(f"[IN ] CSV  : {CSV_PATH}")
    print(f"[IN ] MESH : {MESH_PATH}")
    print(f"[OUT] VTP  : {OUT_VTP_PATH}")

    surf = pv.read(str(MESH_PATH))
    mesh_pts = np.asarray(surf.points, dtype=np.float32)

    # Optional unit scaling sanity (mm<->m)
    scale_info = {"auto_scale": 1.0}
    csv_xyz_mapped = csv_xyz
    if AUTO_SCALE_COORDS:
        csv_xyz_mapped, scale_info = choose_csv_scale(csv_xyz, mesh_pts)

    # NN mapping: CSV rows -> mesh point indices
    nn_idx, nn_dist = nearest_neighbor_map(csv_xyz_mapped.astype(np.float32), mesh_pts.astype(np.float32))

    if NN_DISTANCE_REPORT:
        p95 = float(np.quantile(nn_dist, 0.95)) if nn_dist.size else float("nan")
        p99 = float(np.quantile(nn_dist, 0.99)) if nn_dist.size else float("nan")
        print(f"[MAP] auto_scale={scale_info.get('auto_scale', 1.0):.6g}  "
              f"mean_dist={float(np.mean(nn_dist)):.6g}  p95={p95:.6g}  p99={p99:.6g}")

    # Aggregate CSV fields onto mesh points:
    # - If multiple CSV rows hit the same mesh point, average them.
    N = mesh_pts.shape[0]
    out_fields: Dict[str, np.ndarray] = {}

    for name, arr in csv_fields.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            sumv = np.zeros((N,), dtype=np.float64)
            cnt = np.zeros((N,), dtype=np.float64)
            np.add.at(sumv, nn_idx, a.astype(np.float64))
            np.add.at(cnt, nn_idx, 1.0)
            out = (sumv / np.where(cnt > 0, cnt, 1.0)).astype(np.float32)
            out_fields[name] = out
            out_fields[f"{name}_hitcount"] = cnt.astype(np.float32)
        elif a.ndim == 2:
            C = a.shape[1]
            sumv = np.zeros((N, C), dtype=np.float64)
            cnt = np.zeros((N, 1), dtype=np.float64)
            np.add.at(sumv, nn_idx, a.astype(np.float64))
            np.add.at(cnt, nn_idx, 1.0)
            out = (sumv / np.where(cnt > 0, cnt, 1.0)).astype(np.float32)
            out_fields[name] = out
            out_fields[f"{name}_hitcount"] = cnt[:, 0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported field shape for '{name}': {a.shape}")

    # Attach to mesh and save as .vtp
    for k, v in out_fields.items():
        surf.point_data[str(k)] = v

    surf.save(str(OUT_VTP_PATH))
    print("[OK] Wrote mesh-attached VTP.")

if __name__ == "__main__":
    main()