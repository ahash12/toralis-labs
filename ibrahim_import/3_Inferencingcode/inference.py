#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFERENCE SCRIPT — Unseen Fluent .msh -> graph -> WSS prediction -> clinician-friendly exports

Pipeline context:
- Step A: Fluent generates wall_data_Re*.csv (ground truth for training/validation)
- Step B: trainingpre.py creates Re*.pt graphs (pos/edge_index/face/x + labels y + re)
- Step C: training.py trains WSSNet and saves best_model.pt checkpoint
- This script:
    (1) preprocesses a NEW .msh (unseen geometry) into a graph
    (2) loads best_model.pt
    (3) runs inference (WSS vector per wall node)
    (4) exports:
         A) .vtp surface with WSS as point data (ParaView)
         B) .csv aligned to wall nodes
    (5) optionally validates against a provided Fluent wall CSV via nearest-neighbor mapping

Author: (you + ChatGPT)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data


# ======================================================================================
# Model (must match training.py exactly)
# ======================================================================================

class GraphMP(nn.Module):
    """
    Simple message passing:
      h' = W_self h + W_nei * mean(h_neighbors)
    Implemented with pure PyTorch index_add_ (no torch_scatter).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei  = nn.Linear(in_dim, out_dim)
        self.dropout  = dropout
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]  # messages: col -> row

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])

        deg = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        ones = torch.ones((row.numel(), 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, row, ones)

        agg = agg / (deg + 1e-12)

        out = self.lin_self(x) + self.lin_nei(agg)
        out = self.norm(out)
        out = F.silu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class WSSNet(nn.Module):
    """
    Node-level regression network:
      input: data.x (7 geom feats, optionally +1 Re feature)
      output: y_hat [N,3] (WSS vector)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([
            GraphMP(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        ei = data.edge_index.long()
        h = self.enc(x)
        for layer in self.layers:
            h_new = layer(h, ei)
            h = h + h_new
        return self.dec(h)


# ======================================================================================
# Fluent legacy ASCII .msh parsing + wall-surface graph construction
# (Matches your preprocessing_wall19.py logic; default wall_zone=19)
# ======================================================================================

def _strip_comments(line: str) -> str:
    return line.split(";", 1)[0] if ";" in line else line


def _read_all_lines(msh_path: Path) -> List[str]:
    with open(msh_path, "r", errors="ignore") as f:
        lines = f.readlines()
    return [ln.rstrip("\n") for ln in lines]


def _find_scaling_factor(lines: List[str]) -> float:
    for ln in lines:
        s = _strip_comments(ln).strip()
        if s.startswith("(meshing-to-solver-scaling-factor"):
            parts = s.strip("()").split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except Exception:
                    pass
    return 1.0


def _parse_int_auto(tok: str) -> int:
    """
    NOTE: Your trainingpre.py parsed digit-only tokens as HEX by default.
    Here we keep the conservative heuristic:
      - if a-f present -> base16 else base10
    If you hit 'wall zone not found' on some meshes, the robust fix is to force-hex parsing.
    """
    t = tok.strip().lower()
    if any(c in t for c in "abcdef"):
        return int(t, 16)
    return int(t, 10)


def parse_fluent_ascii_msh_wall(msh_path: Path, wall_zone: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      nodes_xyz: [N,3] float64 (global nodes, 0-based indexing in array)
      faces: [F,3] int64 (global node indices, 0-based)
    """
    lines = _read_all_lines(msh_path)
    scale = _find_scaling_factor(lines)

    # ---- nodes ----
    nodes: Dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        raw = _strip_comments(lines[i]).strip()
        if raw.startswith("(10"):
            hdr = raw
            while hdr.count("(") > hdr.count(")") and i + 1 < len(lines):
                i += 1
                hdr += " " + _strip_comments(lines[i]).strip()

            try:
                inner = hdr.split("(10", 1)[1].strip()
                p0 = inner.find("(")
                p1 = inner.find(")", p0 + 1)
                meta = inner[p0 + 1 : p1].split()
                start = _parse_int_auto(meta[1]) if len(meta) > 1 else -1
                end   = _parse_int_auto(meta[2]) if len(meta) > 2 else -1
            except Exception:
                start, end = -1, -1

            n_expected = (end - start + 1) if (start > 0 and end >= start) else None
            coords: List[List[float]] = []
            j = i + 1
            while j < len(lines):
                s = _strip_comments(lines[j]).strip()
                if not s:
                    j += 1
                    continue
                if s.startswith(")"):
                    break
                parts = s.replace("(", " ").replace(")", " ").split()
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except Exception:
                        pass
                for k in range(0, len(vals), 3):
                    if k + 2 < len(vals):
                        coords.append(vals[k:k+3])
                j += 1
                if n_expected is not None and len(coords) >= n_expected:
                    while j < len(lines):
                        s2 = _strip_comments(lines[j]).strip()
                        if s2.startswith(")"):
                            break
                        j += 1
                    break

            if start > 0 and end >= start and coords:
                nn = min(len(coords), end - start + 1)
                for idx in range(nn):
                    nid = start + idx
                    nodes[nid] = np.array(coords[idx], dtype=np.float64) * scale

            i = j
        i += 1

    if not nodes:
        raise RuntimeError(f"No nodes parsed from {msh_path}")

    max_id = max(nodes.keys())
    nodes_xyz = np.zeros((max_id, 3), dtype=np.float64)
    for nid, xyz in nodes.items():
        nodes_xyz[nid - 1, :] = xyz  # to 0-based

    # ---- faces (wall zone) ----
    faces: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(lines):
        raw = _strip_comments(lines[i]).strip()
        if raw.startswith("(13"):
            hdr = raw
            while hdr.count("(") > hdr.count(")") and i + 1 < len(lines):
                i += 1
                hdr += " " + _strip_comments(lines[i]).strip()

            try:
                inner = hdr.split("(13", 1)[1].strip()
                p0 = inner.find("(")
                p1 = inner.find(")", p0 + 1)
                meta = inner[p0 + 1 : p1].split()
                zone = _parse_int_auto(meta[0]) if len(meta) > 0 else -1
            except Exception:
                zone = -1

            j = i + 1
            if zone == wall_zone:
                while j < len(lines):
                    s = _strip_comments(lines[j]).strip()
                    if not s:
                        j += 1
                        continue
                    if s.startswith(")"):
                        break
                    parts = s.replace("(", " ").replace(")", " ").split()
                    ints = []
                    for p in parts:
                        try:
                            ints.append(_parse_int_auto(p))
                        except Exception:
                            pass
                    if not ints:
                        j += 1
                        continue

                    if ints[0] in (3, 4):
                        n = ints[0]
                        verts = ints[1 : 1 + n]
                    else:
                        n = 3
                        verts = ints[0:3]

                    if len(verts) >= 3:
                        v0, v1, v2 = verts[0] - 1, verts[1] - 1, verts[2] - 1
                        faces.append((v0, v1, v2))
                        if n == 4 and len(verts) == 4:
                            v3 = verts[3] - 1
                            faces.append((v0, v2, v3))
                    j += 1
            else:
                while j < len(lines):
                    s = _strip_comments(lines[j]).strip()
                    if s.startswith(")"):
                        break
                    j += 1

            i = j
        i += 1

    if not faces:
        raise RuntimeError(
            f"No faces found for wall_zone={wall_zone} in {msh_path}. "
            f"Try a different --wall_zone, or add zone listing support."
        )

    return nodes_xyz, np.array(faces, dtype=np.int64)


def build_surface_graph(nodes_xyz: np.ndarray, faces_global: np.ndarray) -> Data:
    """
    Build wall-only graph with:
      pos [Nw,3], face [3,F], edge_index [2,E], x [Nw,7], global_node_ids [Nw]
    """
    wall_nodes = np.unique(faces_global.reshape(-1))
    wall_nodes_sorted = np.sort(wall_nodes)
    global_to_local = {int(g): i for i, g in enumerate(wall_nodes_sorted)}

    pos = nodes_xyz[wall_nodes_sorted, :].astype(np.float32)

    local_faces = np.vectorize(lambda g: global_to_local[int(g)])(faces_global).astype(np.int64)
    face = torch.from_numpy(local_faces.T.copy())  # [3,F]

    # edges from triangles
    e_set = set()
    for tri in local_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            if u == v:
                continue
            e_set.add((u, v) if u < v else (v, u))
    edges = np.array(list(e_set), dtype=np.int64)
    if edges.size == 0:
        raise RuntimeError("No edges constructed from faces.")
    edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T  # undirected -> directed

    # vertex normals
    vnorm = np.zeros_like(pos, dtype=np.float64)
    p = pos.astype(np.float64)
    for tri in local_faces:
        i0, i1, i2 = tri
        v0, v1, v2 = p[i0], p[i1], p[i2]
        n = np.cross(v1 - v0, v2 - v0)
        vnorm[i0] += n
        vnorm[i1] += n
        vnorm[i2] += n
    nrm = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vnorm = (vnorm / np.maximum(nrm, 1e-12)).astype(np.float32)

    # bbox normalized xyz
    mn = pos.min(axis=0, keepdims=True)
    mx = pos.max(axis=0, keepdims=True)
    span = np.maximum(mx - mn, 1e-12)
    xyz_norm = ((pos - mn) / span).astype(np.float32)

    is_wall = np.ones((pos.shape[0], 1), dtype=np.float32)
    x = np.concatenate([xyz_norm, vnorm, is_wall], axis=1)  # [Nw,7]

    data = Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index.astype(np.int64)),
        face=face,
        global_node_ids=torch.from_numpy(wall_nodes_sorted.astype(np.int64)),
    )
    return data


# ======================================================================================
# Optional: Fluent wall CSV loading + nearest-neighbor mapping (for validation)
# ======================================================================================

def read_wall_csv_xyz_wss(wall_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads Fluent wall_data_Re*.csv exported as whitespace-delimited table.
    Returns:
      xyz: [M,3]
      wss_vec: [M,3]  (x-wall-shear, y-wall-shear, z-wall-shear)
    """
    import pandas as pd

    df = pd.read_csv(wall_csv, sep=r"\s+", engine="python", comment=None)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # robust column lookup
    def pick(name_opts):
        for n in name_opts:
            if n in df.columns:
                return n
        raise KeyError(f"Missing column. Looked for: {name_opts}. Found: {df.columns.tolist()}")

    cx = pick(["x-coordinate", "X-Coordinate", "x", "X"])
    cy = pick(["y-coordinate", "Y-Coordinate", "y", "Y"])
    cz = pick(["z-coordinate", "Z-Coordinate", "z", "Z"])

    wx = pick(["x-wall-shear", "X-Wall-Shear", "x_wall_shear", "x-wss"])
    wy = pick(["y-wall-shear", "Y-Wall-Shear", "y_wall_shear", "y-wss"])
    wz = pick(["z-wall-shear", "Z-Wall-Shear", "z_wall_shear", "z-wss"])

    xyz = df[[cx, cy, cz]].to_numpy(dtype=np.float64)
    wss = df[[wx, wy, wz]].to_numpy(dtype=np.float64)
    return xyz, wss


def maybe_rescale_csv_xyz_to_match_graph(csv_xyz: np.ndarray, graph_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    If CSV is in mm and graph is in m (or vice versa), rescale csv_xyz.
    Uses bbox diagonal heuristic.
    """
    def diag(x):
        mn = x.min(axis=0)
        mx = x.max(axis=0)
        return float(np.linalg.norm(mx - mn))

    d_csv = diag(csv_xyz)
    d_g   = diag(graph_xyz)

    if d_csv < 1e-12 or d_g < 1e-12:
        return csv_xyz, 1.0

    ratio = d_csv / d_g

    # common mismatch: 1000x (mm vs m)
    if ratio > 200 and ratio < 5000:
        return csv_xyz / ratio, 1.0 / ratio
    if ratio < 0.005 and ratio > 1e-6:
        return csv_xyz / ratio, 1.0 / ratio

    return csv_xyz, 1.0


def nn_map_csv_to_nodes(csv_xyz: np.ndarray, csv_wss: np.ndarray, node_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map CSV sample points to nearest wall nodes.
    Returns:
      wss_node_avg: [N,3]
      counts: [N] number of hits per node
    """
    N = node_xyz.shape[0]
    wss_sum = np.zeros((N, 3), dtype=np.float64)
    counts = np.zeros((N,), dtype=np.int64)

    # KDTree if available
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(node_xyz)
        d, idx = tree.query(csv_xyz, k=1)
    except Exception:
        # brute force chunked
        idx = np.empty((csv_xyz.shape[0],), dtype=np.int64)
        chunk = 2000
        for i0 in range(0, csv_xyz.shape[0], chunk):
            pts = csv_xyz[i0:i0+chunk]
            # (chunk,N,3) diff -> norms
            diff = pts[:, None, :] - node_xyz[None, :, :]
            dist2 = np.sum(diff*diff, axis=2)
            idx[i0:i0+chunk] = np.argmin(dist2, axis=1)

    for k, ni in enumerate(idx):
        wss_sum[ni] += csv_wss[k]
        counts[ni] += 1

    wss_avg = wss_sum / np.maximum(counts[:, None], 1)
    return wss_avg, counts


# ======================================================================================
# Export helpers: CSV + VTP (meshio)
# ======================================================================================

def export_csv(out_csv: Path, pos: np.ndarray, wss_vec: np.ndarray, extra: Optional[Dict[str, np.ndarray]] = None):
    """
    Writes x,y,z,WSSx,WSSy,WSSz,WSSmag (+ optional columns from extra dict).
    """
    wss_mag = np.linalg.norm(wss_vec, axis=1)
    header_cols = ["x", "y", "z", "WSSx", "WSSy", "WSSz", "WSSmag"]
    data_cols = [pos[:,0], pos[:,1], pos[:,2], wss_vec[:,0], wss_vec[:,1], wss_vec[:,2], wss_mag]

    if extra:
        for k, arr in extra.items():
            if arr.ndim == 1:
                header_cols.append(k)
                data_cols.append(arr)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                header_cols += [f"{k}x", f"{k}y", f"{k}z"]
                data_cols += [arr[:,0], arr[:,1], arr[:,2]]
            else:
                raise ValueError(f"Unsupported extra array shape for {k}: {arr.shape}")

    M = np.column_stack(data_cols)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, M, delimiter=",", header=",".join(header_cols), comments="")
    print(f"[OK] Wrote CSV: {out_csv}")


def export_vtp_meshio(out_vtp: Path, pos: np.ndarray, faces_local: np.ndarray, point_data: Dict[str, np.ndarray]):
    """
    Export a surface mesh as .vtp with point_data arrays.
    Requires meshio.
    faces_local: [F,3] triangles in local node indexing.
    """
    try:
        import meshio
    except Exception as e:
        print("[WARN] meshio not installed; skipping VTP export. Install via: pip install meshio")
        return

    cells = [("triangle", faces_local.astype(np.int32))]
    mesh = meshio.Mesh(points=pos.astype(np.float64), cells=cells, point_data=point_data)
    out_vtp.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(out_vtp)
    print(f"[OK] Wrote VTP: {out_vtp}")


# ======================================================================================
# Main inference
# ======================================================================================

def main():
    ap = argparse.ArgumentParser(description="Inference: new Fluent .msh -> WSS prediction -> exports")
    ap.add_argument("--ckpt", required=True, type=str, help="Path to best_model.pt (checkpoint from training.py)")
    ap.add_argument("--msh", required=True, type=str, help="Path to NEW Fluent legacy ASCII .msh (unseen geometry)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--wall_zone", default=19, type=int, help="Wall face zone id (default 19)")
    ap.add_argument("--re", default=None, type=float, help="Reynolds number for conditioning (required if ckpt expects it)")
    ap.add_argument("--device", default=None, type=str, help="cpu or cuda (default auto)")
    ap.add_argument("--export_vtp", action="store_true", help="Export .vtp surface with point data arrays")
    ap.add_argument("--export_csv", action="store_true", help="Export CSV aligned to wall nodes")
    ap.add_argument("--save_pred_pt", action="store_true", help="Save a .pt with graph + pred arrays")
    ap.add_argument("--cfd_wall_csv", default=None, type=str, help="Optional Fluent wall_data_Re*.csv for validation")
    ap.add_argument("--name", default=None, type=str, help="Output name stem (default derived from msh filename)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    msh_path = Path(args.msh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.name if args.name else msh_path.stem

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ load checkpoint ------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    out_dim = int(ckpt.get("out_dim", 3))
    hidden_dim = int(ckpt["hidden_dim"])
    num_layers = int(ckpt["num_layers"])
    dropout = float(ckpt["dropout"])
    append_re = bool(ckpt.get("append_re_to_x", False))
    normalize_y = bool(ckpt.get("normalize_y", False))

    re_mu = float(ckpt.get("re_mu", 0.0))
    re_sd = float(ckpt.get("re_sd", 1.0))
    y_mu = ckpt.get("y_mu", torch.zeros((out_dim,), dtype=torch.float32)).float()
    y_sd = ckpt.get("y_sd", torch.ones((out_dim,), dtype=torch.float32)).float()

    # ------------------ build graph from msh ------------------
    nodes_xyz, faces_global = parse_fluent_ascii_msh_wall(msh_path, wall_zone=int(args.wall_zone))
    data = build_surface_graph(nodes_xyz, faces_global)

    # store Re if provided
    if args.re is not None:
        data.re = torch.tensor(float(args.re), dtype=torch.float32)

    # enforce conditioning consistency
    if append_re:
        if args.re is None and not hasattr(data, "re"):
            raise ValueError(
                "Checkpoint expects Re appended to node features (append_re_to_x=True), but --re was not provided.\n"
                "Run again with: --re <value> (e.g., --re 400)"
            )
        re_val = float(data.re.item()) if hasattr(data, "re") else float(args.re)
        re_norm = (re_val - re_mu) / (re_sd if re_sd > 1e-12 else 1.0)
        re_feat = torch.full((data.num_nodes, 1), float(re_norm), dtype=torch.float32)
        data.x = torch.cat([data.x.float(), re_feat], dim=1)

    # final sanity check on input dim
    if data.x.shape[1] != in_dim:
        raise RuntimeError(
            f"Feature dim mismatch: graph has x.shape[1]={data.x.shape[1]} but checkpoint expects in_dim={in_dim}.\n"
            f"Checkpoint append_re_to_x={append_re}. You likely forgot --re, or trained with different features."
        )

    # ------------------ load model ------------------
    model = WSSNet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # ------------------ inference ------------------
    data = data.to(device)
    with torch.no_grad():
        pred = model(data).detach().cpu()  # [N,3] in normalized space if normalize_y=True

    # denormalize if needed
    if normalize_y:
        pred = pred * y_sd.view(1, -1) + y_mu.view(1, -1)

    pred_np = pred.numpy().astype(np.float64)

    # ------------------ optional validation vs Fluent CSV ------------------
    extra_csv_cols: Dict[str, np.ndarray] = {}
    point_data: Dict[str, np.ndarray] = {}

    pos_np = data.pos.detach().cpu().numpy().astype(np.float64)
    faces_local = data.face.detach().cpu().numpy().T.astype(np.int64)  # [F,3] local

    pred_mag = np.linalg.norm(pred_np, axis=1)
    point_data["WSS_vec"] = pred_np
    point_data["WSS_mag"] = pred_mag

    metrics = {}
    if args.cfd_wall_csv:
        wall_csv = Path(args.cfd_wall_csv)
        csv_xyz, csv_wss = read_wall_csv_xyz_wss(wall_csv)
        csv_xyz2, scale_used = maybe_rescale_csv_xyz_to_match_graph(csv_xyz, pos_np)
        cfd_node_wss, hit_counts = nn_map_csv_to_nodes(csv_xyz2, csv_wss, pos_np)

        # errors
        err_vec = pred_np - cfd_node_wss
        err_mag = np.linalg.norm(err_vec, axis=1)
        cfd_mag = np.linalg.norm(cfd_node_wss, axis=1)

        # metrics (mask very low WSS to avoid exploding relative error)
        eps = 1e-6
        mask = cfd_mag > (1e-4)  # adjust if needed
        rel_err = np.zeros_like(cfd_mag)
        rel_err[mask] = err_mag[mask] / (cfd_mag[mask] + eps)

        # correlation / R^2 on magnitudes
        if np.any(mask):
            x = cfd_mag[mask]
            y = pred_mag[mask]
            # Pearson r
            r = np.corrcoef(x, y)[0, 1] if x.size > 3 else float("nan")
            # R^2
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((x - x.mean()) ** 2) + 1e-12)
            r2 = 1.0 - ss_res / ss_tot
        else:
            r, r2 = float("nan"), float("nan")

        metrics = {
            "csv_scale_applied_to_xyz": float(scale_used),
            "nn_hits_nonzero_frac": float(np.mean(hit_counts > 0)),
            "mse_vec": float(np.mean(err_vec ** 2)),
            "mae_mag": float(np.mean(err_mag)),
            "mae_mag_masked": float(np.mean(err_mag[mask])) if np.any(mask) else float("nan"),
            "rel_err_mean_masked": float(np.mean(rel_err[mask])) if np.any(mask) else float("nan"),
            "pearson_r_mag_masked": float(r),
            "r2_mag_masked": float(r2),
        }

        print("\n" + "=" * 88)
        print("VALIDATION vs Fluent CSV (nearest-neighbor mapped to nodes)")
        print("=" * 88)
        for k, v in metrics.items():
            print(f"{k:28s}: {v}")

        # add to exports
        point_data["CFD_WSS_vec"] = cfd_node_wss
        point_data["CFD_WSS_mag"] = cfd_mag
        point_data["ERR_vec"] = err_vec
        point_data["ERR_mag"] = err_mag
        point_data["NN_hit_count"] = hit_counts.astype(np.float64)

        extra_csv_cols["CFD_WSS"] = cfd_node_wss
        extra_csv_cols["CFD_WSSmag"] = cfd_mag
        extra_csv_cols["ERR"] = err_vec
        extra_csv_cols["ERRmag"] = err_mag
        extra_csv_cols["NN_hit_count"] = hit_counts.astype(np.float64)

    # ------------------ exports ------------------
    if args.export_csv:
        out_csv = out_dir / f"{stem}_pred.csv"
        export_csv(out_csv, pos_np, pred_np, extra=extra_csv_cols)

    if args.export_vtp:
        out_vtp = out_dir / f"{stem}_pred.vtp"
        export_vtp_meshio(out_vtp, pos_np, faces_local, point_data=point_data)

    if args.save_pred_pt:
        # Save a compact artifact: original graph (cpu) + predictions as tensors
        out_pt = out_dir / f"{stem}_pred.pt"
        data_cpu = data.detach().cpu()
        data_cpu.pred = torch.from_numpy(pred_np.astype(np.float32))
        data_cpu.pred_mag = torch.from_numpy(pred_mag.astype(np.float32))
        if metrics:
            data_cpu.metrics = metrics
        torch.save(data_cpu, out_pt)
        print(f"[OK] Wrote PT: {out_pt}")

    # write a metadata json for auditability
    meta = {
        "msh": str(msh_path),
        "ckpt": str(ckpt_path),
        "device": device,
        "wall_zone": int(args.wall_zone),
        "append_re_to_x": append_re,
        "normalize_y": normalize_y,
        "re": float(args.re) if args.re is not None else None,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_faces": int(data.face.shape[1]),
        "validation_wall_csv": str(args.cfd_wall_csv) if args.cfd_wall_csv else None,
        "validation_metrics": metrics if metrics else None,
    }
    with open(out_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote meta: {out_dir / f'{stem}_meta.json'}")

    print("\nDONE.")


if __name__ == "__main__":
    main()