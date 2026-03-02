#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INFERENCE — Fluent legacy ASCII .msh -> wall graph -> WSS prediction -> exports (+ optional CSV validation)

Compatible with checkpoint format and graph semantics from upgradedtraining.py.

PHASE 1 IMPROVEMENTS (contract + validation hardening, minimal-invasive):
- Contract check mode (feature schema/order/stats, normals stats, graph stats, wall zones, sample rows)
- Normal handling controls (raw / flip / auto_centroid) + debug flipped-normal validation
- Vector convention sanity tests (permutations/sign flips) during validation
- Entity-aware validation (node_nn vs face_nn, auto detect)
- Patch-aware validation summaries and manual include/exclude for validation
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# =============================================================================
# Logging / serialization helpers
# =============================================================================

class Logger:
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else None
        self._fh = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open("w", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(x: Any):
        if isinstance(x, Path):
            return str(x)
        if torch.is_tensor(x):
            if x.ndim == 0:
                return x.item()
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (set, tuple)):
            return list(x)
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =============================================================================
# Model (MUST match training)
# =============================================================================

class GraphMP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([GraphMP(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)])
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
            h = h + layer(h, ei)
        return self.dec(h)


# =============================================================================
# Fluent legacy ASCII .msh parsing (adapted from upgradedtraining.py)
# =============================================================================

def _strip_parens(s: str) -> str:
    return s.replace("(", " ").replace(")", " ").strip()


def _balance_count(s: str) -> int:
    return s.count("(") - s.count(")")


def parse_int_token(tok: str, int_mode: str = "hex") -> int:
    t = tok.strip()
    if t == "":
        raise ValueError("Empty integer token")

    sign = 1
    if t[0] == "-":
        sign = -1
        t = t[1:]

    if int_mode == "hex":
        return sign * int(t, 16)
    if int_mode == "dec":
        return sign * int(t, 10)

    v_hex = sign * int(t, 16)
    v_dec = sign * int(t, 10)
    if abs(v_hex) > 100 * max(1, abs(v_dec)) and abs(v_dec) < 10_000_000:
        return v_dec
    return v_hex


@dataclass
class FaceBlockMeta:
    zone_id: int
    first: int
    last: int
    bc_type: int
    face_type: int
    header_line: int

    @property
    def n_faces(self) -> int:
        return max(0, self.last - self.first + 1)

    @property
    def bc_type_base(self) -> int:
        return int(self.bc_type % 1000)


@dataclass
class NodeDeclMeta:
    first: int
    last: int

    @property
    def n_nodes(self) -> int:
        return max(0, self.last - self.first + 1)


@dataclass
class MeshHeaderScan:
    msh_path: str
    int_mode: str
    scaling_factor: float
    node_decl: Optional[NodeDeclMeta]
    face_blocks: List[FaceBlockMeta]


def scan_fluent_msh_headers(msh_path: Path, int_mode: str = "hex") -> MeshHeaderScan:
    scaling = 1.0
    node_decl: Optional[NodeDeclMeta] = None
    face_blocks: List[FaceBlockMeta] = []

    with Path(msh_path).open("r", errors="ignore") as f:
        ln = 0
        for line in f:
            ln += 1
            s = line.strip()

            if "meshing-to-solver-scaling-factor" in s:
                toks = _strip_parens(s).split()
                try:
                    scaling = float(toks[-1])
                except Exception:
                    pass

            if s.startswith("(10"):
                hdr = s
                bal = _balance_count(hdr)
                while bal > 0:
                    nxt = f.readline()
                    if not nxt:
                        break
                    ln += 1
                    s2 = nxt.strip()
                    hdr += " " + s2
                    bal += _balance_count(s2)

                toks = _strip_parens(hdr).split()
                if len(toks) >= 5 and toks[0] == "10":
                    try:
                        zone = parse_int_token(toks[1], int_mode=int_mode)
                    except Exception:
                        zone = None
                    if zone == 0:
                        try:
                            first = parse_int_token(toks[2], int_mode=int_mode)
                            last = parse_int_token(toks[3], int_mode=int_mode)
                            node_decl = NodeDeclMeta(first=first, last=last)
                        except Exception:
                            pass

            if s.startswith("(13"):
                hdr = s
                bal = _balance_count(hdr)
                while bal > 0:
                    nxt = f.readline()
                    if not nxt:
                        break
                    ln += 1
                    s2 = nxt.strip()
                    hdr += " " + s2
                    bal += _balance_count(s2)

                toks = _strip_parens(hdr).split()
                if len(toks) >= 6 and toks[0] == "13":
                    try:
                        zone = parse_int_token(toks[1], int_mode=int_mode)
                        first = parse_int_token(toks[2], int_mode=int_mode)
                        last = parse_int_token(toks[3], int_mode=int_mode)
                        bc = parse_int_token(toks[4], int_mode=int_mode)
                        ft = parse_int_token(toks[5], int_mode=int_mode)
                    except Exception:
                        continue
                    if zone != 0:
                        face_blocks.append(FaceBlockMeta(zone, first, last, bc, ft, ln))

    return MeshHeaderScan(
        msh_path=str(msh_path),
        int_mode=str(int_mode),
        scaling_factor=float(scaling),
        node_decl=node_decl,
        face_blocks=face_blocks,
    )


def parse_fluent_msh_nodes_and_faces(
    msh_path: Path,
    wanted_face_zones: Optional[Sequence[int]],
    int_mode: str = "hex",
    force_dim3: bool = True,
    max_nodes_guard: int = 20_000_000,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], MeshHeaderScan]:
    msh_path = Path(msh_path)
    scan = scan_fluent_msh_headers(msh_path, int_mode=int_mode)

    if scan.node_decl is None:
        raise RuntimeError(
            f"Could not find node declaration section (10 (0 ...)) in mesh: {msh_path}. "
            f"Ensure this is a Fluent legacy ASCII .msh."
        )

    n_nodes = scan.node_decl.n_nodes
    if n_nodes <= 0 or n_nodes > max_nodes_guard:
        raise RuntimeError(
            f"Unreasonable node count parsed ({n_nodes}). Try --int_mode dec if this mesh uses decimal indices."
        )

    node_first = scan.node_decl.first
    node_last = scan.node_decl.last
    nodes_xyz = np.zeros((n_nodes, 3), dtype=np.float32)

    wanted_set = None if wanted_face_zones is None else set(int(z) for z in wanted_face_zones)
    faces_by_zone: Dict[int, List[List[int]]] = {}

    def node_to_local(nid_1based: int) -> int:
        idx = nid_1based - node_first
        if idx < 0 or idx >= n_nodes:
            raise IndexError(f"Node id {nid_1based} out of declared bounds [{node_first},{node_last}]")
        return idx

    def _read_header_line_only(fh, first_line: str, line_no: int) -> Tuple[str, int]:
        hdr = first_line.strip()
        toks_try = _strip_parens(hdr).split()
        if len(toks_try) >= 6:
            return hdr, line_no
        for _ in range(5):
            nxt = fh.readline()
            if not nxt:
                break
            line_no += 1
            s2 = nxt.strip()
            if s2 and (s2[0].isdigit() or s2[0] in "+-."):
                hdr += " " + s2
                break
            hdr += " " + s2
            toks_try = _strip_parens(hdr).split()
            if len(toks_try) >= 6:
                break
        return hdr, line_no

    with msh_path.open("r", errors="ignore") as f:
        ln = 0
        while True:
            line = f.readline()
            if not line:
                break
            ln += 1
            s = line.strip()

            if s.startswith("(10"):
                hdr, ln = _read_header_line_only(f, s, ln)
                toks = _strip_parens(hdr).split()
                if len(toks) < 6 or toks[0] != "10":
                    continue
                try:
                    zone = parse_int_token(toks[1], int_mode=int_mode)
                except Exception:
                    continue
                if zone == 0:
                    continue
                try:
                    start = parse_int_token(toks[2], int_mode=int_mode)
                    end = parse_int_token(toks[3], int_mode=int_mode)
                    dim = parse_int_token(toks[5], int_mode=int_mode)
                except Exception:
                    continue
                if force_dim3 and dim != 3:
                    raise RuntimeError(f"Expected 3D nodes (dim=3) but got dim={dim} in node zone {zone} at line {ln}")
                expected = end - start + 1
                if expected <= 0:
                    continue

                nid = start
                while nid <= end:
                    body_line = f.readline()
                    if not body_line:
                        break
                    ln += 1
                    raw = body_line.strip()
                    if raw.startswith(")"):
                        break
                    t = _strip_parens(raw)
                    if t == "":
                        continue
                    vals: List[float] = []
                    for a in t.split():
                        try:
                            vals.append(float(a))
                        except Exception:
                            pass
                    if len(vals) == 0:
                        continue
                    if len(vals) % dim != 0:
                        raise RuntimeError(f"Node coord line at {ln} not multiple of {dim} floats.")
                    for k in range(0, len(vals), dim):
                        if nid > end:
                            break
                        nodes_xyz[node_to_local(nid), :] = (vals[k], vals[k + 1], vals[k + 2])
                        nid += 1

                if nid <= end:
                    raise RuntimeError(
                        f"Node section for zone {zone} ended before all nodes were read "
                        f"(read up to node-id {nid-1}, expected end {end}). "
                        f"This usually indicates wrong integer parsing (--int_mode) or malformed/unsupported .msh."
                    )
                continue

            if s.startswith("(13"):
                hdr, ln = _read_header_line_only(f, s, ln)
                toks = _strip_parens(hdr).split()
                if len(toks) < 6 or toks[0] != "13":
                    continue
                try:
                    zone = parse_int_token(toks[1], int_mode=int_mode)
                except Exception:
                    continue
                if zone == 0:
                    continue

                if wanted_set is not None and zone not in wanted_set:
                    while True:
                        body_line = f.readline()
                        if not body_line:
                            break
                        ln += 1
                        if body_line.strip().startswith(")"):
                            break
                    continue

                try:
                    start = parse_int_token(toks[2], int_mode=int_mode)
                    end = parse_int_token(toks[3], int_mode=int_mode)
                    expected_faces = (end - start + 1) if (start > 0 and end >= start) else None
                    face_type = parse_int_token(toks[5], int_mode=int_mode)
                except Exception:
                    expected_faces = None
                    face_type = -1

                faces_by_zone.setdefault(zone, [])
                src_faces_read = 0

                while True:
                    body_line = f.readline()
                    if not body_line:
                        break
                    ln += 1
                    raw = body_line.strip()
                    if raw.startswith(")"):
                        break
                    t = _strip_parens(raw)
                    if t == "":
                        continue
                    ints: List[int] = []
                    for a in t.split():
                        try:
                            ints.append(parse_int_token(a, int_mode=int_mode))
                        except Exception:
                            pass
                    if not ints:
                        continue

                    if ints[0] in (3, 4) and (face_type in (0, 5) or len(ints) >= (1 + ints[0] + 2)):
                        nv = ints[0]
                        verts = ints[1 : 1 + nv]
                    elif face_type == 4 and len(ints) >= 6:
                        nv = 4
                        verts = ints[0:4]
                    else:
                        nv = 3
                        verts = ints[0:3]

                    if len(verts) < 3:
                        continue
                    try:
                        vid = [node_to_local(v) for v in verts]
                    except Exception:
                        continue

                    if nv == 3:
                        faces_by_zone[zone].append([vid[0], vid[1], vid[2]])
                    elif nv == 4:
                        faces_by_zone[zone].append([vid[0], vid[1], vid[2]])
                        faces_by_zone[zone].append([vid[0], vid[2], vid[3]])
                    else:
                        v0 = vid[0]
                        for k in range(1, nv - 1):
                            faces_by_zone[zone].append([v0, vid[k], vid[k + 1]])

                    src_faces_read += 1
                    if expected_faces is not None and src_faces_read >= expected_faces:
                        while True:
                            maybe = f.readline()
                            if not maybe:
                                break
                            ln += 1
                            if maybe.strip().startswith(")"):
                                break
                        break

                if expected_faces is not None and src_faces_read < expected_faces:
                    raise RuntimeError(
                        f"Face section for zone {zone} ended early (read {src_faces_read} faces, expected {expected_faces}). "
                        f"This usually indicates wrong integer parsing (--int_mode) or malformed/unsupported .msh."
                    )
                continue

    if scan.scaling_factor is not None and abs(scan.scaling_factor - 1.0) > 1e-15:
        nodes_xyz = (nodes_xyz.astype(np.float64) * float(scan.scaling_factor)).astype(np.float32)

    faces_out = {z: np.array(f, dtype=np.int64) for z, f in faces_by_zone.items()}
    return nodes_xyz, faces_out, scan


# =============================================================================
# Wall zone selection (same logic as training)
# =============================================================================

BC_TYPE_MAP = {
    2: "interior",
    3: "wall",
    4: "pressure-inlet",
    5: "pressure-outlet",
    7: "symmetry",
    8: "periodic-shadow",
    9: "pressure-far-field",
    10: "velocity-inlet",
    12: "periodic",
    14: "fan/porous/radiator",
    20: "mass-flow-inlet",
    24: "interface",
    31: "parent(hanging)",
    36: "outflow",
    37: "axis",
}


@dataclass
class WallSelection:
    mode: str
    manual_zone_ids: Optional[List[int]] = None
    auto_min_faces_frac: float = 0.0


def select_wall_zones(scan: MeshHeaderScan, sel: WallSelection, logger: Logger) -> List[int]:
    if sel.mode == "manual":
        if not sel.manual_zone_ids:
            raise ValueError("Wall selection mode is manual but no --wall_zone provided.")
        zones = [int(z) for z in sel.manual_zone_ids]
        logger.log(f"[WALL] Manual wall zones: {zones}")
        return zones

    wall_blocks = [b for b in scan.face_blocks if b.bc_type_base == 3]
    if not wall_blocks:
        bc_present = sorted(set(b.bc_type_base for b in scan.face_blocks))
        logger.log(f"[WALL][ERROR] No bc-type wall zones found (bc%1000==3). bc-types present: "
                   f"{[(bc, BC_TYPE_MAP.get(bc,'unknown')) for bc in bc_present]}")
        logger.log("[WALL][HINT] Try --wall_mode manual --wall_zone <id> and/or --int_mode dec.")
        return []

    total_faces = sum(b.n_faces for b in wall_blocks)
    keep: List[int] = []
    for b in wall_blocks:
        frac = b.n_faces / max(total_faces, 1)
        if frac + 1e-12 >= float(sel.auto_min_faces_frac):
            keep.append(b.zone_id)

    keep = sorted(set(keep))
    logger.log(f"[WALL] Auto wall zones by bc-type: {keep}")
    for b in sorted(wall_blocks, key=lambda x: x.n_faces, reverse=True):
        frac = b.n_faces / max(total_faces, 1)
        logger.log(
            f"       zone={b.zone_id:>6d} bc={b.bc_type} (base {b.bc_type_base}:{BC_TYPE_MAP.get(b.bc_type_base,'?')}) "
            f"face_type={b.face_type} n_faces={b.n_faces} frac={frac:.3f}"
        )
    dropped = sorted(set(bb.zone_id for bb in wall_blocks) - set(keep))
    if dropped:
        logger.log(f"[WALL][WARN] Dropped small wall zones (auto_min_faces_frac={sel.auto_min_faces_frac}): {dropped}")
    return keep


# =============================================================================
# Graph construction (same semantics as training) + contract helpers
# =============================================================================

BASE_FEATURE_NAMES = ["x_norm", "y_norm", "z_norm", "nx", "ny", "nz", "is_wall"]


def bbox_diag(xyz: np.ndarray) -> float:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def normalize_bbox(xyz: np.ndarray) -> np.ndarray:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    span = np.where((mx - mn) > 1e-12, (mx - mn), 1.0)
    return ((xyz - mn) / span).astype(np.float32)


def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    norm = np.linalg.norm(normals, axis=1)
    norm = np.where(norm > 1e-12, norm, 1.0)
    normals = (normals.T / norm).T
    return normals.astype(np.float32)


def apply_normal_mode(points: np.ndarray, normals: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Phase 1 minimal canonicalization:
    - raw: no change
    - flip: n -> -n
    - auto_centroid: choose global sign so normals roughly point away from centroid
                     (heuristic; not perfect for all vessels, but useful as consistency check)
    """
    normals = normals.astype(np.float32, copy=True)
    info: Dict[str, Any] = {"mode": mode, "applied_global_flip": False}

    if mode == "raw":
        return normals, info

    if mode == "flip":
        normals *= -1.0
        info["applied_global_flip"] = True
        return normals, info

    if mode == "auto_centroid":
        ctr = points.mean(axis=0, keepdims=True).astype(np.float32)
        radial = points.astype(np.float32) - ctr
        radial_norm = np.linalg.norm(radial, axis=1)
        valid = radial_norm > 1e-12
        if np.any(valid):
            dots = np.sum(normals[valid] * radial[valid], axis=1)
            mean_dot = float(np.mean(dots))
            frac_pos = float(np.mean(dots > 0))
            frac_neg = float(np.mean(dots < 0))
            info["centroid_dot_mean_before"] = mean_dot
            info["centroid_dot_frac_pos_before"] = frac_pos
            info["centroid_dot_frac_neg_before"] = frac_neg
            # If predominantly inward relative to centroid radial direction, flip all
            if mean_dot < 0.0:
                normals *= -1.0
                info["applied_global_flip"] = True
                info["centroid_dot_mean_after"] = float(-mean_dot)
            else:
                info["centroid_dot_mean_after"] = mean_dot
        return normals, info

    raise ValueError(f"Unknown normal mode: {mode}")


def compute_node_patch_membership(
    n_nodes: int,
    faces_local: np.ndarray,
    faces_zone_ids: np.ndarray,
) -> np.ndarray:
    """
    Dominant patch zone per node (majority by incident face count).
    Nodes shared across patches get the dominant one.
    """
    zone_vote: Dict[int, np.ndarray] = {}
    for zi in np.unique(faces_zone_ids):
        mask = (faces_zone_ids == zi)
        if not np.any(mask):
            continue
        arr = np.zeros((n_nodes,), dtype=np.int64)
        f = faces_local[mask]
        np.add.at(arr, f[:, 0], 1)
        np.add.at(arr, f[:, 1], 1)
        np.add.at(arr, f[:, 2], 1)
        zone_vote[int(zi)] = arr

    node_zone = np.full((n_nodes,), -1, dtype=np.int64)
    if not zone_vote:
        return node_zone
    zones = sorted(zone_vote.keys())
    votes = np.stack([zone_vote[z] for z in zones], axis=1)  # [N, Z]
    best_idx = np.argmax(votes, axis=1)
    best_val = votes[np.arange(n_nodes), best_idx]
    node_zone = np.where(best_val > 0, np.array(zones, dtype=np.int64)[best_idx], -1)
    return node_zone


def build_feature_matrix(
    wall_points: np.ndarray,
    faces_local: np.ndarray,
    normal_mode: str = "raw",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    xyz_norm = normalize_bbox(wall_points)
    normals_raw = compute_vertex_normals(wall_points, faces_local.astype(np.int64))
    normals, normal_mode_info = apply_normal_mode(wall_points, normals_raw, normal_mode)
    is_wall = np.ones((wall_points.shape[0], 1), dtype=np.float32)
    x = np.concatenate([xyz_norm, normals, is_wall], axis=1).astype(np.float32)

    nmag_raw = np.linalg.norm(normals_raw, axis=1)
    nmag = np.linalg.norm(normals, axis=1)
    near_zero_thr = 1e-6
    normal_stats = {
        "normal_mode": str(normal_mode),
        "raw_norm_mean": float(np.mean(nmag_raw)),
        "raw_norm_std": float(np.std(nmag_raw)),
        "raw_norm_min": float(np.min(nmag_raw)),
        "raw_norm_max": float(np.max(nmag_raw)),
        "norm_mean": float(np.mean(nmag)),
        "norm_std": float(np.std(nmag)),
        "norm_min": float(np.min(nmag)),
        "norm_max": float(np.max(nmag)),
        "near_zero_frac": float(np.mean(nmag < near_zero_thr)),
        "near_zero_count": int(np.sum(nmag < near_zero_thr)),
        "mode_info": normal_mode_info,
    }
    return x, normals, normal_stats


@dataclass
class GraphBuildInfo:
    Nw: int
    Fw: int
    Ew: int
    feature_dim: int
    wall_bbox_diag: float
    wall_zone_ids_used: List[int]
    face_counts_by_zone: Dict[int, int]


def build_wall_graph_from_faces(
    nodes_xyz: np.ndarray,
    faces_by_zone: Dict[int, np.ndarray],
    wall_zone_ids: Sequence[int],
    normal_mode: str = "raw",
) -> Tuple[Data, GraphBuildInfo, Dict[str, Any]]:
    if not wall_zone_ids:
        raise ValueError("No wall_zone_ids provided.")
    faces_list: List[np.ndarray] = []
    face_zone_ids_list: List[np.ndarray] = []
    face_counts_by_zone: Dict[int, int] = {}

    for z in wall_zone_ids:
        if z not in faces_by_zone:
            continue
        fz = faces_by_zone[z]
        if fz.size:
            faces_list.append(fz)
            face_zone_ids_list.append(np.full((fz.shape[0],), int(z), dtype=np.int64))
            face_counts_by_zone[int(z)] = int(fz.shape[0])

    if not faces_list:
        raise RuntimeError(f"No faces found for wall zones: {list(wall_zone_ids)}.")

    faces_global = np.vstack(faces_list).astype(np.int64)
    faces_zone_ids = np.concatenate(face_zone_ids_list).astype(np.int64)
    wall_node_ids_sorted = np.sort(np.unique(faces_global.reshape(-1)))

    global_to_local = -np.ones((nodes_xyz.shape[0],), dtype=np.int64)
    global_to_local[wall_node_ids_sorted] = np.arange(wall_node_ids_sorted.size, dtype=np.int64)

    faces_local = global_to_local[faces_global]
    wall_points = nodes_xyz[wall_node_ids_sorted].astype(np.float32)

    x, normals, normal_stats = build_feature_matrix(wall_points, faces_local, normal_mode=normal_mode)

    a, b, c = faces_local[:, 0], faces_local[:, 1], faces_local[:, 2]
    edges = np.vstack([
        np.stack([a, b], axis=1),
        np.stack([b, c], axis=1),
        np.stack([c, a], axis=1),
    ]).astype(np.int64)
    edges_undir = np.unique(np.sort(edges, axis=1), axis=0)
    edge_index = np.vstack([edges_undir, edges_undir[:, ::-1]]).T

    node_patch_zone = compute_node_patch_membership(
        n_nodes=int(wall_points.shape[0]),
        faces_local=faces_local.astype(np.int64),
        faces_zone_ids=faces_zone_ids.astype(np.int64),
    )

    data = Data(
        pos=torch.from_numpy(wall_points).float(),
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        face=torch.from_numpy(faces_local.T.astype(np.int64)).long(),
        num_nodes=int(wall_points.shape[0]),
    )
    data.global_node_ids = torch.from_numpy(wall_node_ids_sorted.astype(np.int64))
    data.face_zone_ids = torch.from_numpy(faces_zone_ids.astype(np.int64))
    data.node_patch_zone = torch.from_numpy(node_patch_zone.astype(np.int64))
    data.vertex_normals = torch.from_numpy(normals.astype(np.float32))

    info = GraphBuildInfo(
        Nw=int(wall_points.shape[0]),
        Fw=int(faces_local.shape[0]),
        Ew=int(edge_index.shape[1]),
        feature_dim=int(x.shape[1]),
        wall_bbox_diag=float(bbox_diag(wall_points)),
        wall_zone_ids_used=[int(z) for z in wall_zone_ids if int(z) in face_counts_by_zone],
        face_counts_by_zone={int(k): int(v) for k, v in face_counts_by_zone.items()},
    )
    contract_aux = {
        "feature_names_base": list(BASE_FEATURE_NAMES),
        "normal_stats": normal_stats,
        "node_patch_zone_unique": sorted([int(z) for z in np.unique(node_patch_zone) if z >= 0]),
    }
    return data, info, contract_aux


def summarize_feature_matrix(x: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    out: Dict[str, Any] = {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "first5_rows": x[:5].tolist(),
        "feature_names": list(feature_names),
        "per_feature": {},
    }
    for j in range(x.shape[1]):
        name = feature_names[j] if j < len(feature_names) else f"f{j}"
        col = x[:, j]
        out["per_feature"][str(name)] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return out


# =============================================================================
# CSV parsing / NN mapping / validation helpers (training-compatible spirit)
# =============================================================================

def try_read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return pd.read_csv(path)


def extract_wall_csv_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        for want in names:
            w = want.lower()
            for k, orig in cols.items():
                if w in k:
                    return orig
        raise KeyError(f"Missing column. Tried {names}. Available: {list(df.columns)}")

    xcol = pick("x-coordinate", "x", "x_coordinate")
    ycol = pick("y-coordinate", "y", "y_coordinate")
    zcol = pick("z-coordinate", "z", "z_coordinate")
    wx = pick("x-wall-shear", "x_wall_shear", "wss_x", "x-wss")
    wy = pick("y-wall-shear", "y_wall_shear", "wss_y", "y-wss")
    wz = pick("z-wall-shear", "z_wall_shear", "wss_z", "z-wss")

    out: Dict[str, np.ndarray] = {
        "xyz": df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32),
        "wss_vec3": df[[wx, wy, wz]].to_numpy(dtype=np.float32),
    }
    for key, tries in [
        ("wss_mag", ("wall-shear", "wall_shear", "wallshear", "wss")),
        ("pressure", ("pressure",)),
        ("node_id", ("nodenumber", "node", "node-id", "node_id")),
    ]:
        try:
            c = pick(*tries)
            out[key] = df[[c]].to_numpy(dtype=np.float32).reshape(-1, 1)
        except Exception:
            pass
    return out


def nearest_neighbor_map(query_xyz: np.ndarray, ref_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(ref_xyz)
        dist, idx = tree.query(query_xyz, k=1)
        return idx.astype(np.int64), dist.astype(np.float32)
    except Exception:
        pass

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


def choose_csv_scale(csv_xyz: np.ndarray, wall_xyz: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    diag_csv = bbox_diag(csv_xyz)
    diag_wall = bbox_diag(wall_xyz)
    ratio = (diag_wall / diag_csv) if (diag_csv > 1e-12 and diag_wall > 1e-12) else 1.0

    candidates = [1.0]
    if 0.0005 <= ratio <= 0.002 or 500.0 <= ratio <= 2000.0:
        candidates.append(float(ratio))

    best_scale = 1.0
    best_p99 = float("inf")
    for sc in candidates:
        xyz2 = (csv_xyz * sc).astype(np.float32)
        _, dist = nearest_neighbor_map(xyz2, wall_xyz)
        p99 = float(np.quantile(dist, 0.99)) if dist.size else float("inf")
        if p99 < best_p99:
            best_p99 = p99
            best_scale = sc

    out = (csv_xyz * best_scale).astype(np.float32)
    info = {
        "auto_scale": float(best_scale),
        "diag_csv": float(diag_csv),
        "diag_wall": float(diag_wall),
        "diag_ratio_wall_over_csv": float(ratio),
        "best_p99_for_scale_choice": float(best_p99),
    }
    return out, info


def face_centroids(points_xyz: np.ndarray, faces_local: np.ndarray) -> np.ndarray:
    p0 = points_xyz[faces_local[:, 0]]
    p1 = points_xyz[faces_local[:, 1]]
    p2 = points_xyz[faces_local[:, 2]]
    return ((p0 + p1 + p2) / 3.0).astype(np.float32)


def face_areas(points_xyz: np.ndarray, faces_local: np.ndarray) -> np.ndarray:
    p0 = points_xyz[faces_local[:, 0]]
    p1 = points_xyz[faces_local[:, 1]]
    p2 = points_xyz[faces_local[:, 2]]
    a = np.cross(p1 - p0, p2 - p0)
    return (0.5 * np.linalg.norm(a, axis=1)).astype(np.float32)


def node_pred_to_face_pred(pred_node_vec: np.ndarray, faces_local: np.ndarray, mode: str = "mean") -> np.ndarray:
    v = pred_node_vec.astype(np.float64)
    f = faces_local.astype(np.int64)
    if mode != "mean":
        raise ValueError(f"Unsupported node->face projection mode: {mode}")
    out = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    return out.astype(np.float64)


def aggregate_row_vectors_to_entities(
    nn_idx: np.ndarray,
    row_vec3: np.ndarray,
    n_entities: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sum_vec = np.zeros((n_entities, 3), dtype=np.float64)
    cnt = np.zeros((n_entities, 1), dtype=np.float64)
    np.add.at(sum_vec, nn_idx, row_vec3.astype(np.float64))
    np.add.at(cnt, nn_idx, 1.0)
    mean_vec = (sum_vec / np.where(cnt > 0, cnt, 1.0)).astype(np.float64)
    hit_counts = cnt[:, 0].astype(np.int64)
    return mean_vec, hit_counts


def compute_vector_metrics(
    pred_vec: np.ndarray,
    truth_vec: np.ndarray,
    has_truth_mask: np.ndarray,
    rel_err_mask_min_truth_mag: float = 1e-4,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    pred = pred_vec.astype(np.float64)
    truth = truth_vec.astype(np.float64)
    err_vec = pred - truth
    err_mag = np.linalg.norm(err_vec, axis=1)
    pred_mag = np.linalg.norm(pred, axis=1)
    truth_mag = np.linalg.norm(truth, axis=1)

    has_truth = has_truth_mask.astype(bool)
    mask_rel = has_truth & (truth_mag > float(rel_err_mask_min_truth_mag))
    eps = 1e-12

    rel_err = np.full_like(truth_mag, np.nan, dtype=np.float64)
    rel_err[mask_rel] = err_mag[mask_rel] / (truth_mag[mask_rel] + eps)

    # magnitude correlation
    if np.sum(mask_rel) >= 3:
        x = truth_mag[mask_rel]
        y = pred_mag[mask_rel]
        r_mag = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((x - x.mean()) ** 2)) + 1e-12
        r2_mag = float(1.0 - ss_res / ss_tot)
    else:
        r_mag = float("nan")
        r2_mag = float("nan")

    # component correlations
    comp_corrs: List[float] = []
    for j in range(3):
        if np.sum(mask_rel) >= 3:
            tx = truth[mask_rel, j]
            px = pred[mask_rel, j]
            if np.std(tx) > 0 and np.std(px) > 0:
                comp_corrs.append(float(np.corrcoef(tx, px)[0, 1]))
            else:
                comp_corrs.append(float("nan"))
        else:
            comp_corrs.append(float("nan"))

    # angular metrics (only where both pred and truth magnitudes are non-trivial)
    pred_nonzero = pred_mag > float(rel_err_mask_min_truth_mag)
    truth_nonzero = truth_mag > float(rel_err_mask_min_truth_mag)
    mask_ang = has_truth & pred_nonzero & truth_nonzero
    cos_sim = np.full((pred.shape[0],), np.nan, dtype=np.float64)
    ang_deg = np.full((pred.shape[0],), np.nan, dtype=np.float64)
    if np.any(mask_ang):
        dot = np.sum(pred[mask_ang] * truth[mask_ang], axis=1)
        denom = (pred_mag[mask_ang] * truth_mag[mask_ang]) + eps
        c = np.clip(dot / denom, -1.0, 1.0)
        cos_sim[mask_ang] = c
        ang_deg[mask_ang] = np.degrees(np.arccos(c))

    metrics = {
        "mse_vec_all": float(np.mean(err_vec ** 2)),
        "mse_vec_hit": float(np.mean(err_vec[has_truth] ** 2)) if np.any(has_truth) else float("nan"),
        "mae_mag_all": float(np.mean(err_mag)),
        "mae_mag_hit": float(np.mean(err_mag[has_truth])) if np.any(has_truth) else float("nan"),
        "rel_err_mean_masked": float(np.nanmean(rel_err[mask_rel])) if np.any(mask_rel) else float("nan"),
        "rel_err_median_masked": float(np.nanmedian(rel_err[mask_rel])) if np.any(mask_rel) else float("nan"),
        "pearson_r_mag_masked": r_mag,
        "r2_mag_masked": r2_mag,
        "pearson_r_x_masked": comp_corrs[0],
        "pearson_r_y_masked": comp_corrs[1],
        "pearson_r_z_masked": comp_corrs[2],
        "cos_sim_mean_masked": float(np.nanmean(cos_sim[mask_ang])) if np.any(mask_ang) else float("nan"),
        "cos_sim_median_masked": float(np.nanmedian(cos_sim[mask_ang])) if np.any(mask_ang) else float("nan"),
        "ang_deg_mean_masked": float(np.nanmean(ang_deg[mask_ang])) if np.any(mask_ang) else float("nan"),
        "ang_deg_median_masked": float(np.nanmedian(ang_deg[mask_ang])) if np.any(mask_ang) else float("nan"),
        "ang_deg_p90_masked": float(np.nanquantile(ang_deg[mask_ang], 0.90)) if np.any(mask_ang) else float("nan"),
        "n_mask_rel": float(np.sum(mask_rel)),
        "n_mask_ang": float(np.sum(mask_ang)),
    }
    arrays = {
        "pred_mag": pred_mag,
        "truth_mag": truth_mag,
        "err_vec": err_vec,
        "err_mag": err_mag,
        "rel_err": rel_err,
        "cos_sim": cos_sim,
        "ang_deg": ang_deg,
        "has_truth": has_truth.astype(np.float64),
    }
    return metrics, arrays


def infer_csv_entity_kind_auto(
    csv_xyz: np.ndarray,
    wall_node_xyz: np.ndarray,
    wall_face_centroids_xyz: np.ndarray,
    n_nodes: int,
    n_faces: int,
) -> Tuple[str, Dict[str, Any]]:
    nn_node_idx, nn_node_dist = nearest_neighbor_map(csv_xyz.astype(np.float32), wall_node_xyz.astype(np.float32))
    nn_face_idx, nn_face_dist = nearest_neighbor_map(csv_xyz.astype(np.float32), wall_face_centroids_xyz.astype(np.float32))

    # Distance and cardinality heuristics
    p99_node = float(np.quantile(nn_node_dist, 0.99)) if nn_node_dist.size else float("inf")
    p99_face = float(np.quantile(nn_face_dist, 0.99)) if nn_face_dist.size else float("inf")
    rows = int(csv_xyz.shape[0])

    rel_node = abs(rows - n_nodes) / max(n_nodes, 1)
    rel_face = abs(rows - n_faces) / max(n_faces, 1)

    score_node = (math.log10(p99_node + 1e-18) if np.isfinite(p99_node) else 99.0) + 0.25 * rel_node
    score_face = (math.log10(p99_face + 1e-18) if np.isfinite(p99_face) else 99.0) + 0.25 * rel_face

    kind = "node_nn" if score_node <= score_face else "face_nn"
    info = {
        "rows_csv": rows,
        "n_nodes": int(n_nodes),
        "n_faces": int(n_faces),
        "p99_node_dist": p99_node,
        "p99_face_dist": p99_face,
        "mean_node_dist": float(np.mean(nn_node_dist)) if nn_node_dist.size else float("nan"),
        "mean_face_dist": float(np.mean(nn_face_dist)) if nn_face_dist.size else float("nan"),
        "rel_rows_to_nodes": float(rel_node),
        "rel_rows_to_faces": float(rel_face),
        "score_node": float(score_node),
        "score_face": float(score_face),
        "chosen_mode": kind,
    }
    return kind, info


def apply_vector_convention(vec: np.ndarray, perm: Tuple[int, int, int], signs: Tuple[int, int, int]) -> np.ndarray:
    out = vec[:, list(perm)].astype(np.float64, copy=True)
    out[:, 0] *= float(signs[0])
    out[:, 1] *= float(signs[1])
    out[:, 2] *= float(signs[2])
    return out


def vector_convention_sanity_scan(
    pred_vec: np.ndarray,
    truth_vec: np.ndarray,
    has_truth_mask: np.ndarray,
    rel_err_mask_min_truth_mag: float,
    top_k: int = 10,
) -> Dict[str, Any]:
    perms = list(itertools.permutations([0, 1, 2], 3))
    sign_opts = list(itertools.product([-1, 1], repeat=3))
    results: List[Dict[str, Any]] = []
    for perm in perms:
        for signs in sign_opts:
            pred2 = apply_vector_convention(pred_vec, perm, signs)
            m, _ = compute_vector_metrics(
                pred_vec=pred2,
                truth_vec=truth_vec,
                has_truth_mask=has_truth_mask,
                rel_err_mask_min_truth_mag=rel_err_mask_min_truth_mag,
            )
            results.append({
                "perm": list(perm),
                "signs": list(signs),
                "cos_sim_mean_masked": m.get("cos_sim_mean_masked"),
                "cos_sim_median_masked": m.get("cos_sim_median_masked"),
                "ang_deg_median_masked": m.get("ang_deg_median_masked"),
                "pearson_r_mag_masked": m.get("pearson_r_mag_masked"),
                "r2_mag_masked": m.get("r2_mag_masked"),
            })

    def _key(r: Dict[str, Any]) -> Tuple[float, float, float]:
        c = r.get("cos_sim_mean_masked")
        a = r.get("ang_deg_median_masked")
        rm = r.get("pearson_r_mag_masked")
        c = -1e9 if c is None or not np.isfinite(c) else float(c)
        a = 1e9 if a is None or not np.isfinite(a) else float(a)
        rm = -1e9 if rm is None or not np.isfinite(rm) else float(rm)
        return (c, -a, rm)

    results_sorted = sorted(results, key=_key, reverse=True)
    return {
        "n_tested": len(results_sorted),
        "top": results_sorted[:int(top_k)],
        "best": results_sorted[0] if results_sorted else None,
    }


def filter_entities_by_zone(
    entity_zone_ids: Optional[np.ndarray],
    include_zones: Optional[Sequence[int]],
    exclude_zones: Optional[Sequence[int]],
    n_entities: int,
) -> np.ndarray:
    mask = np.ones((n_entities,), dtype=bool)
    if entity_zone_ids is None:
        return mask
    z = entity_zone_ids.astype(np.int64)

    if include_zones:
        inc = set(int(v) for v in include_zones)
        mask &= np.array([int(v) in inc for v in z], dtype=bool)
    if exclude_zones:
        exc = set(int(v) for v in exclude_zones)
        mask &= np.array([int(v) not in exc for v in z], dtype=bool)
    return mask


@dataclass
class ValidationResult:
    metrics: Dict[str, float]
    point_data: Dict[str, np.ndarray]
    extra_csv_cols: Dict[str, np.ndarray]
    map_info: Dict[str, Any]


def validate_against_fluent_wall_csv(
    wall_csv_path: Path,
    wall_pos_xyz: np.ndarray,                # [N_nodes,3]
    pred_wss_vec_node: np.ndarray,           # [N_nodes,3]
    wall_faces_local: np.ndarray,            # [N_faces,3]
    face_zone_ids: Optional[np.ndarray] = None,   # [N_faces]
    node_patch_zone: Optional[np.ndarray] = None, # [N_nodes]
    rel_err_mask_min_cfd_mag: float = 1e-4,
    entity_mode: str = "auto",               # auto | node_nn | face_nn
    include_wall_zones: Optional[Sequence[int]] = None,
    exclude_wall_zones: Optional[Sequence[int]] = None,
    vector_convention_sanity: bool = False,
) -> ValidationResult:
    df = try_read_table(wall_csv_path)
    cols = extract_wall_csv_columns(df)
    csv_xyz = cols["xyz"].astype(np.float32)
    csv_wss = cols["wss_vec3"].astype(np.float32)

    # unit sanity for coordinates
    csv_xyz2, scale_info = choose_csv_scale(csv_xyz, wall_pos_xyz.astype(np.float32))

    # Prepare face representation
    faces_local = wall_faces_local.astype(np.int64)
    face_xyz = face_centroids(wall_pos_xyz.astype(np.float64), faces_local)
    pred_face_wss = node_pred_to_face_pred(pred_wss_vec_node.astype(np.float64), faces_local, mode="mean")

    # Decide entity mode
    detect_info: Dict[str, Any] = {}
    if entity_mode == "auto":
        chosen_mode, detect_info = infer_csv_entity_kind_auto(
            csv_xyz=csv_xyz2,
            wall_node_xyz=wall_pos_xyz.astype(np.float32),
            wall_face_centroids_xyz=face_xyz.astype(np.float32),
            n_nodes=int(wall_pos_xyz.shape[0]),
            n_faces=int(faces_local.shape[0]),
        )
    else:
        chosen_mode = str(entity_mode)

    if chosen_mode not in {"node_nn", "face_nn"}:
        raise ValueError(f"Unsupported entity mode: {chosen_mode}")

    point_data: Dict[str, np.ndarray] = {}
    extra_csv_cols: Dict[str, np.ndarray] = {}
    map_info: Dict[str, Any] = {
        "csv_rows": int(csv_xyz.shape[0]),
        "csv_scale_info": scale_info,
        "entity_mode_requested": str(entity_mode),
        "entity_mode_used": str(chosen_mode),
        "entity_auto_detect": detect_info if detect_info else None,
    }

    if chosen_mode == "node_nn":
        nn_idx, nn_dist = nearest_neighbor_map(csv_xyz2, wall_pos_xyz.astype(np.float32))
        cfd_node_wss, hit_counts = aggregate_row_vectors_to_entities(
            nn_idx=nn_idx,
            row_vec3=csv_wss,
            n_entities=int(wall_pos_xyz.shape[0]),
        )

        entity_zone = node_patch_zone.astype(np.int64) if node_patch_zone is not None else None
        entity_keep = filter_entities_by_zone(
            entity_zone_ids=entity_zone,
            include_zones=include_wall_zones,
            exclude_zones=exclude_wall_zones,
            n_entities=int(wall_pos_xyz.shape[0]),
        )
        has_truth = (hit_counts > 0) & entity_keep

        metrics_core, arrays = compute_vector_metrics(
            pred_vec=pred_wss_vec_node.astype(np.float64),
            truth_vec=cfd_node_wss,
            has_truth_mask=has_truth,
            rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
        )

        metrics = {
            "csv_rows": float(csv_xyz.shape[0]),
            "entity_mode_used": 0.0,  # numeric placeholder for flat metrics dict compatibility
            "n_wall_nodes": float(wall_pos_xyz.shape[0]),
            "n_wall_faces": float(faces_local.shape[0]),
            "csv_scale_applied_to_xyz": float(scale_info.get("auto_scale", 1.0)),
            "diag_csv": float(scale_info.get("diag_csv", float("nan"))),
            "diag_wall": float(scale_info.get("diag_wall", float("nan"))),
            "diag_ratio_wall_over_csv": float(scale_info.get("diag_ratio_wall_over_csv", float("nan"))),
            "nn_dist_mean_csv_to_node": float(np.mean(nn_dist)) if nn_dist.size else float("nan"),
            "nn_dist_p95_csv_to_node": float(np.quantile(nn_dist, 0.95)) if nn_dist.size else float("nan"),
            "nn_dist_p99_csv_to_node": float(np.quantile(nn_dist, 0.99)) if nn_dist.size else float("nan"),
            "nn_hits_nonzero_frac": float(np.mean(hit_counts > 0)),
            "validation_entity_keep_frac": float(np.mean(entity_keep)),
            "validation_has_truth_frac": float(np.mean(has_truth)),
        }
        metrics.update(metrics_core)

        # per-zone summary (node dominant patch zone)
        per_zone_metrics: Dict[str, Dict[str, float]] = {}
        if entity_zone is not None:
            for z in sorted([int(v) for v in np.unique(entity_zone) if v >= 0]):
                zmask = (entity_zone == z) & has_truth
                if np.sum(zmask) < 3:
                    continue
                m_z, _ = compute_vector_metrics(
                    pred_vec=pred_wss_vec_node.astype(np.float64),
                    truth_vec=cfd_node_wss,
                    has_truth_mask=zmask,
                    rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
                )
                per_zone_metrics[str(z)] = {
                    "n": float(np.sum(zmask)),
                    "pearson_r_mag_masked": float(m_z.get("pearson_r_mag_masked", float("nan"))),
                    "cos_sim_mean_masked": float(m_z.get("cos_sim_mean_masked", float("nan"))),
                    "ang_deg_median_masked": float(m_z.get("ang_deg_median_masked", float("nan"))),
                    "mae_mag_hit": float(m_z.get("mae_mag_hit", float("nan"))),
                }

        sanity = None
        if vector_convention_sanity:
            sanity = vector_convention_sanity_scan(
                pred_vec=pred_wss_vec_node.astype(np.float64),
                truth_vec=cfd_node_wss,
                has_truth_mask=has_truth,
                rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
                top_k=10,
            )

        point_data = {
            "CFD_WSS_vec": cfd_node_wss,
            "CFD_WSS_mag": arrays["truth_mag"],
            "ERR_vec": arrays["err_vec"],
            "ERR_mag": arrays["err_mag"],
            "NN_hit_count": hit_counts.astype(np.float64),
            "cos_sim": arrays["cos_sim"],
            "ang_deg": arrays["ang_deg"],
        }
        extra_csv_cols = {
            "CFD_WSS": cfd_node_wss,
            "CFD_WSSmag": arrays["truth_mag"],
            "ERR": arrays["err_vec"],
            "ERRmag": arrays["err_mag"],
            "NN_hit_count": hit_counts.astype(np.float64),
            "cos_sim": arrays["cos_sim"],
            "ang_deg": arrays["ang_deg"],
        }
        map_info.update({
            "nn_idx_csv_to_node": nn_idx,
            "nn_dist_csv_to_node": nn_dist,
            "hit_counts_per_node": hit_counts,
            "per_zone_metrics_node_dominant": per_zone_metrics,
            "vector_convention_sanity": sanity,
            "zone_filter_include": [int(z) for z in include_wall_zones] if include_wall_zones else None,
            "zone_filter_exclude": [int(z) for z in exclude_wall_zones] if exclude_wall_zones else None,
        })
        return ValidationResult(metrics=metrics, point_data=point_data, extra_csv_cols=extra_csv_cols, map_info=map_info)

    # --- FACE-CENTERED validation path ---
    nn_idx_f, nn_dist_f = nearest_neighbor_map(csv_xyz2, face_xyz.astype(np.float32))
    cfd_face_wss, face_hit_counts = aggregate_row_vectors_to_entities(
        nn_idx=nn_idx_f,
        row_vec3=csv_wss,
        n_entities=int(faces_local.shape[0]),
    )

    entity_zone_f = face_zone_ids.astype(np.int64) if face_zone_ids is not None else None
    entity_keep_f = filter_entities_by_zone(
        entity_zone_ids=entity_zone_f,
        include_zones=include_wall_zones,
        exclude_zones=exclude_wall_zones,
        n_entities=int(faces_local.shape[0]),
    )
    has_truth_f = (face_hit_counts > 0) & entity_keep_f

    metrics_core_f, arrays_f = compute_vector_metrics(
        pred_vec=pred_face_wss.astype(np.float64),
        truth_vec=cfd_face_wss.astype(np.float64),
        has_truth_mask=has_truth_f,
        rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
    )

    metrics_f = {
        "csv_rows": float(csv_xyz.shape[0]),
        "entity_mode_used": 1.0,  # numeric placeholder
        "n_wall_nodes": float(wall_pos_xyz.shape[0]),
        "n_wall_faces": float(faces_local.shape[0]),
        "csv_scale_applied_to_xyz": float(scale_info.get("auto_scale", 1.0)),
        "diag_csv": float(scale_info.get("diag_csv", float("nan"))),
        "diag_wall": float(scale_info.get("diag_wall", float("nan"))),
        "diag_ratio_wall_over_csv": float(scale_info.get("diag_ratio_wall_over_csv", float("nan"))),
        "nn_dist_mean_csv_to_face": float(np.mean(nn_dist_f)) if nn_dist_f.size else float("nan"),
        "nn_dist_p95_csv_to_face": float(np.quantile(nn_dist_f, 0.95)) if nn_dist_f.size else float("nan"),
        "nn_dist_p99_csv_to_face": float(np.quantile(nn_dist_f, 0.99)) if nn_dist_f.size else float("nan"),
        "nn_hits_nonzero_frac_face": float(np.mean(face_hit_counts > 0)),
        "validation_entity_keep_frac_face": float(np.mean(entity_keep_f)),
        "validation_has_truth_frac_face": float(np.mean(has_truth_f)),
    }
    metrics_f.update(metrics_core_f)

    per_zone_face_metrics: Dict[str, Dict[str, float]] = {}
    if entity_zone_f is not None:
        for z in sorted([int(v) for v in np.unique(entity_zone_f) if v >= 0]):
            zmask = (entity_zone_f == z) & has_truth_f
            if np.sum(zmask) < 3:
                continue
            mz, _ = compute_vector_metrics(
                pred_vec=pred_face_wss.astype(np.float64),
                truth_vec=cfd_face_wss.astype(np.float64),
                has_truth_mask=zmask,
                rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
            )
            per_zone_face_metrics[str(z)] = {
                "n": float(np.sum(zmask)),
                "pearson_r_mag_masked": float(mz.get("pearson_r_mag_masked", float("nan"))),
                "cos_sim_mean_masked": float(mz.get("cos_sim_mean_masked", float("nan"))),
                "ang_deg_median_masked": float(mz.get("ang_deg_median_masked", float("nan"))),
                "mae_mag_hit": float(mz.get("mae_mag_hit", float("nan"))),
            }

    sanity_f = None
    if vector_convention_sanity:
        sanity_f = vector_convention_sanity_scan(
            pred_vec=pred_face_wss.astype(np.float64),
            truth_vec=cfd_face_wss.astype(np.float64),
            has_truth_mask=has_truth_f,
            rel_err_mask_min_truth_mag=float(rel_err_mask_min_cfd_mag),
            top_k=10,
        )

    # For point_data/extra_csv_cols in face mode, keep empty/minimal to avoid misleading node-sized "truth"
    point_data = {}
    extra_csv_cols = {}
    map_info.update({
        "nn_idx_csv_to_face": nn_idx_f,
        "nn_dist_csv_to_face": nn_dist_f,
        "hit_counts_per_face": face_hit_counts,
        "per_zone_metrics_face": per_zone_face_metrics,
        "vector_convention_sanity": sanity_f,
        "zone_filter_include": [int(z) for z in include_wall_zones] if include_wall_zones else None,
        "zone_filter_exclude": [int(z) for z in exclude_wall_zones] if exclude_wall_zones else None,
    })
    return ValidationResult(metrics=metrics_f, point_data=point_data, extra_csv_cols=extra_csv_cols, map_info=map_info)


# =============================================================================
# Export helpers
# =============================================================================

def export_csv(out_csv: Path, pos_xyz: np.ndarray, pred_wss: np.ndarray, extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(pos_xyz, dtype=np.float64)
    pred = np.asarray(pred_wss, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos_xyz must be [N,3], got {pos.shape}")
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred_wss must be [N,3], got {pred.shape}")
    if pred.shape[0] != pos.shape[0]:
        raise ValueError("pos_xyz and pred_wss must have same N")

    pred_mag = np.linalg.norm(pred, axis=1, keepdims=True)
    cols = [
        pos[:, [0]], pos[:, [1]], pos[:, [2]],
        pred[:, [0]], pred[:, [1]], pred[:, [2]],
        pred_mag,
    ]
    headers = ["x", "y", "z", "WSSx", "WSSy", "WSSz", "WSSmag"]

    if extra:
        for k, v in extra.items():
            arr = np.asarray(v)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != pos.shape[0]:
                continue
            cols.append(arr.astype(np.float64))
            if arr.shape[1] == 1:
                headers.append(str(k))
            elif arr.shape[1] == 3:
                headers.extend([f"{k}x", f"{k}y", f"{k}z"])
            else:
                headers.extend([f"{k}_{j}" for j in range(arr.shape[1])])

    mat = np.hstack(cols)
    np.savetxt(out_csv, mat, delimiter=",", header=",".join(headers), comments="")


def export_vtp_pyvista(out_vtp: str, points_xyz, faces_tri, point_data=None):
    """
    Write VTK XML PolyData (.vtp) for ParaView using PyVista.

    points_xyz: (N,3) float array
    faces_tri : (M,3) int array, 0-based triangle indices
    point_data: dict[str, np.ndarray] arrays of shape (N,) or (N,3)
    """
    import numpy as np
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError("PyVista not available. Install with: pip install pyvista") from e

    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    faces_tri  = np.asarray(faces_tri, dtype=np.int64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {points_xyz.shape}")
    if faces_tri.ndim != 2 or faces_tri.shape[1] != 3:
        raise ValueError(f"faces_tri must be (M,3), got {faces_tri.shape}")
    if faces_tri.size > 0:
        if faces_tri.min() < 0 or faces_tri.max() >= points_xyz.shape[0]:
            raise ValueError("faces_tri has indices out of range for points_xyz")

    # PyVista face format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    if faces_tri.shape[0] == 0:
        # empty mesh
        mesh = pv.PolyData(points_xyz)
    else:
        faces_pv = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri]).ravel()
        mesh = pv.PolyData(points_xyz, faces_pv)

    # Attach point data
    if point_data:
        for k, arr in point_data.items():
            a = np.asarray(arr)
            if a.ndim == 1:
                if a.shape[0] != points_xyz.shape[0]:
                    raise ValueError(f"Point array '{k}' length {a.shape[0]} != N {points_xyz.shape[0]}")
                mesh.point_data[k] = a.astype(np.float32)
            elif a.ndim == 2 and a.shape[0] == points_xyz.shape[0]:
                mesh.point_data[k] = a.astype(np.float32)
            else:
                raise ValueError(f"Point array '{k}' must be (N,) or (N,C); got {a.shape}")

    mesh.save(out_vtp)
    return True


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    parts = re.split(r"[,\s]+", str(s).strip())
    return [int(p) for p in parts if p]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inference compatible with upgradedtraining.py checkpoint + mesh parser + graph semantics."
    )
    ap.add_argument("--ckpt", required=True, type=str, help="Path to best_model.pt from training")
    ap.add_argument("--msh", required=True, type=str, help="Path to NEW Fluent legacy ASCII .msh")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--name", default=None, type=str, help="Output stem (default: msh stem)")
    ap.add_argument("--device", default=None, type=str, help="cpu or cuda (default auto)")

    # mesh parser / wall selection controls (training-style)
    ap.add_argument("--int_mode", default="hex", choices=["hex", "dec", "auto"], help="Fluent integer parsing mode")
    ap.add_argument("--wall_mode", default="auto", choices=["auto", "manual"], help="Wall zone selection mode")
    ap.add_argument("--wall_zone", default=None, type=str, help="Manual wall zone IDs (comma/space-separated)")
    ap.add_argument("--wall_auto_min_faces_frac", type=float, default=0.0, help="Auto mode: drop tiny wall patches")

    # conditioning
    ap.add_argument("--re", default=None, type=float, help="Reynolds number (required if checkpoint append_re_to_x=True)")

    # phase1: normals / contract checks
    ap.add_argument("--normal_mode", default="raw", choices=["raw", "flip", "auto_centroid"],
                    help="Normal handling mode before inference (Phase1 contract/orientation debugging).")
    ap.add_argument("--contract_check", action="store_true",
                    help="Log and save strict feature/graph/normal contract diagnostics.")
    ap.add_argument("--debug_try_flipped_normals", action="store_true",
                    help="If validating, run an additional debug inference with flipped normals and compare validation metrics.")

    # outputs
    ap.add_argument("--export_csv", action="store_true", help="Export clinician-friendly CSV")
    ap.add_argument("--export_vtp", action="store_true", help="Export .vtp for ParaView (requires meshio)")
    ap.add_argument("--save_pred_pt", action="store_true", help="Save PyG Data + predictions .pt artifact")
    ap.add_argument("--save_json", action="store_true", help="Save metadata JSON (default on if any output)")
    ap.add_argument("--log", default=None, type=str, help="Optional log file path")

    # optional validation
    ap.add_argument("--cfd_wall_csv", default=None, type=str, help="Optional Fluent wall_data_Re*.csv for validation")
    ap.add_argument("--rel_err_mask_min_cfd_mag", type=float, default=1e-4, help="Mask threshold for relative error stats")

    # phase1: validation semantics / forensic checks
    ap.add_argument("--validation_entity_mode", default="auto", choices=["auto", "node_nn", "face_nn"],
                    help="Validation entity mode: auto-detect nodal vs face-centered CSV semantics.")
    ap.add_argument("--validation_include_wall_zones", default=None, type=str,
                    help="Optional validation filter: only include these wall zones (comma/space-separated).")
    ap.add_argument("--validation_exclude_wall_zones", default=None, type=str,
                    help="Optional validation filter: exclude these wall zones (comma/space-separated).")
    ap.add_argument("--vector_convention_sanity", action="store_true",
                    help="Run permutation/sign forensic scan during validation (does not alter exported predictions).")

    args = ap.parse_args()

    logger = Logger(Path(args.log) if args.log else None)
    try:
        ckpt_path = Path(args.ckpt)
        msh_path = Path(args.msh)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.name if args.name else msh_path.stem
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.log("=" * 96)
        logger.log("INFERENCE")
        logger.log("=" * 96)
        logger.log(f"CKPT     : {ckpt_path}")
        logger.log(f"MSH      : {msh_path}")
        logger.log(f"OUT_DIR  : {out_dir}")
        logger.log(f"DEVICE   : {device}")
        logger.log(f"INT_MODE : {args.int_mode}")
        logger.log(f"NORMAL_MODE: {args.normal_mode}")
        logger.log("")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required_ckpt_keys = ["model_state", "in_dim", "hidden_dim", "num_layers", "dropout"]
        missing = [k for k in required_ckpt_keys if k not in ckpt]
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")

        in_dim = int(ckpt["in_dim"])
        out_dim = int(ckpt.get("out_dim", 3))
        hidden_dim = int(ckpt["hidden_dim"])
        num_layers = int(ckpt["num_layers"])
        dropout = float(ckpt["dropout"])
        append_re_to_x = bool(ckpt.get("append_re_to_x", False))
        normalize_y = bool(ckpt.get("normalize_y", False))
        re_mu = float(ckpt.get("re_mu", 0.0))
        re_sd = float(ckpt.get("re_sd", 1.0))
        y_mu = ckpt.get("y_mu", torch.zeros((out_dim,), dtype=torch.float32))
        y_sd = ckpt.get("y_sd", torch.ones((out_dim,), dtype=torch.float32))
        y_mu = y_mu.float().cpu().view(-1)
        y_sd = y_sd.float().cpu().view(-1)
        if y_mu.numel() != out_dim or y_sd.numel() != out_dim:
            raise RuntimeError(f"Checkpoint y_mu/y_sd shape mismatch with out_dim={out_dim}: "
                               f"y_mu={tuple(y_mu.shape)} y_sd={tuple(y_sd.shape)}")

        logger.log(f"[CKPT] in_dim={in_dim} out_dim={out_dim} hidden_dim={hidden_dim} num_layers={num_layers} dropout={dropout}")
        logger.log(f"[CKPT] append_re_to_x={append_re_to_x} normalize_y={normalize_y} re_mu={re_mu:.6g} re_sd={re_sd:.6g}")

        scan = scan_fluent_msh_headers(msh_path, int_mode=args.int_mode)
        logger.log(f"[MESH] scaling_factor={scan.scaling_factor}")
        if scan.node_decl:
            logger.log(f"[MESH] node_decl first={scan.node_decl.first} last={scan.node_decl.last} n={scan.node_decl.n_nodes}")
        logger.log(f"[MESH] face_blocks={len(scan.face_blocks)}")

        wall_sel = WallSelection(
            mode=str(args.wall_mode),
            manual_zone_ids=parse_int_list(args.wall_zone) if args.wall_zone is not None else None,
            auto_min_faces_frac=float(args.wall_auto_min_faces_frac),
        )
        wall_zones = select_wall_zones(scan, wall_sel, logger)
        if not wall_zones:
            raise RuntimeError("No wall zones selected. Use --wall_mode manual --wall_zone <id(s)> if auto fails.")

        nodes_xyz, faces_by_zone, scan2 = parse_fluent_msh_nodes_and_faces(
            msh_path=msh_path,
            wanted_face_zones=wall_zones,
            int_mode=str(args.int_mode),
        )
        data, ginfo, contract_aux = build_wall_graph_from_faces(
            nodes_xyz, faces_by_zone, wall_zones, normal_mode=str(args.normal_mode)
        )
        logger.log(f"[GRAPH] nodes={ginfo.Nw} faces={ginfo.Fw} edges={ginfo.Ew} x_dim={ginfo.feature_dim} bbox_diag={ginfo.wall_bbox_diag:.6g}")
        logger.log(f"[GRAPH] wall zones used: {ginfo.wall_zone_ids_used}")
        if ginfo.face_counts_by_zone:
            logger.log(f"[GRAPH] face counts by zone: {ginfo.face_counts_by_zone}")

        # Feature schema contract (base + optional Re appended)
        feature_names = list(BASE_FEATURE_NAMES)
        if append_re_to_x:
            feature_names_ckpt_expected = feature_names + ["re_norm"]
        else:
            feature_names_ckpt_expected = feature_names[:]

        if append_re_to_x:
            if args.re is None:
                raise ValueError("Checkpoint expects Re appended to x (append_re_to_x=True); provide --re <value>.")
            data.re = torch.tensor(float(args.re), dtype=torch.float32)
            re_val = float(data.re.item())
            re_norm = (re_val - re_mu) / (re_sd if abs(re_sd) > 1e-12 else 1.0)
            re_feat = torch.full((data.num_nodes, 1), float(re_norm), dtype=torch.float32)
            data.x = torch.cat([data.x.float(), re_feat], dim=1)
            logger.log(f"[COND] Appended Re feature: Re={re_val:.6g}, Re_norm={re_norm:.6g}")
        else:
            if args.re is not None:
                data.re = torch.tensor(float(args.re), dtype=torch.float32)
                logger.log(f"[COND] --re provided ({args.re}) but checkpoint append_re_to_x=False; not appending.")

        xdim = int(data.x.shape[1])
        if xdim != in_dim:
            raise RuntimeError(
                f"Feature dim mismatch: built x.shape[1]={xdim} but checkpoint expects in_dim={in_dim}. "
                f"append_re_to_x={append_re_to_x}, --re={args.re}"
            )

        # Contract check summary
        contract_check_info: Dict[str, Any] = {}
        if args.contract_check:
            x_np_contract = data.x.cpu().numpy().astype(np.float64)
            feature_summary = summarize_feature_matrix(x_np_contract, feature_names_ckpt_expected)
            normals_stats = contract_aux.get("normal_stats", {})
            contract_check_info = {
                "feature_schema_expected_inference": feature_names_ckpt_expected,
                "feature_summary": feature_summary,
                "normal_stats": normals_stats,
                "graph_counts": {
                    "nodes": ginfo.Nw,
                    "faces": ginfo.Fw,
                    "edges_directed": ginfo.Ew,
                },
                "wall_zone_ids": list(wall_zones),
                "face_counts_by_zone": dict(ginfo.face_counts_by_zone),
            }
            logger.log("=" * 96)
            logger.log("CONTRACT CHECK")
            logger.log("=" * 96)
            logger.log(f"[CONTRACT] feature_names={feature_names_ckpt_expected}")
            logger.log(f"[CONTRACT] x.shape={tuple(data.x.shape)} checkpoint_in_dim={in_dim}")
            # compact per-feature stats logging
            for fname in feature_summary["feature_names"]:
                s = feature_summary["per_feature"][fname]
                logger.log(f"[CONTRACT] {fname:>10s}: mean={s['mean']:.6g} std={s['std']:.6g} min={s['min']:.6g} max={s['max']:.6g}")
            ns = normals_stats
            if ns:
                logger.log(f"[CONTRACT] normals mode={ns.get('normal_mode')} near_zero_frac={ns.get('near_zero_frac'):.6g} "
                           f"norm_mean={ns.get('norm_mean'):.6g} norm_min={ns.get('norm_min'):.6g} norm_max={ns.get('norm_max'):.6g}")

        model = WSSNet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, dropout=dropout)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()
        logger.log("[MODEL] Checkpoint weights loaded successfully.")

        data_dev = data.to(device)
        with torch.no_grad():
            pred = model(data_dev).detach().cpu()

        if normalize_y:
            pred = pred * y_sd.view(1, -1) + y_mu.view(1, -1)
            logger.log("[POST] Applied y denormalization from checkpoint y_mu/y_sd.")
        pred_np = pred.numpy().astype(np.float64)

        pos_np = data.pos.cpu().numpy().astype(np.float64)
        faces_local = data.face.cpu().numpy().T.astype(np.int64)
        face_zone_ids_np = data.face_zone_ids.cpu().numpy().astype(np.int64) if hasattr(data, "face_zone_ids") else None
        node_patch_zone_np = data.node_patch_zone.cpu().numpy().astype(np.int64) if hasattr(data, "node_patch_zone") else None
        pred_mag = np.linalg.norm(pred_np, axis=1)

        point_data: Dict[str, np.ndarray] = {
            "WSS_vec": pred_np,
            "WSS_mag": pred_mag.astype(np.float64),
        }
        extra_csv_cols: Dict[str, np.ndarray] = {}
        validation_metrics: Dict[str, float] = {}
        validation_map_info: Dict[str, Any] = {}
        debug_flipped_normals_validation: Optional[Dict[str, Any]] = None

        if args.cfd_wall_csv:
            include_vzones = parse_int_list(args.validation_include_wall_zones)
            exclude_vzones = parse_int_list(args.validation_exclude_wall_zones)

            vres = validate_against_fluent_wall_csv(
                wall_csv_path=Path(args.cfd_wall_csv),
                wall_pos_xyz=pos_np,
                pred_wss_vec_node=pred_np,
                wall_faces_local=faces_local,
                face_zone_ids=face_zone_ids_np,
                node_patch_zone=node_patch_zone_np,
                rel_err_mask_min_cfd_mag=float(args.rel_err_mask_min_cfd_mag),
                entity_mode=str(args.validation_entity_mode),
                include_wall_zones=include_vzones if include_vzones else None,
                exclude_wall_zones=exclude_vzones if exclude_vzones else None,
                vector_convention_sanity=bool(args.vector_convention_sanity),
            )
            validation_metrics = vres.metrics
            validation_map_info = vres.map_info
            point_data.update(vres.point_data)
            extra_csv_cols.update(vres.extra_csv_cols)

            logger.log("=" * 96)
            logger.log("VALIDATION vs Fluent wall CSV")
            logger.log("=" * 96)
            mode_used = validation_map_info.get("entity_mode_used", "unknown")
            logger.log(f"[VAL] entity_mode_requested={args.validation_entity_mode} entity_mode_used={mode_used}")
            if validation_map_info.get("entity_auto_detect"):
                logger.log(f"[VAL] auto_detect={validation_map_info['entity_auto_detect']}")
            for k, v in validation_metrics.items():
                logger.log(f"  {k:32s}: {v}")

            # Phase 1 forensic: try flipped normals without changing main prediction export
            if args.debug_try_flipped_normals:
                logger.log("=" * 96)
                logger.log("DEBUG: FLIPPED-NORMALS VALIDATION A/B TEST")
                logger.log("=" * 96)
                # Rebuild graph with flipped normals, same wall zones, same connectivity semantics
                data_flip, ginfo_flip, contract_aux_flip = build_wall_graph_from_faces(
                    nodes_xyz, faces_by_zone, wall_zones, normal_mode="flip"
                )
                # append Re same as main path
                if append_re_to_x:
                    re_val = float(args.re)
                    re_norm = (re_val - re_mu) / (re_sd if abs(re_sd) > 1e-12 else 1.0)
                    re_feat_flip = torch.full((data_flip.num_nodes, 1), float(re_norm), dtype=torch.float32)
                    data_flip.x = torch.cat([data_flip.x.float(), re_feat_flip], dim=1)
                if int(data_flip.x.shape[1]) != in_dim:
                    raise RuntimeError("[DEBUG flip] x dim mismatch after flipped normal rebuild.")

                with torch.no_grad():
                    pred_flip = model(data_flip.to(device)).detach().cpu()
                if normalize_y:
                    pred_flip = pred_flip * y_sd.view(1, -1) + y_mu.view(1, -1)
                pred_flip_np = pred_flip.numpy().astype(np.float64)

                vres_flip = validate_against_fluent_wall_csv(
                    wall_csv_path=Path(args.cfd_wall_csv),
                    wall_pos_xyz=data_flip.pos.cpu().numpy().astype(np.float64),
                    pred_wss_vec_node=pred_flip_np,
                    wall_faces_local=data_flip.face.cpu().numpy().T.astype(np.int64),
                    face_zone_ids=data_flip.face_zone_ids.cpu().numpy().astype(np.int64) if hasattr(data_flip, "face_zone_ids") else None,
                    node_patch_zone=data_flip.node_patch_zone.cpu().numpy().astype(np.int64) if hasattr(data_flip, "node_patch_zone") else None,
                    rel_err_mask_min_cfd_mag=float(args.rel_err_mask_min_cfd_mag),
                    entity_mode=str(args.validation_entity_mode),
                    include_wall_zones=include_vzones if include_vzones else None,
                    exclude_wall_zones=exclude_vzones if exclude_vzones else None,
                    vector_convention_sanity=bool(args.vector_convention_sanity),
                )
                debug_flipped_normals_validation = {
                    "graph_info": dataclasses.asdict(ginfo_flip),
                    "normal_stats": contract_aux_flip.get("normal_stats"),
                    "metrics": vres_flip.metrics,
                    "map_info_summary": {
                        "entity_mode_used": vres_flip.map_info.get("entity_mode_used"),
                        "entity_auto_detect": vres_flip.map_info.get("entity_auto_detect"),
                    },
                }
                # concise comparison log
                for key in [
                    "pearson_r_mag_masked",
                    "cos_sim_mean_masked",
                    "ang_deg_median_masked",
                    "mae_mag_hit",
                    "r2_mag_masked",
                ]:
                    a = validation_metrics.get(key, float("nan"))
                    b = vres_flip.metrics.get(key, float("nan"))
                    logger.log(f"[DEBUG flip] {key:24s}: main={a} | flipped={b}")

        do_any_output = bool(args.export_csv or args.export_vtp or args.save_pred_pt or args.save_json or args.cfd_wall_csv)

        if args.export_csv:
            out_csv = out_dir / f"{stem}_pred.csv"
            export_csv(out_csv, pos_np, pred_np, extra=extra_csv_cols)
            logger.log(f"[OK] Wrote CSV: {out_csv}")

        vtp_written = None
        if args.export_vtp:
            out_vtp = out_dir / f"{stem}_pred.vtp"
            ok = export_vtp_pyvista(out_vtp, pos_np, faces_local, point_data=point_data)
            vtp_written = bool(ok)
            if ok:
                logger.log(f"[OK] Wrote VTP: {out_vtp}")
            else:
                logger.log("[WARN] meshio not installed; skipping VTP export. Install with: pip install meshio")

        if args.save_pred_pt:
            out_pt = out_dir / f"{stem}_pred.pt"
            data_cpu = data.clone().cpu()  # PyG-safe clone/cpu (avoid .detach() on Data)
            data_cpu.pred = torch.from_numpy(pred_np.astype(np.float32))
            data_cpu.pred_mag = torch.from_numpy(pred_mag.astype(np.float32))
            data_cpu.inference_meta = {
                "ckpt": str(ckpt_path),
                "msh": str(msh_path),
                "wall_zone_ids": wall_zones,
                "append_re_to_x": append_re_to_x,
                "normalize_y": normalize_y,
                "re": float(args.re) if args.re is not None else None,
                "normal_mode": str(args.normal_mode),
            }
            if validation_metrics:
                data_cpu.validation_metrics = validation_metrics
            torch.save(data_cpu, out_pt)
            logger.log(f"[OK] Wrote PT: {out_pt}")

        if args.save_json or do_any_output:
            # Human-readable validation mode string (metrics dict stores numeric placeholder for compatibility)
            val_mode_used_str = validation_map_info.get("entity_mode_used") if validation_map_info else None
            meta = {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "msh": str(msh_path),
                "msh_sha1": sha1_file(msh_path) if msh_path.exists() else None,
                "ckpt": str(ckpt_path),
                "ckpt_keys_present": sorted([str(k) for k in ckpt.keys() if isinstance(k, str)]),
                "device": str(device),
                "int_mode": str(args.int_mode),
                "wall_mode": str(args.wall_mode),
                "wall_zone_ids": wall_zones,
                "wall_auto_min_faces_frac": float(args.wall_auto_min_faces_frac),
                "scan_header": dataclasses.asdict(scan2),
                "graph_info": dataclasses.asdict(ginfo),
                "normal_mode": str(args.normal_mode),
                "contract_check": bool(args.contract_check),
                "contract_check_info": contract_check_info if contract_check_info else None,
                "checkpoint_contract": {
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "append_re_to_x": append_re_to_x,
                    "normalize_y": normalize_y,
                    "re_mu": re_mu,
                    "re_sd": re_sd,
                    "y_mu": y_mu,
                    "y_sd": y_sd,
                    "best_epoch": ckpt.get("best_epoch"),
                    "feature_version": ckpt.get("feature_version"),
                    "target_version": ckpt.get("target_version"),
                    "pipeline_version": ckpt.get("pipeline_version"),
                    "feature_names_inference_expected": feature_names_ckpt_expected,
                },
                "input_re": float(args.re) if args.re is not None else None,
                "feature_dim_built": int(data.x.shape[1]),
                "prediction_summary": {
                    "num_nodes": int(pred_np.shape[0]),
                    "pred_vec_shape": list(pred_np.shape),
                    "pred_mag_mean": float(np.mean(pred_mag)),
                    "pred_mag_std": float(np.std(pred_mag)),
                    "pred_mag_min": float(np.min(pred_mag)),
                    "pred_mag_max": float(np.max(pred_mag)),
                },
                "validation_wall_csv": str(args.cfd_wall_csv) if args.cfd_wall_csv else None,
                "validation_requested": {
                    "entity_mode": str(args.validation_entity_mode),
                    "include_wall_zones": parse_int_list(args.validation_include_wall_zones),
                    "exclude_wall_zones": parse_int_list(args.validation_exclude_wall_zones),
                    "vector_convention_sanity": bool(args.vector_convention_sanity),
                } if args.cfd_wall_csv else None,
                "validation_mode_used": str(val_mode_used_str) if val_mode_used_str else None,
                "validation_metrics": validation_metrics if validation_metrics else None,
                "validation_map_info_summary": {
                    "csv_rows": int(validation_map_info.get("csv_rows", 0)) if validation_map_info else None,
                    "entity_mode_used": validation_map_info.get("entity_mode_used") if validation_map_info else None,
                    "entity_auto_detect": validation_map_info.get("entity_auto_detect") if validation_map_info else None,
                    "nn_dist_mean_node": float(np.mean(validation_map_info["nn_dist_csv_to_node"]))
                        if (validation_map_info and "nn_dist_csv_to_node" in validation_map_info) else None,
                    "nn_dist_p99_node": float(np.quantile(validation_map_info["nn_dist_csv_to_node"], 0.99))
                        if (validation_map_info and "nn_dist_csv_to_node" in validation_map_info) else None,
                    "nn_dist_mean_face": float(np.mean(validation_map_info["nn_dist_csv_to_face"]))
                        if (validation_map_info and "nn_dist_csv_to_face" in validation_map_info) else None,
                    "nn_dist_p99_face": float(np.quantile(validation_map_info["nn_dist_csv_to_face"], 0.99))
                        if (validation_map_info and "nn_dist_csv_to_face" in validation_map_info) else None,
                    "per_zone_metrics_node_dominant": validation_map_info.get("per_zone_metrics_node_dominant") if validation_map_info else None,
                    "per_zone_metrics_face": validation_map_info.get("per_zone_metrics_face") if validation_map_info else None,
                    "vector_convention_sanity": validation_map_info.get("vector_convention_sanity") if validation_map_info else None,
                } if validation_map_info else None,
                "debug_flipped_normals_validation": debug_flipped_normals_validation,
                "outputs": {
                    "csv": str(out_dir / f"{stem}_pred.csv") if args.export_csv else None,
                    "vtp": str(out_dir / f"{stem}_pred.vtp") if args.export_vtp else None,
                    "vtp_written": vtp_written,
                    "pred_pt": str(out_dir / f"{stem}_pred.pt") if args.save_pred_pt else None,
                },
            }
            meta_path = out_dir / f"{stem}_meta.json"
            write_json(meta_path, meta)
            logger.log(f"[OK] Wrote metadata JSON: {meta_path}")

        logger.log("DONE.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()