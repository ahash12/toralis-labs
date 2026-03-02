#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFD-to-GNN: Fluent/ANSYS vascular hemodynamics pipeline (single-file).

This script merges and hardens the functionality that used to be split across:
  - trainingpre.py (Fluent legacy ASCII .msh + wall_data_Re*.csv -> Re*.pt)
  - training.py     (train a node-level regression GNN on Re*.pt graphs)

Non-negotiables (kept):
  - PyTorch Geometric
  - vertex-based wall graph (nodes = wall vertices)
  - targets: wall shear stress vector per node [tau_x, tau_y, tau_z]
  - conditioning: Reynolds number (Re), appended as a node feature channel

Primary goals (robustness/correctness-first):
  - robust wall extraction across meshes (auto by bc-type wall + optional overrides)
  - robust CSV<->mesh alignment (nearest-neighbor with diagnostics; no silent dropping)
  - unit mismatch detection (mm vs m) with explicit decision + reporting
  - consistent graph/label semantics so .pt files are reusable for future inference

Author: you + ChatGPT
Date: 2026-02-24 (AU/Brisbane)
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import random
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
from torch_geometric.loader import DataLoader


# ======================================================================================
# Versioning / semantics
# ======================================================================================

PIPELINE_VERSION = "fluent-wss-gnn-pipeline@2026-02-24"
FEATURE_VERSION = "wall_vtx_geom_normXYZ_normals_iswall_v1"
TARGET_VERSION = "wss_vec3_v1"  # [tau_x, tau_y, tau_z] at wall vertices


# ======================================================================================
# Reproducibility
# ======================================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ======================================================================================
# Logging helpers
# ======================================================================================

class Logger:
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self._fh = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_path, "w", encoding="utf-8")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(x):
        # pathlib.Path (includes WindowsPath / PosixPath)
        if isinstance(x, Path):
            return str(x)

        # torch tensors (for small metadata tensors/scalars)
        if torch.is_tensor(x):
            if x.ndim == 0:
                return x.item()
            return x.detach().cpu().tolist()

        # numpy scalars / arrays
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()

        # sets/tuples (rare but safe to support)
        if isinstance(x, (set, tuple)):
            return list(x)

        # fallback: stringify to avoid hard crash on metadata-only objects
        return str(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ======================================================================================
# Fluent legacy ASCII .msh parsing (robust, hex-default)
# ======================================================================================

# Fluent legacy mesh file format uses hexadecimal indices for nodes/faces/cells in grid sections.
# In practice, tokens like "13" often mean hex 0x13 = decimal 19.
# This is one of your primary failure modes - so we default to base16 for integer fields.


def _strip_parens(s: str) -> str:
    return s.replace("(", " ").replace(")", " ").strip()


def _balance_count(s: str) -> int:
    return s.count("(") - s.count(")")


def parse_int_token(tok: str, int_mode: str = "hex") -> int:
    """
    Parse an integer token from a Fluent legacy ASCII mesh.

    int_mode:
      - "hex": parse as base16 ALWAYS (including digit-only tokens). (Recommended for Fluent)
      - "dec": parse as base10 ALWAYS.
      - "auto": try hex; if it looks wildly implausible for common fields, fall back to dec.

    Notes:
      - Handles optional leading '-'.
      - Does NOT accept floats.
    """
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

    # auto: attempt hex, then sanity-check for common header fields
    v_hex = sign * int(t, 16)
    v_dec = sign * int(t, 10)

    # Heuristic: zone IDs for vascular meshes are typically < 1e6,
    # and indices (node IDs, face IDs) are positive.
    # If hex parse explodes by >100x compared to dec, prefer dec.
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
        """bc-type can be offset by 1000 for non-conformal intersections (e.g., 1003 wall)."""
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
    """
    Header-only scan:
      - meshing-to-solver-scaling-factor (if present)
      - node declaration section: (10 (0 first last 0 [ND]))
      - face block headers:      (13 (zone-id first last bc-type face-type))

    Does NOT parse full nodes/faces bodies (fast and safe).
    """
    scaling = 1.0
    node_decl: Optional[NodeDeclMeta] = None
    face_blocks: List[FaceBlockMeta] = []

    with msh_path.open("r", errors="ignore") as f:
        ln = 0
        for line in f:
            ln += 1
            s = line.strip()

            # scaling factor
            if "meshing-to-solver-scaling-factor" in s:
                toks = _strip_parens(s).split()
                try:
                    scaling = float(toks[-1])
                except Exception:
                    pass

            # nodes declaration (zone 0)
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

            # faces headers
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
                        face_blocks.append(
                            FaceBlockMeta(zone_id=zone, first=first, last=last, bc_type=bc, face_type=ft, header_line=ln)
                        )

    return MeshHeaderScan(
        msh_path=str(msh_path),
        int_mode=int_mode,
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
    """
    Full parse:
      - nodes_xyz: dense array, 0-based index in array corresponds to Fluent node-id starting at node_decl.first
      - faces_by_zone: dict zone_id -> [Fz, 3] global indices (0-based into nodes_xyz)
        (quads/polys are triangulated)

    wanted_face_zones:
      - None => do not parse any faces (nodes only)
      - empty list => parse no faces
      - list of zone IDs => parse only those face zones
    """
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
            f"Unreasonable node count parsed ({n_nodes}). "
            f"Try --int_mode dec if this mesh uses decimal indices."
        )

    node_first = scan.node_decl.first
    node_last = scan.node_decl.last

    nodes_xyz = np.zeros((n_nodes, 3), dtype=np.float32)

    wanted_set: Optional[set] = None
    if wanted_face_zones is None:
        wanted_set = None
    else:
        wanted_set = set(int(z) for z in wanted_face_zones)

    faces_by_zone: Dict[int, List[List[int]]] = {}

    def node_to_local(nid_1based: int) -> int:
        idx = nid_1based - node_first
        if idx < 0 or idx >= n_nodes:
            raise IndexError(f"Node id {nid_1based} out of declared bounds [{node_first},{node_last}]")
        return idx

    def _read_header_line_only(fh, first_line: str, line_no: int) -> Tuple[str, int]:
        """
        Read only the section header, NOT the whole section body.

        Fluent headers are often a single line like:
            (10 (c2e d45d 3392b 1 3) (
            (13 (13 1 dfed 3 3)(

        We stop once the nested header tuple '(...)' is closed. We do NOT
        balance the trailing body-opening '('.
        """
        hdr = first_line.strip()

        # Fast path: most meshes have full header on one line.
        # We need at least two ')' chars to close:
        #   (10 ( ... ) (
        #        ^ closes nested tuple, and usually outer syntax also contributes formatting
        # In practice one-line header parsing works if tokens are present.
        toks_try = _strip_parens(hdr).split()
        if len(toks_try) >= 6:
            return hdr, line_no

        # Fallback: continue a few lines until token count is enough or we hit body-ish data.
        for _ in range(5):
            nxt = fh.readline()
            if not nxt:
                break
            line_no += 1
            s2 = nxt.strip()

            # If next line looks like numeric body data, stop (header was already complete enough)
            if s2 and (s2[0].isdigit() or s2[0] in "+-."):
                # We cannot unread safely; append and let parser fail loudly if malformed.
                # This path is unlikely for Fluent legacy headers.
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

            # nodes section
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
                    # toks[4] is node type flag, toks[5] is dim
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

                    # End of this node section
                    if raw.startswith(")"):
                        break

                    t = _strip_parens(raw)
                    if t == "":
                        continue

                    vals = []
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
                        f"This usually indicates wrong integer parsing (--int_mode) or a malformed/unsupported .msh."
                    )

                continue

            # faces section
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

                # Only parse wanted zones if specified
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
                    face_type = parse_int_token(toks[5], int_mode=int_mode)  # 0 mixed, 3 tri, 4 quad, 5 poly
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

                    toks2 = t.split()
                    ints: List[int] = []
                    for a in toks2:
                        try:
                            ints.append(parse_int_token(a, int_mode=int_mode))
                        except Exception:
                            pass
                    if not ints:
                        continue

                    # Mixed/polygonal: x n0 n1 ... nf c0 c1
                    # Fixed: n0 n1 n2 c0 c1 (tri) or n0 n1 n2 n3 c0 c1 (quad)
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
                        f"This usually indicates wrong integer parsing (--int_mode) or a malformed/unsupported .msh."
                    )

                continue

    if scan.scaling_factor is not None and abs(scan.scaling_factor - 1.0) > 1e-15:
        nodes_xyz = (nodes_xyz.astype(np.float64) * float(scan.scaling_factor)).astype(np.float32)

    faces_out: Dict[int, np.ndarray] = {z: np.array(f, dtype=np.int64) for z, f in faces_by_zone.items()}
    return nodes_xyz, faces_out, scan


# ======================================================================================
# Wall zone selection (robust)
# ======================================================================================

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
    mode: str  # "auto" or "manual"
    manual_zone_ids: Optional[List[int]] = None
    auto_min_faces_frac: float = 0.0


def select_wall_zones(scan: MeshHeaderScan, sel: WallSelection, logger: Logger) -> List[int]:
    """
    Decide which face zones represent the vessel wall surface.

    Manual:
      --wall_zone 19
      --wall_zone 19,21,22

    Auto (default):
      - choose face zones whose bc_type % 1000 == 3  (wall)
      - optionally drop tiny wall zones below auto_min_faces_frac
    """
    if sel.mode == "manual":
        if not sel.manual_zone_ids:
            raise ValueError("Wall selection mode is manual but no --wall_zone provided.")
        zones = [int(z) for z in sel.manual_zone_ids]
        logger.log(f"[WALL] Manual wall zones: {zones}")
        return zones

    wall_blocks = [b for b in scan.face_blocks if b.bc_type_base == 3]
    if not wall_blocks:
        bc_present = sorted(set(b.bc_type_base for b in scan.face_blocks))
        bc_present_named = [(bc, BC_TYPE_MAP.get(bc, "unknown")) for bc in bc_present]
        logger.log(f"[WALL][ERROR] No bc-type wall zones found (bc%1000==3). bc-types present: {bc_present_named}")
        logger.log("[WALL][HINT] Try --wall_mode manual --wall_zone <id> and/or --int_mode dec if indices are decimal.")
        return []

    total_faces = sum(b.n_faces for b in wall_blocks)
    zones_keep: List[int] = []
    for b in wall_blocks:
        frac = (b.n_faces / max(total_faces, 1)) if total_faces > 0 else 0.0
        if frac + 1e-12 >= sel.auto_min_faces_frac:
            zones_keep.append(b.zone_id)

    zones_keep = sorted(set(zones_keep))
    logger.log(f"[WALL] Auto wall zones by bc-type: {zones_keep}")
    for b in sorted(wall_blocks, key=lambda x: x.n_faces, reverse=True):
        frac = b.n_faces / max(total_faces, 1)
        logger.log(
            f"       zone={b.zone_id:>6d}  bc={b.bc_type} (base {b.bc_type_base}:{BC_TYPE_MAP.get(b.bc_type_base,'?')})"
            f"  face_type={b.face_type}  n_faces={b.n_faces}  frac={frac:.3f}"
        )

    dropped = sorted(set(bb.zone_id for bb in wall_blocks) - set(zones_keep))
    if dropped:
        logger.log(f"[WALL][WARN] Dropped small wall zones (auto_min_faces_frac={sel.auto_min_faces_frac}): {dropped}")

    return zones_keep


# ======================================================================================
# Graph construction (vertex wall graph)
# ======================================================================================

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
    """Area-weighted vertex normals from triangles."""
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


@dataclass
class GraphBuildInfo:
    Nw: int
    Fw: int
    Ew: int
    feature_dim: int
    wall_bbox_diag: float


def build_wall_graph_from_faces(
    nodes_xyz: np.ndarray,
    faces_by_zone: Dict[int, np.ndarray],
    wall_zone_ids: Sequence[int],
) -> Tuple[Data, GraphBuildInfo]:
    """
    Build a wall-only vertex graph from selected wall zones.

    Outputs a PyG Data with:
      pos: [Nw,3] wall vertex coordinates (float32)
      face: [3,Fw] triangles in local indexing
      edge_index: [2,E] directed edges (both directions)
      x: [Nw,7] features = [xyz_norm(3), normals(3), is_wall(1)]
      global_node_ids: [Nw] mapping local->global node index in nodes_xyz
    """
    if not wall_zone_ids:
        raise ValueError("No wall_zone_ids provided.")
    faces_list: List[np.ndarray] = []
    for z in wall_zone_ids:
        if z not in faces_by_zone:
            continue
        fz = faces_by_zone[z]
        if fz.size:
            faces_list.append(fz)
    if not faces_list:
        raise RuntimeError(f"No faces found for wall zones: {list(wall_zone_ids)}. Check wall selection.")

    faces_global = np.vstack(faces_list).astype(np.int64)

    wall_node_ids = np.unique(faces_global.reshape(-1))
    wall_node_ids_sorted = np.sort(wall_node_ids)

    global_to_local = -np.ones((nodes_xyz.shape[0],), dtype=np.int64)
    global_to_local[wall_node_ids_sorted] = np.arange(wall_node_ids_sorted.size, dtype=np.int64)

    faces_local = global_to_local[faces_global]
    wall_points = nodes_xyz[wall_node_ids_sorted].astype(np.float32)

    normals = compute_vertex_normals(wall_points, faces_local.astype(np.int64))
    xyz_norm = normalize_bbox(wall_points)
    is_wall = np.ones((wall_points.shape[0], 1), dtype=np.float32)
    x = np.concatenate([xyz_norm, normals, is_wall], axis=1).astype(np.float32)

    a, b, c = faces_local[:, 0], faces_local[:, 1], faces_local[:, 2]
    edges = np.vstack([
        np.stack([a, b], axis=1),
        np.stack([b, c], axis=1),
        np.stack([c, a], axis=1),
    ]).astype(np.int64)

    edges_undir = np.unique(np.sort(edges, axis=1), axis=0)
    edge_index = np.vstack([edges_undir, edges_undir[:, ::-1]]).T

    data = Data(
        pos=torch.from_numpy(wall_points).float(),
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        face=torch.from_numpy(faces_local.T.astype(np.int64)).long(),
        num_nodes=int(wall_points.shape[0]),
    )
    data.global_node_ids = torch.from_numpy(wall_node_ids_sorted.astype(np.int64))

    info = GraphBuildInfo(
        Nw=int(wall_points.shape[0]),
        Fw=int(faces_local.shape[0]),
        Ew=int(edge_index.shape[1]),
        feature_dim=int(x.shape[1]),
        wall_bbox_diag=float(bbox_diag(wall_points)),
    )
    return data, info


# ======================================================================================
# CSV parsing + alignment (Fluent wall export -> wall vertices)
# ======================================================================================

def try_read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return pd.read_csv(path)


def extract_wall_csv_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Flexible column picking for Fluent wall_data CSV.
    Required:
      - xyz: x/y/z coordinate columns
      - wss_vec3: x-wall-shear, y-wall-shear, z-wall-shear
    """
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        for n in names:
            k = n.lower()
            if k in cols:
                return cols[k]
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

    M = query_xyz.shape[0]
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


@dataclass
class AlignStats:
    wall_csv: str
    csv_rows: int
    n_wall_nodes: int
    tol_abs: float
    match_ok_frac: float
    unique_hit_frac: float
    match_mean_dist: float
    match_p95_dist: float
    match_p99_dist: float
    match_max_dist: float
    auto_scale: float
    diag_csv: float
    diag_wall: float
    diag_ratio_wall_over_csv: float


def choose_csv_scale(csv_xyz: np.ndarray, wall_xyz: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Decide whether to rescale csv_xyz (mm vs m mismatch) based on bbox diagonals.
    Explicit policy:
      - consider scale candidates: 1.0 and (diag_wall/diag_csv) if it's close to 1e-3 or 1e3 scale jump.
      - choose the candidate that minimizes p99 NN distance (quick NN query).
    """
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


def align_wall_csv_to_graph(base_graph: Data, wall_csv_path: Path, tol_abs: float) -> Tuple[torch.Tensor, torch.Tensor, AlignStats]:
    """
    Robust alignment:
      - reads CSV (xyz + wss_vec3)
      - chooses unit scale if needed (mm vs m)
      - nearest-neighbor map csv_xyz -> wall_xyz
      - aggregates ALL CSV rows per node (mean)
      - returns y [Nw,3] and y_mask [Nw,1] where mask=1 if node had >=1 hit
      - diagnostics (ok_frac, p95/p99/max, unique_hit_frac, scaling info)

    tol_abs is diagnostic only (ok_frac), default based on mesh bbox diag.
    """
    df = try_read_table(wall_csv_path)
    cols = extract_wall_csv_columns(df)

    csv_xyz = cols["xyz"].astype(np.float32)
    csv_wss = cols["wss_vec3"].astype(np.float32)

    wall_xyz = base_graph.pos.detach().cpu().numpy().astype(np.float32)

    csv_xyz2, scale_info = choose_csv_scale(csv_xyz, wall_xyz)
    nn_idx, nn_dist = nearest_neighbor_map(csv_xyz2, wall_xyz)

    ok = (nn_dist <= tol_abs)
    ok_frac = float(ok.mean()) if ok.size else 0.0

    Nw = wall_xyz.shape[0]
    sum_wss = np.zeros((Nw, 3), dtype=np.float64)
    cnt = np.zeros((Nw, 1), dtype=np.float64)

    np.add.at(sum_wss, nn_idx, csv_wss.astype(np.float64))
    np.add.at(cnt, nn_idx, 1.0)

    hit_mask = (cnt[:, 0] > 0).astype(np.float32).reshape(-1, 1)
    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    mean_wss = (sum_wss / cnt_safe).astype(np.float32)

    unique_hit = int(hit_mask.sum())
    unique_hit_frac = float(unique_hit / max(Nw, 1))

    if nn_dist.size:
        p95 = float(np.quantile(nn_dist, 0.95))
        p99 = float(np.quantile(nn_dist, 0.99))
        dmax = float(nn_dist.max())
        dmean = float(nn_dist.mean())
    else:
        p95 = p99 = dmax = dmean = float("nan")

    stats = AlignStats(
        wall_csv=str(wall_csv_path),
        csv_rows=int(csv_xyz.shape[0]),
        n_wall_nodes=int(Nw),
        tol_abs=float(tol_abs),
        match_ok_frac=float(ok_frac),
        unique_hit_frac=float(unique_hit_frac),
        match_mean_dist=float(dmean),
        match_p95_dist=float(p95),
        match_p99_dist=float(p99),
        match_max_dist=float(dmax),
        auto_scale=float(scale_info.get("auto_scale", 1.0)),
        diag_csv=float(scale_info.get("diag_csv", float("nan"))),
        diag_wall=float(scale_info.get("diag_wall", float("nan"))),
        diag_ratio_wall_over_csv=float(scale_info.get("diag_ratio_wall_over_csv", float("nan"))),
    )

    y = torch.from_numpy(mean_wss).float()
    y_mask = torch.from_numpy(hit_mask).float()
    return y, y_mask, stats


# ======================================================================================
# Results discovery (.csv per Re folder)
# ======================================================================================

_RE_DIR_RE = re.compile(r"^Re(\d+)$")


@dataclass
class ReCase:
    re_value: int
    re_dir: str
    wall_csv: str


def find_wall_csv(re_dir: Path, re_value: int) -> Path:
    candidates: List[Path] = []
    patterns = [
        f"wall_data_Re{re_value}.csv",
        f"wall_data_Re{re_value}.txt",
        f"wall_data_Re{re_value}",
        f"wall_data_Re{re_value}.*",
        f"wall_data_Re{re_value}*",
    ]
    for pat in patterns:
        candidates.extend(sorted(re_dir.glob(pat)))
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not find wall_data for Re{re_value} in: {re_dir}")


def discover_re_cases(results_root: Path, logger: Logger) -> List[ReCase]:
    if not results_root.exists():
        raise FileNotFoundError(f"RESULTS_ROOT not found: {results_root}")

    re_dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and _RE_DIR_RE.match(p.name)])
    logger.log(f"[DISCOVER] Found {len(re_dirs)} Re folders under {results_root}")

    cases: List[ReCase] = []
    for d in re_dirs:
        m = _RE_DIR_RE.match(d.name)
        if not m:
            continue
        re_val = int(m.group(1))
        try:
            csv_path = find_wall_csv(d, re_val)
            cases.append(ReCase(re_value=re_val, re_dir=str(d), wall_csv=str(csv_path)))
        except Exception as e:
            logger.log(f"[DISCOVER][WARN] {d.name}: {e}")

    cases = sorted(cases, key=lambda c: c.re_value)
    if cases:
        logger.log(f"[DISCOVER] Will process {len(cases)} cases: Re={cases[0].re_value}..Re={cases[-1].re_value}")
    else:
        logger.log("[DISCOVER][ERROR] No valid Re cases found (no wall_data_Re*.csv).")
    return cases


# ======================================================================================
# Preprocess end-to-end (one geometry)
# ======================================================================================

@dataclass
class PreprocessConfig:
    mesh_path: Path
    results_root: Path
    out_processed: Path

    # Wall extraction controls
    wall_mode: str = "auto"  # "auto" or "manual"
    wall_zone: Optional[List[int]] = None
    wall_auto_min_faces_frac: float = 0.0

    # Integer parsing in .msh
    int_mode: str = "hex"  # "hex" recommended

    # Alignment diagnostic tolerance
    tol_frac: float = 1e-4  # tol_abs = tol_frac * wall_bbox_diag

    # Save per-case meta json
    write_case_meta: bool = True

    # Save base graph (geometry-only) .pt for reuse
    save_base_graph: bool = True

    # Failure handling
    strict_wall: bool = False
    strict_align: bool = False

    # Alignment quality thresholds (warnings / strict)
    warn_ok_frac: float = 0.98
    warn_unique_hit_frac: float = 0.98


def preprocess_geometry(cfg: PreprocessConfig, logger: Logger) -> Tuple[List[Path], Dict[str, Any]]:
    mesh_path = Path(cfg.mesh_path)
    results_root = Path(cfg.results_root)
    out_processed = Path(cfg.out_processed)

    out_processed.mkdir(parents=True, exist_ok=True)

    mesh_sha1 = sha1_file(mesh_path)
    geom_id = mesh_path.stem

    logger.log("=" * 96)
    logger.log("PREPROCESS")
    logger.log("=" * 96)
    logger.log(f"MESH_PATH      : {mesh_path}")
    logger.log(f"MESH_SHA1      : {mesh_sha1}")
    logger.log(f"GEOM_ID        : {geom_id}")
    logger.log(f"RESULTS_ROOT   : {results_root}")
    logger.log(f"OUT_PROCESSED  : {out_processed}")
    logger.log(f"INT_MODE       : {cfg.int_mode}")
    logger.log(f"WALL_MODE      : {cfg.wall_mode}")
    logger.log(f"WALL_ZONE      : {cfg.wall_zone}")
    logger.log(f"TOL_FRAC       : {cfg.tol_frac}")
    logger.log("")

    cases = discover_re_cases(results_root, logger)
    if not cases:
        raise RuntimeError("No Re cases discovered; cannot preprocess.")

    scan = scan_fluent_msh_headers(mesh_path, int_mode=cfg.int_mode)

    logger.log(f"[MESH] scaling_factor={scan.scaling_factor}")
    if scan.node_decl:
        logger.log(f"[MESH] node_decl: first={scan.node_decl.first} last={scan.node_decl.last} n={scan.node_decl.n_nodes}")
    logger.log(f"[MESH] face_blocks: {len(scan.face_blocks)}")

    wall_sel = WallSelection(
        mode=cfg.wall_mode,
        manual_zone_ids=cfg.wall_zone,
        auto_min_faces_frac=cfg.wall_auto_min_faces_frac,
    )
    wall_zones = select_wall_zones(scan, wall_sel, logger)

    if not wall_zones:
        msg = "No wall zones selected."
        if cfg.strict_wall:
            raise RuntimeError(msg)
        logger.log(f"[WALL][WARN] {msg} Proceeding without preprocessing (no .pt will be saved).")
        return [], {"error": msg}

    nodes_xyz, faces_by_zone, scan2 = parse_fluent_msh_nodes_and_faces(
        msh_path=mesh_path,
        wanted_face_zones=wall_zones,
        int_mode=cfg.int_mode,
    )

    base_graph, ginfo = build_wall_graph_from_faces(nodes_xyz, faces_by_zone, wall_zones)
    tol_abs = float(cfg.tol_frac) * float(ginfo.wall_bbox_diag)

    logger.log("[GRAPH] Built base wall graph")
    logger.log(f"        wall_zones      : {wall_zones}")
    logger.log(f"        nodes (wall)    : {ginfo.Nw}")
    logger.log(f"        faces (tri)     : {ginfo.Fw}")
    logger.log(f"        edges (directed): {ginfo.Ew}")
    logger.log(f"        feature_dim     : {ginfo.feature_dim}")
    logger.log(f"        bbox_diag (m)   : {ginfo.wall_bbox_diag:.6e}  -> tol_abs={tol_abs:.6e}")
    logger.log("")

    base_graph_path = out_processed / "base_wall_graph.pt"
    if cfg.save_base_graph:
        bg = base_graph.cpu()
        bg.feature_version = FEATURE_VERSION
        bg.pipeline_version = PIPELINE_VERSION
        bg.mesh_sha1 = mesh_sha1
        bg.mesh_path = str(mesh_path)
        bg.geom_id = geom_id
        bg.wall_zone_ids = wall_zones
        torch.save(bg, base_graph_path)
        logger.log(f"[SAVE] base wall graph: {base_graph_path}")

    pt_paths: List[Path] = []
    case_meta: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for c in cases:
        re_val = int(c.re_value)
        wall_csv = Path(c.wall_csv)

        try:
            y, y_mask, stats = align_wall_csv_to_graph(
                base_graph=base_graph,
                wall_csv_path=wall_csv,
                tol_abs=tol_abs,
            )

            data = Data(
                pos=base_graph.pos.clone(),
                x=base_graph.x.clone(),
                edge_index=base_graph.edge_index.clone(),
                face=base_graph.face.clone(),
                num_nodes=base_graph.num_nodes,
            )
            data.global_node_ids = base_graph.global_node_ids.clone()
            data.re = torch.tensor(float(re_val), dtype=torch.float32)
            data.y = y
            data.y_mask = y_mask

            data.feature_version = FEATURE_VERSION
            data.target_version = TARGET_VERSION
            data.pipeline_version = PIPELINE_VERSION
            data.mesh_sha1 = mesh_sha1
            data.mesh_path = str(mesh_path)
            data.geom_id = geom_id
            data.wall_zone_ids = wall_zones

            out_pt = out_processed / f"Re{re_val}.pt"
            torch.save(data.cpu(), out_pt)

            pt_paths.append(out_pt)

            d_stats = dataclasses.asdict(stats)
            d_stats.update({"re": re_val, "out_pt": str(out_pt), "wall_zone_ids": wall_zones, "geom_id": geom_id})

            warn_bits: List[str] = []
            if stats.match_ok_frac < cfg.warn_ok_frac:
                warn_bits.append(f"ok_frac={stats.match_ok_frac:.3f} (<{cfg.warn_ok_frac})")
            if stats.unique_hit_frac < cfg.warn_unique_hit_frac:
                warn_bits.append(f"unique_hit_frac={stats.unique_hit_frac:.3f} (<{cfg.warn_unique_hit_frac})")
            if stats.auto_scale != 1.0:
                warn_bits.append(f"auto_scale={stats.auto_scale}")

            warn_txt = (" | WARN: " + ", ".join(warn_bits)) if warn_bits else ""

            logger.log(
                f"[OK] Re{re_val}: saved {out_pt.name} | y={list(y.shape)} mask_hit={float(y_mask.mean()):.3f} "
                f"| p99={stats.match_p99_dist:.2e} max={stats.match_max_dist:.2e}{warn_txt}"
            )

            if cfg.write_case_meta:
                meta_path = out_processed / f"Re{re_val}_meta.json"
                write_json(meta_path, d_stats)

            if cfg.strict_align and warn_bits:
                raise RuntimeError("Alignment below thresholds in strict_align mode: " + ", ".join(warn_bits))

            case_meta.append(d_stats)

        except Exception as e:
            logger.log(f"[FAIL] Re{re_val}: {e}")
            failures.append({"re": re_val, "wall_csv": str(wall_csv), "error": str(e)})

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "feature_version": FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "mesh_path": str(mesh_path),
        "mesh_sha1": mesh_sha1,
        "geom_id": geom_id,
        "results_root": str(results_root),
        "out_processed": str(out_processed),
        "int_mode": cfg.int_mode,
        "wall_mode": cfg.wall_mode,
        "wall_zone_ids": wall_zones,
        "scaling_factor": scan.scaling_factor,
        "graph": dataclasses.asdict(ginfo),
        "tol_frac": cfg.tol_frac,
        "tol_abs": tol_abs,
        "cases": case_meta,
        "failures": failures,
        "base_graph_pt": str(base_graph_path) if cfg.save_base_graph else None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_processed / "processed_manifest.json", manifest)

    if failures:
        logger.log(f"[PREPROCESS][WARN] {len(failures)} failures. See processed_manifest.json")
    else:
        logger.log("[PREPROCESS] All cases processed successfully.")

    return pt_paths, manifest


# ======================================================================================
# Training model (moderate complexity, easy to debug, no torch_scatter needed)
# ======================================================================================

class GraphMP(nn.Module):
    """
    Simple message passing without torch_scatter:
      h' = W_self h + W_nei mean(h_neighbors)
    """
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


# ======================================================================================
# Training utilities
# ======================================================================================

@dataclass
class TrainConfig:
    data_dir: Path = Path(r"C:\\Users\\radie\\Desktop\\2_trainingprocess\\ProcessedData\\coronary_extracted_vessel")
    out_dir: Path = Path(r"C:\\Users\\radie\\Desktop\\2_trainingprocess\\TRAIN_OUT")
    ckpt_name: str = "best_model.pt"

    # Split
    split_seed: int = 42
    split_by_geometry: bool = True  # if multiple geometries are present, group split to avoid leakage
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Training hyperparams
    epochs: int = 400
    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # Early stopping
    patience: int = 40
    min_delta: float = 1e-6

    # Model config
    hidden_dim: int = 128
    num_layers: int = 6
    dropout: float = 0.10

    # Conditioning
    append_re_to_x: bool = True

    # Target normalization
    normalize_y: bool = True

    # Training inclusion filters
    use_quality_filter: bool = True
    min_ok_frac: float = 0.95
    min_unique_hit_frac: float = 0.95

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def list_graph_files(data_dir: Path) -> List[Path]:
    files = sorted([Path(p) for p in glob.glob(str(data_dir / "Re*.pt"))])
    if not files:
        raise FileNotFoundError(f"No Re*.pt files found in: {data_dir}")
    return files


def load_graphs(files: List[Path]) -> List[Data]:
    graphs: List[Data] = []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(d, Data):
            raise TypeError(f"{p.name} did not load as torch_geometric.data.Data")
        for attr in ["y", "re", "x", "edge_index", "pos"]:
            if not hasattr(d, attr):
                raise ValueError(f"{p.name} missing required field: {attr}")
        if not hasattr(d, "y_mask"):
            d.y_mask = torch.ones((d.num_nodes, 1), dtype=torch.float32)
        graphs.append(d)
    return graphs


def compute_re_stats(graphs: List[Data]) -> Tuple[float, float]:
    re_vals = np.array([float(g.re.item() if torch.is_tensor(g.re) else g.re) for g in graphs], dtype=np.float64)
    mu = float(re_vals.mean())
    sd = float(re_vals.std() + 1e-12)
    return mu, sd


def compute_y_stats(graphs: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = []
    for g in graphs:
        y = g.y.reshape(-1, g.y.shape[-1]).float()
        m = g.y_mask.reshape(-1, 1).float()
        keep = (m[:, 0] > 0.5)
        if keep.any():
            ys.append(y[keep])
    if not ys:
        raise RuntimeError("No labeled nodes found across training graphs (y_mask all zero).")
    Y = torch.cat(ys, dim=0)
    mu = Y.mean(dim=0)
    sd = Y.std(dim=0) + 1e-12
    return mu, sd


def attach_re_feature(data: Data, re_mu: float, re_sd: float) -> Data:
    re_val = float(data.re.item() if torch.is_tensor(data.re) else data.re)
    re_norm = (re_val - re_mu) / (re_sd if re_sd > 1e-12 else 1.0)
    re_feat = torch.full((data.num_nodes, 1), float(re_norm), dtype=torch.float32)
    data.x = torch.cat([data.x.float(), re_feat], dim=1)
    return data


def normalize_y(data: Data, y_mu: torch.Tensor, y_sd: torch.Tensor) -> Data:
    data.y = (data.y.float() - y_mu.view(1, -1)) / y_sd.view(1, -1)
    return data


def split_indices(n: int, train_frac: float, val_frac: float, test_frac: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    assert len(test_idx) == n_test
    return train_idx, val_idx, test_idx


def mse_masked(pred: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if y_mask.dim() == 2 and y_mask.shape[1] == 1:
        mask = y_mask.expand_as(y)
    else:
        mask = y_mask
    diff2 = (pred - y) ** 2 * mask
    if reduction == "sum":
        return diff2.sum()
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, y_mu: torch.Tensor, y_sd: torch.Tensor, normalized_y: bool) -> Dict[str, float]:
    model.eval()
    total_mse_norm = 0.0
    total_mse_phys = 0.0
    total_mae_mag_phys = 0.0
    total_count = 0.0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.float()
        m = batch.y_mask.float()

        mse_n = float(mse_masked(pred, y, m, reduction="sum").item())
        count = float(m.expand_as(y).sum().item())
        total_mse_norm += mse_n
        total_count += count

        if normalized_y:
            pred_phys = pred * y_sd.view(1, -1).to(device) + y_mu.view(1, -1).to(device)
            y_phys = y * y_sd.view(1, -1).to(device) + y_mu.view(1, -1).to(device)
        else:
            pred_phys = pred
            y_phys = y

        mse_p = float(mse_masked(pred_phys, y_phys, m, reduction="sum").item())
        total_mse_phys += mse_p

        pred_mag = torch.linalg.norm(pred_phys, dim=1, keepdim=True)
        y_mag = torch.linalg.norm(y_phys, dim=1, keepdim=True)
        mae_mag = (torch.abs(pred_mag - y_mag) * m).sum().item()
        total_mae_mag_phys += float(mae_mag)

    denom = max(total_count, 1.0)
    return {
        "mse_norm": total_mse_norm / denom,
        "mse_phys": total_mse_phys / denom,
        "mae_mag_phys": total_mae_mag_phys / max(total_count / 3.0, 1.0),
        "count": denom,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        y = batch.y.float()
        m = batch.y_mask.float()

        loss = mse_masked(pred, y, m, reduction="mean")
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        loss_sum = float(mse_masked(pred, y, m, reduction="sum").item())
        count = float(m.expand_as(y).sum().item())
        total_loss += loss_sum
        total_count += count

    return total_loss / max(total_count, 1.0)


# ======================================================================================
# Training end-to-end
# ======================================================================================

def train_from_processed(cfg: TrainConfig, logger: Logger) -> Dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 96)
    logger.log("TRAIN")
    logger.log("=" * 96)
    logger.log(f"DATA_DIR    : {data_dir}")
    logger.log(f"OUT_DIR     : {out_dir}")
    logger.log(f"DEVICE      : {cfg.device}")
    logger.log(f"EPOCHS      : {cfg.epochs}")
    logger.log(f"BATCH_SIZE  : {cfg.batch_size}")
    logger.log(f"APPEND_RE   : {cfg.append_re_to_x}")
    logger.log(f"NORMALIZE_Y : {cfg.normalize_y}")
    logger.log("")

    files = list_graph_files(data_dir)
    graphs = load_graphs(files)
    n = len(graphs)

    include_mask = np.ones((n,), dtype=bool)
    excluded: List[Dict[str, Any]] = []

    if cfg.use_quality_filter:
        for i, p in enumerate(files):
            meta_path = data_dir / f"{p.stem}_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ok_frac = float(meta.get("match_ok_frac", 1.0))
                uhf = float(meta.get("unique_hit_frac", 1.0))
                if ok_frac < cfg.min_ok_frac or uhf < cfg.min_unique_hit_frac:
                    include_mask[i] = False
                    excluded.append({"file": p.name, "ok_frac": ok_frac, "unique_hit_frac": uhf, "reason": "quality_filter"})
            except Exception:
                continue

    kept_files = [p for i, p in enumerate(files) if include_mask[i]]
    kept_graphs = [g for i, g in enumerate(graphs) if include_mask[i]]

    if excluded:
        logger.log(f"[FILTER] Excluding {len(excluded)} / {n} graphs from training by quality thresholds:")
        for ex in excluded:
            logger.log(f"        {ex['file']}: ok_frac={ex['ok_frac']:.3f} unique_hit_frac={ex['unique_hit_frac']:.3f}")
    else:
        logger.log("[FILTER] No graphs excluded by quality filter.")

    if len(kept_graphs) < 3:
        logger.log("[WARN] Very small dataset after filtering; training/val/test split may be degenerate.")

    # Split
    # Contract:
    #   - If there are multiple geometries present and split_by_geometry=True,
    #     group split so all Re cases from same geometry stay together (avoids leakage).
    #   - If there is only one geometry, split by Re-cases (graphs).

    def get_geom_key(g: Data) -> str:
        if hasattr(g, "geom_id"):
            return str(getattr(g, "geom_id"))
        if hasattr(g, "mesh_sha1"):
            return str(getattr(g, "mesh_sha1"))
        return "geom0"

    groups: Dict[str, List[int]] = {}
    for i, g in enumerate(kept_graphs):
        groups.setdefault(get_geom_key(g), []).append(i)

    geom_keys = sorted(groups.keys())
    if cfg.split_by_geometry and len(geom_keys) > 1:
        logger.log(f"[SPLIT] Detected {len(geom_keys)} geometries; performing geometry-group split (no leakage).")
        rng = random.Random(cfg.split_seed)
        rng.shuffle(geom_keys)

        nG = len(geom_keys)
        n_trainG = int(round(cfg.train_frac * nG))
        n_valG = int(round(cfg.val_frac * nG))
        n_trainG = min(n_trainG, nG)
        n_valG = min(n_valG, nG - n_trainG)
        train_geoms = set(geom_keys[:n_trainG])
        val_geoms = set(geom_keys[n_trainG:n_trainG + n_valG])
        test_geoms = set(geom_keys[n_trainG + n_valG:])

        train_idx = [i for gk in train_geoms for i in groups[gk]]
        val_idx = [i for gk in val_geoms for i in groups[gk]]
        test_idx = [i for gk in test_geoms for i in groups[gk]]

        logger.log(f"[SPLIT] train_geoms={len(train_geoms)} val_geoms={len(val_geoms)} test_geoms={len(test_geoms)}")
    else:
        if len(geom_keys) > 1 and not cfg.split_by_geometry:
            logger.log(f"[SPLIT][WARN] Multiple geometries detected ({len(geom_keys)}) but split_by_geometry=False; splitting by Re-cases may leak geometry.")
        train_idx, val_idx, test_idx = split_indices(len(kept_graphs), cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.split_seed)

    train_graphs = [kept_graphs[i] for i in train_idx]
    val_graphs = [kept_graphs[i] for i in val_idx]
    test_graphs = [kept_graphs[i] for i in test_idx]

    logger.log(f"SPLIT: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")
    logger.log(f"Train idx (kept): {sorted(train_idx)}")
    logger.log(f"Val idx   (kept): {sorted(val_idx)}")
    logger.log(f"Test idx  (kept): {sorted(test_idx)}")
    logger.log("")

    re_mu, re_sd = compute_re_stats(train_graphs)
    logger.log(f"Re stats (train): mean={re_mu:.4f}, std={re_sd:.4f}")

    if cfg.normalize_y:
        y_mu, y_sd = compute_y_stats(train_graphs)
        logger.log(f"Y stats (train): mean={y_mu.tolist()}, std={y_sd.tolist()}")
    else:
        y_mu = torch.zeros((train_graphs[0].y.shape[-1],), dtype=torch.float32)
        y_sd = torch.ones((train_graphs[0].y.shape[-1],), dtype=torch.float32)

    def prep(gs: List[Data]) -> List[Data]:
        out: List[Data] = []
        for g in gs:
            gg = g.clone()
            if cfg.append_re_to_x:
                gg = attach_re_feature(gg, re_mu, re_sd)
            if cfg.normalize_y:
                gg = normalize_y(gg, y_mu, y_sd)
            out.append(gg)
        return out

    train_graphs = prep(train_graphs)
    val_graphs = prep(val_graphs)
    test_graphs = prep(test_graphs)

    in_dim = int(train_graphs[0].x.shape[1])
    out_dim = int(train_graphs[0].y.shape[1])

    logger.log(f"MODEL IO: in_dim={in_dim} -> out_dim={out_dim}")
    logger.log(f"Model: hidden={cfg.hidden_dim} layers={cfg.num_layers} dropout={cfg.dropout}")
    logger.log("")

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size, shuffle=False) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=cfg.batch_size, shuffle=False) if test_graphs else None

    model = WSSNet(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    ckpt_path = out_dir / cfg.ckpt_name

    for epoch in range(1, cfg.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, cfg.device, cfg.grad_clip_norm)

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, cfg.device, y_mu, y_sd, normalized_y=cfg.normalize_y)
            val_mse = val_metrics["mse_norm"]
        else:
            val_metrics = {"mse_norm": train_mse, "mse_phys": float("nan"), "mae_mag_phys": float("nan"), "count": 0.0}
            val_mse = train_mse

        history.append({
            "epoch": epoch,
            "train_mse_norm": train_mse,
            "val_mse_norm": float(val_mse),
            "val_mse_phys": float(val_metrics.get("mse_phys", float("nan"))),
            "val_mae_mag_phys": float(val_metrics.get("mae_mag_phys", float("nan"))),
        })

        improved = (best_val - val_mse) > cfg.min_delta
        if improved:
            best_val = float(val_mse)
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "pipeline_version": PIPELINE_VERSION,
                "feature_version": FEATURE_VERSION,
                "target_version": TARGET_VERSION,
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "append_re_to_x": cfg.append_re_to_x,
                "normalize_y": cfg.normalize_y,
                "re_mu": re_mu,
                "re_sd": re_sd,
                "y_mu": y_mu.cpu(),
                "y_sd": y_sd.cpu(),
                "best_epoch": best_epoch,
                "best_val_mse_norm": best_val,
                "train_cfg": dataclasses.asdict(cfg),
                "files_all": [p.name for p in files],
                "files_kept": [p.name for p in kept_files],
                "excluded": excluded,
                "split": {
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                    "seed": cfg.split_seed,
                    "split_by_geometry": cfg.split_by_geometry,
                },
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            torch.save(ckpt, ckpt_path)
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            tag = " *" if improved else ""
            logger.log(
                f"Epoch {epoch:4d} | train_mse_norm={train_mse:.6e} | val_mse_norm={val_mse:.6e} "
                f"| val_mse_phys={val_metrics.get('mse_phys', float('nan')):.6e}{tag}"
            )

        if bad_epochs >= cfg.patience:
            logger.log(f"[EARLY STOP] No improvement for {cfg.patience} epochs. Best epoch={best_epoch} val={best_val:.6e}")
            break

    test_report: Dict[str, Any] = {}
    if test_loader is not None and test_graphs:
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        test_metrics = evaluate(model, test_loader, cfg.device, ckpt["y_mu"], ckpt["y_sd"], normalized_y=cfg.normalize_y)
        test_report = test_metrics

        logger.log("=" * 96)
        logger.log("TEST EVAL (best checkpoint)")
        logger.log("=" * 96)
        logger.log(f"Test MSE (normalized space): {test_metrics['mse_norm']:.6e}")
        logger.log(f"Test MSE (physical units)  : {test_metrics['mse_phys']:.6e}")
        logger.log(f"Test MAE(|WSS|) phys       : {test_metrics['mae_mag_phys']:.6e}")

    hist_path = out_dir / "train_history.json"
    write_json(hist_path, history)

    meta = {
        "pipeline_version": PIPELINE_VERSION,
        "feature_version": FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "ckpt_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_mse_norm": best_val,
        "test": test_report if test_report else None,
        "train_cfg": dataclasses.asdict(cfg),
        "excluded": excluded,
        "files_kept": [p.name for p in kept_files],
    }
    write_json(out_dir / "train_meta.json", meta)

    logger.log(f"[OK] Saved checkpoint: {ckpt_path}")
    logger.log(f"[OK] Saved history   : {hist_path}")
    logger.log(f"[OK] Saved meta      : {out_dir / 'train_meta.json'}")

    return meta


# ======================================================================================
# CLI / entrypoint
# ======================================================================================

def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    parts = re.split(r"[,\s]+", str(s).strip())
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-file Fluent(.msh)+wall CSV -> PyG graphs -> GNN training pipeline (WSS vector)."
    )

    ap.add_argument("--stage", default="all", choices=["all", "preprocess", "train", "dryrun"], help="Which stage to run.")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed.")

    # Preprocess inputs
    ap.add_argument("--mesh", type=str, required=True, help="Path to Fluent legacy ASCII .msh")
    ap.add_argument("--results_root", type=str, required=True, help="Folder containing Re### subfolders with wall_data_Re*.csv")
    ap.add_argument("--out_processed", type=str, required=True, help="Output folder for processed Re*.pt files")

    # Wall selection
    ap.add_argument("--wall_mode", default="auto", choices=["auto", "manual"], help="Wall selection mode.")
    ap.add_argument("--wall_zone", default=None, type=str, help="Manual wall zone IDs (comma/space-separated), e.g. '19' or '19,21'")
    ap.add_argument("--wall_auto_min_faces_frac", type=float, default=0.0, help="Auto: drop wall patches below this fraction of wall faces.")

    # Mesh int parsing
    ap.add_argument("--int_mode", default="hex", choices=["hex", "dec", "auto"], help="Integer parsing mode for Fluent indices.")

    # Alignment tolerance
    ap.add_argument("--tol_frac", type=float, default=1e-4, help="Diagnostic tolerance fraction of wall bbox diag for ok_frac.")

    ap.add_argument("--strict_wall", action="store_true", help="Fail if wall zones cannot be identified.")
    ap.add_argument("--strict_align", action="store_true", help="Fail if any Re case alignment falls below warn thresholds.")
    ap.add_argument("--warn_ok_frac", type=float, default=0.98, help="Warn threshold for ok_frac.")
    ap.add_argument("--warn_unique_hit_frac", type=float, default=0.98, help="Warn threshold for unique_hit_frac.")

    ap.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing, train from existing out_processed/Re*.pt")
    ap.add_argument("--skip_train", action="store_true", help="Skip training (preprocess only).")
    ap.add_argument("--log", type=str, default=None, help="Optional log file path.")

    # Training args
    ap.add_argument("--out_train", type=str, required=True, help="Training output directory (checkpoints, logs)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--split_by_geometry", action="store_true", help="If multiple geometries are present, split by geometry (recommended).")
    ap.add_argument("--no_split_by_geometry", action="store_true", help="Force split by individual Re cases even if multiple geometries are present.")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--append_re", action="store_true", help="Append Re as node feature (recommended).")
    ap.add_argument("--no_append_re", action="store_true", help="Do NOT append Re to node feature.")
    ap.add_argument("--normalize_y", action="store_true", help="Normalize y (recommended).")
    ap.add_argument("--no_normalize_y", action="store_true", help="Do NOT normalize y.")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (default auto).")

    # Training quality filter
    ap.add_argument("--use_quality_filter", action="store_true", help="Exclude low-quality aligned cases from training.")
    ap.add_argument("--min_ok_frac", type=float, default=0.95)
    ap.add_argument("--min_unique_hit_frac", type=float, default=0.95)

    args = ap.parse_args()

    set_global_seed(int(args.seed))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = Logger(Path(args.log) if args.log else None)
    try:
        mesh_path = Path(args.mesh)
        results_root = Path(args.results_root)
        out_processed = Path(args.out_processed)
        out_train = Path(args.out_train)

        wall_zone_ids = parse_int_list(args.wall_zone) if args.wall_zone is not None else None

        preprocess_cfg = PreprocessConfig(
            mesh_path=mesh_path,
            results_root=results_root,
            out_processed=out_processed,
            wall_mode=args.wall_mode,
            wall_zone=wall_zone_ids,
            wall_auto_min_faces_frac=float(args.wall_auto_min_faces_frac),
            int_mode=args.int_mode,
            tol_frac=float(args.tol_frac),
            strict_wall=bool(args.strict_wall),
            strict_align=bool(args.strict_align),
            warn_ok_frac=float(args.warn_ok_frac),
            warn_unique_hit_frac=float(args.warn_unique_hit_frac),
        )

        append_re = True
        if args.no_append_re:
            append_re = False
        elif args.append_re:
            append_re = True

        normalize_y = True
        if args.no_normalize_y:
            normalize_y = False
        elif args.normalize_y:
            normalize_y = True

        split_by_geom = True
        if args.no_split_by_geometry:
            split_by_geom = False
        elif args.split_by_geometry:
            split_by_geom = True

        train_cfg = TrainConfig(
            data_dir=out_processed,
            out_dir=out_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip),
            patience=int(args.patience),
            split_by_geometry=bool(split_by_geom),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            append_re_to_x=bool(append_re),
            normalize_y=bool(normalize_y),
            device=str(device),
            use_quality_filter=bool(args.use_quality_filter),
            min_ok_frac=float(args.min_ok_frac),
            min_unique_hit_frac=float(args.min_unique_hit_frac),
        )

        stage = args.stage.lower()

        if stage in ("all", "preprocess", "dryrun"):
            if args.skip_preprocess:
                logger.log("[SKIP] preprocessing skipped (using existing Re*.pt in out_processed)")
            else:
                preprocess_geometry(preprocess_cfg, logger)

            if stage == "dryrun":
                logger.log("[DRYRUN] Completed preprocessing diagnostics; exiting without training.")
                return

        if stage in ("all", "train") and not args.skip_train:
            train_from_processed(train_cfg, logger)

        logger.log("DONE.")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
