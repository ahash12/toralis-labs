#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align Mesh 2 inlet-facing direction to Mesh 1 inlet-facing direction (rigid rotation only, no remeshing).

Compatible with the Fluent legacy ASCII .msh parsing style used in your current training/inference pipeline.

What it does:
- Parses both meshes
- Auto-selects wall zones (bc_type_base == 3)
- Auto-selects one inlet zone (prefers velocity-inlet / mass-flow-inlet / pressure-inlet)
- Computes inlet-facing direction d = wall_centroid - inlet_centroid
- Rotates Mesh 2 so d2 aligns to d1
- Optionally translates inlet centroid of Mesh 2 to Mesh 1
- Exports transformed nodes + diagnostics

NOTE:
- This script transforms coordinates for ML/preprocessing use.
- It does NOT rewrite a Fluent .msh file (that is possible, but a separate step).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np


# =============================================================================
# User defaults (you can override via CLI)
# =============================================================================

DEFAULT_MESH1 = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model1\\coronary_extracted_vessel.msh"
DEFAULT_MESH2 = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model2\\coronary2.msh"
DEFAULT_OUTDIR = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model2\\alignment_out"


# =============================================================================
# Logging
# =============================================================================

class Logger:
    def __init__(self):
        pass

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")


# =============================================================================
# Fluent legacy ASCII .msh parsing (compatible style)
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

    # auto mode heuristic
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
    wanted_face_zones: Optional[Sequence[int]] = None,
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

            # Node sections
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
                        f"Node section for zone {zone} ended early (read up to node-id {nid-1}, expected {end}). "
                        f"This usually indicates wrong integer parsing (--int_mode) or malformed/unsupported .msh."
                    )
                continue

            # Face sections
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
                        verts = ints[1: 1 + nv]
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
# Zone selection helpers
# =============================================================================

def select_wall_zones(scan: MeshHeaderScan) -> List[int]:
    wall_blocks = [b for b in scan.face_blocks if b.bc_type_base == 3]
    return sorted(set(b.zone_id for b in wall_blocks))


def select_inlet_zone(scan: MeshHeaderScan, logger: Logger) -> int:
    """
    Auto-pick a single inlet zone.
    Prefers:
      10 velocity-inlet
      20 mass-flow-inlet
      4 pressure-inlet
    If multiple candidates, picks the one with largest face count.
    """
    priority = [10, 20, 4]
    candidates: List[FaceBlockMeta] = []

    for bc in priority:
        c = [b for b in scan.face_blocks if b.bc_type_base == bc]
        if c:
            candidates = c
            logger.log(f"[ZONE] Inlet candidates by bc_type_base={bc} ({BC_TYPE_MAP.get(bc,'?')}): "
                       + ", ".join([f"zone={x.zone_id} n_faces={x.n_faces}" for x in c]))
            break

    if not candidates:
        # Fallback: if exactly one non-wall boundary among common in/out types, use it
        fallback = [b for b in scan.face_blocks if b.bc_type_base in (4, 5, 10, 20, 36)]
        if not fallback:
            raise RuntimeError("Could not auto-detect inlet zone from bc types. Pass manual zone IDs in a future version.")
        candidates = fallback
        logger.log("[ZONE][WARN] Falling back to common boundary candidates.")

    # choose largest face count
    pick = sorted(candidates, key=lambda x: x.n_faces, reverse=True)[0]
    logger.log(f"[ZONE] Picked inlet zone = {pick.zone_id} (bc={pick.bc_type_base}:{BC_TYPE_MAP.get(pick.bc_type_base,'?')}, n_faces={pick.n_faces})")
    return int(pick.zone_id)


def face_block_summary(scan: MeshHeaderScan) -> List[Dict[str, Any]]:
    out = []
    for b in sorted(scan.face_blocks, key=lambda x: (x.bc_type_base, -x.n_faces, x.zone_id)):
        out.append({
            "zone_id": int(b.zone_id),
            "bc_type": int(b.bc_type),
            "bc_type_base": int(b.bc_type_base),
            "bc_name": BC_TYPE_MAP.get(b.bc_type_base, "unknown"),
            "face_type": int(b.face_type),
            "n_faces": int(b.n_faces),
            "header_line": int(b.header_line),
        })
    return out


# =============================================================================
# Geometry / alignment math
# =============================================================================

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def compute_patch_node_ids(faces_by_zone: Dict[int, np.ndarray], zone_id: int) -> np.ndarray:
    if zone_id not in faces_by_zone:
        raise KeyError(f"Zone {zone_id} not present in parsed faces_by_zone.")
    f = np.asarray(faces_by_zone[zone_id], dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] == 0:
        raise ValueError(f"Zone {zone_id} has invalid/empty face array shape: {f.shape}")
    return np.unique(f.reshape(-1))


def compute_patch_centroid_from_nodes(
    nodes_xyz: np.ndarray,
    faces_by_zone: Dict[int, np.ndarray],
    zone_id: int,
) -> np.ndarray:
    node_ids = compute_patch_node_ids(faces_by_zone, zone_id)
    pts = np.asarray(nodes_xyz, dtype=np.float64)[node_ids]
    if pts.shape[0] == 0:
        raise ValueError(f"No nodes found for zone {zone_id}.")
    return pts.mean(axis=0)


def compute_wall_node_ids_from_zones(
    faces_by_zone: Dict[int, np.ndarray],
    wall_zone_ids: Sequence[int],
) -> np.ndarray:
    all_faces = []
    for z in wall_zone_ids:
        if z in faces_by_zone and np.asarray(faces_by_zone[z]).size > 0:
            all_faces.append(np.asarray(faces_by_zone[z], dtype=np.int64))
    if not all_faces:
        raise ValueError("No faces found for provided wall_zone_ids.")
    f = np.vstack(all_faces)
    return np.unique(f.reshape(-1))


def compute_wall_centroid(
    nodes_xyz: np.ndarray,
    faces_by_zone: Dict[int, np.ndarray],
    wall_zone_ids: Sequence[int],
) -> np.ndarray:
    node_ids = compute_wall_node_ids_from_zones(faces_by_zone, wall_zone_ids)
    pts = np.asarray(nodes_xyz, dtype=np.float64)[node_ids]
    if pts.shape[0] == 0:
        raise ValueError("No wall nodes found.")
    return pts.mean(axis=0)


def compute_inlet_facing_direction(
    nodes_xyz: np.ndarray,
    faces_by_zone: Dict[int, np.ndarray],
    inlet_zone_id: int,
    wall_zone_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      d_inward_unit : unit vector pointing from inlet centroid toward wall centroid (approx inward vessel direction)
      Cin           : inlet patch centroid
      Cwall         : wall centroid used to define inward direction
    """
    Cin = compute_patch_centroid_from_nodes(nodes_xyz, faces_by_zone, inlet_zone_id)
    Cwall = compute_wall_centroid(nodes_xyz, faces_by_zone, wall_zone_ids)
    d = _normalize(Cwall - Cin)
    return d, Cin, Cwall


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def rotation_matrix_from_vec_to_vec(v_from: np.ndarray, v_to: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Proper rotation R such that R @ v_from ~= v_to.
    Handles parallel and anti-parallel cases.
    """
    a = _normalize(v_from, eps=eps)
    b = _normalize(v_to, eps=eps)

    c = float(np.dot(a, b))
    c = max(-1.0, min(1.0, c))

    # already aligned
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)

    # opposite
    if c < -1.0 + 1e-10:
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, trial), eps=eps)
        u = axis.reshape(3, 1)
        # 180° rotation
        return -np.eye(3, dtype=np.float64) + 2.0 * (u @ u.T)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s + eps))
    return R


def rotate_about_point(nodes_xyz: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    pts = np.asarray(nodes_xyz, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    out = (pts - c) @ np.asarray(R, dtype=np.float64).T + c
    return out.astype(np.float32)


def translate_nodes(nodes_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(nodes_xyz, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(1, 3)
    return (pts + t).astype(np.float32)


# =============================================================================
# Export helpers
# =============================================================================

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(x: Any):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.generic,)):
            return x.item()
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_nodes_csv(path: Path, nodes_xyz: np.ndarray, header: str = "x,y,z") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(nodes_xyz, dtype=np.float64), delimiter=",", header=header, comments="")


def save_wall_nodes_csv(path: Path, nodes_xyz: np.ndarray, faces_by_zone: Dict[int, np.ndarray], wall_zone_ids: Sequence[int]) -> None:
    wall_ids = compute_wall_node_ids_from_zones(faces_by_zone, wall_zone_ids)
    pts = np.asarray(nodes_xyz, dtype=np.float64)[wall_ids]
    mat = np.hstack([wall_ids.reshape(-1, 1).astype(np.float64), pts])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, delimiter=",", header="global_node_idx,x,y,z", comments="")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Align Mesh 2 inlet-facing direction to Mesh 1 (rigid transform, no remeshing).")
    ap.add_argument("--mesh1", type=str, default=DEFAULT_MESH1, help="Reference mesh (.msh)")
    ap.add_argument("--mesh2", type=str, default=DEFAULT_MESH2, help="Mesh to rotate (.msh)")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    ap.add_argument("--int_mode", type=str, default="hex", choices=["hex", "dec", "auto"], help="Fluent integer parsing mode")

    # Optional manual overrides (if auto inlet detection picks wrong zone)
    ap.add_argument("--mesh1_inlet_zone", type=int, default=None, help="Manual inlet zone for mesh1")
    ap.add_argument("--mesh2_inlet_zone", type=int, default=None, help="Manual inlet zone for mesh2")
    ap.add_argument("--mesh1_wall_zones", type=str, default=None, help="Comma-separated manual wall zones for mesh1")
    ap.add_argument("--mesh2_wall_zones", type=str, default=None, help="Comma-separated manual wall zones for mesh2")

    ap.add_argument("--translate_inlet_to_match", action="store_true", help="Also translate mesh2 so inlet centroid matches mesh1 inlet centroid")
    ap.add_argument("--save_full_nodes_csv", action="store_true", help="Save all transformed mesh2 node coordinates CSV")
    ap.add_argument("--save_wall_nodes_csv", action="store_true", help="Save transformed wall-only node coordinates CSV")
    args = ap.parse_args()

    logger = Logger()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh1_path = Path(args.mesh1)
    mesh2_path = Path(args.mesh2)

    logger.log("=" * 96)
    logger.log("INLET DIRECTION ALIGNMENT (Mesh2 -> Mesh1)")
    logger.log("=" * 96)
    logger.log(f"MESH1 (reference): {mesh1_path}")
    logger.log(f"MESH2 (to align):  {mesh2_path}")
    logger.log(f"OUT_DIR:           {out_dir}")
    logger.log(f"INT_MODE:          {args.int_mode}")

    # Scan headers first (fast) to inspect zones
    scan1 = scan_fluent_msh_headers(mesh1_path, int_mode=args.int_mode)
    scan2 = scan_fluent_msh_headers(mesh2_path, int_mode=args.int_mode)

    logger.log(f"[MESH1] scaling_factor={scan1.scaling_factor} face_blocks={len(scan1.face_blocks)}")
    logger.log(f"[MESH2] scaling_factor={scan2.scaling_factor} face_blocks={len(scan2.face_blocks)}")

    # Select zones
    wall1 = [int(x) for x in args.mesh1_wall_zones.split(",")] if args.mesh1_wall_zones else select_wall_zones(scan1)
    wall2 = [int(x) for x in args.mesh2_wall_zones.split(",")] if args.mesh2_wall_zones else select_wall_zones(scan2)
    inlet1 = int(args.mesh1_inlet_zone) if args.mesh1_inlet_zone is not None else select_inlet_zone(scan1, logger)
    inlet2 = int(args.mesh2_inlet_zone) if args.mesh2_inlet_zone is not None else select_inlet_zone(scan2, logger)

    logger.log(f"[MESH1] wall zones: {wall1}")
    logger.log(f"[MESH1] inlet zone: {inlet1}")
    logger.log(f"[MESH2] wall zones: {wall2}")
    logger.log(f"[MESH2] inlet zone: {inlet2}")

    # Parse full nodes + selected faces needed for centroid calculations
    wanted1 = sorted(set(wall1 + [inlet1]))
    wanted2 = sorted(set(wall2 + [inlet2]))

    nodes1, faces1, _ = parse_fluent_msh_nodes_and_faces(mesh1_path, wanted_face_zones=wanted1, int_mode=args.int_mode)
    nodes2, faces2, _ = parse_fluent_msh_nodes_and_faces(mesh2_path, wanted_face_zones=wanted2, int_mode=args.int_mode)

    # Compute directions
    d1, Cin1, Cwall1 = compute_inlet_facing_direction(nodes1, faces1, inlet1, wall1)
    d2, Cin2, Cwall2 = compute_inlet_facing_direction(nodes2, faces2, inlet2, wall2)

    logger.log(f"[MESH1] inlet centroid = {Cin1}")
    logger.log(f"[MESH1] wall centroid  = {Cwall1}")
    logger.log(f"[MESH1] d1 (target)    = {d1}")

    logger.log(f"[MESH2] inlet centroid = {Cin2}")
    logger.log(f"[MESH2] wall centroid  = {Cwall2}")
    logger.log(f"[MESH2] d2 (before)    = {d2}")

    # Rotation d2 -> d1
    R = rotation_matrix_from_vec_to_vec(d2, d1)
    det_R = float(np.linalg.det(R))

    nodes2_rot = rotate_about_point(nodes2, R, Cin2)

    if args.translate_inlet_to_match:
        # Recompute inlet centroid after rotation (should be ~same as Cin2 if rotated about Cin2, but recompute for safety)
        Cin2_after_rot = compute_patch_centroid_from_nodes(nodes2_rot, faces2, inlet2)
        t = (Cin1 - Cin2_after_rot).reshape(3)
        nodes2_aligned = translate_nodes(nodes2_rot, t)
    else:
        t = np.zeros(3, dtype=np.float64)
        nodes2_aligned = nodes2_rot

    d2_after, Cin2_after, Cwall2_after = compute_inlet_facing_direction(nodes2_aligned, faces2, inlet2, wall2)
    cosang = float(np.clip(np.dot(_normalize(d1), _normalize(d2_after)), -1.0, 1.0))
    ang_deg = float(np.degrees(np.arccos(cosang)))

    logger.log(f"[ALIGN] det(R) = {det_R:.12g} (should be ~ +1)")
    logger.log(f"[ALIGN] d2_after = {d2_after}")
    logger.log(f"[ALIGN] angle(d1, d2_after) = {ang_deg:.6f} deg")
    logger.log(f"[ALIGN] translation t = {t} (only nonzero if --translate_inlet_to_match)")

    # Save outputs
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"mesh2_aligned_to_mesh1_{ts_tag}"

    if args.save_full_nodes_csv:
        save_nodes_csv(base.with_suffix(".nodes.csv"), nodes2_aligned)
        logger.log(f"[OK] Wrote full transformed mesh2 nodes CSV: {base.with_suffix('.nodes.csv')}")

    if args.save_wall_nodes_csv:
        save_wall_nodes_csv(base.with_suffix(".wall_nodes.csv"), nodes2_aligned, faces2, wall2)
        logger.log(f"[OK] Wrote transformed wall-only nodes CSV: {base.with_suffix('.wall_nodes.csv')}")

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mesh1_path": str(mesh1_path),
        "mesh2_path": str(mesh2_path),
        "int_mode": args.int_mode,
        "mesh1_zone_summary": face_block_summary(scan1),
        "mesh2_zone_summary": face_block_summary(scan2),
        "mesh1_selected": {"inlet_zone": int(inlet1), "wall_zones": [int(z) for z in wall1]},
        "mesh2_selected": {"inlet_zone": int(inlet2), "wall_zones": [int(z) for z in wall2]},
        "transform": {
            "R": R,
            "t": t,
            "det_R": det_R,
            "translate_inlet_to_match": bool(args.translate_inlet_to_match),
        },
        "diagnostics": {
            "Cin1": Cin1,
            "Cwall1": Cwall1,
            "d1_target": d1,
            "Cin2_before": Cin2,
            "Cwall2_before": Cwall2,
            "d2_before": d2,
            "Cin2_after": Cin2_after,
            "Cwall2_after": Cwall2_after,
            "d2_after": d2_after,
            "angle_deg_after": ang_deg,
        },
        "output_files": {
            "full_nodes_csv": str(base.with_suffix(".nodes.csv")) if args.save_full_nodes_csv else None,
            "wall_nodes_csv": str(base.with_suffix(".wall_nodes.csv")) if args.save_wall_nodes_csv else None,
        },
    }
    meta_path = base.with_suffix(".meta.json")
    write_json(meta_path, meta)
    logger.log(f"[OK] Wrote alignment metadata JSON: {meta_path}")

    # Also save a compact NPZ for easy programmatic reuse
    npz_path = base.with_suffix(".transform.npz")
    np.savez(
        npz_path,
        R=R.astype(np.float64),
        t=np.asarray(t, dtype=np.float64),
        Cin1=Cin1.astype(np.float64),
        Cin2_before=Cin2.astype(np.float64),
        Cin2_after=Cin2_after.astype(np.float64),
        d1=d1.astype(np.float64),
        d2_before=d2.astype(np.float64),
        d2_after=d2_after.astype(np.float64),
        angle_deg_after=np.array([ang_deg], dtype=np.float64),
        det_R=np.array([det_R], dtype=np.float64),
    )
    logger.log(f"[OK] Wrote transform NPZ: {npz_path}")

    logger.log("DONE.")
    logger.log("")
    logger.log("Next step (for ML pipeline integration):")
    logger.log("- Apply the same rotation (and translation if used) to Mesh2 coordinates before graph feature construction.")
    logger.log("- If validating against Fluent CSV in the original frame, rotate CSV coordinates/vectors too OR inverse-rotate predictions back.")


if __name__ == "__main__":
    main()