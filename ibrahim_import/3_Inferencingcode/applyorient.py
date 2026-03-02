#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rewrite a Fluent legacy ASCII .msh by applying a rigid transform (R, t) to ALL node coordinates.

Purpose:
- Create a new reoriented mesh (e.g., coronary2_aligned.msh)
- Preserve topology/connectivity/zone definitions exactly
- Change only node coordinates

Inputs:
- Original Fluent legacy ASCII .msh (mesh to transform)
- .transform.npz file produced by your alignment script (contains R and t)

Outputs:
- New Fluent legacy ASCII .msh with transformed node coordinates
- Optional verification JSON summary

Tested design assumptions:
- Fluent legacy ASCII .msh with node sections "(10 ...)" and face sections "(13 ...)"
- 3D node coordinates
- Same parser logic style as your training/inference pipeline
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Defaults (edit or override via CLI)
# =============================================================================

DEFAULT_MSH_IN = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model2\\coronary2.msh"
DEFAULT_TRANSFORM_NPZ = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model2\\alignment_out\\mesh2_aligned_to_mesh1_20260225_113727.transform.npz"
DEFAULT_MSH_OUT = r"C:\\Users\\radie\\Desktop\\1_TrainingSIMS\\real_coronary_meshes\\model2\\coronary2_aligned.msh"

# =============================================================================
# Logging / JSON helpers
# =============================================================================

class Logger:
    def __init__(self):
        pass

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(x: Any):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


# =============================================================================
# Fluent legacy ASCII .msh parsing helpers (compatible with your pipeline style)
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


def scan_fluent_msh_headers(msh_path: Path, int_mode: str = "hex") -> MeshHeaderScan:
    scaling = 1.0
    node_decl: Optional[NodeDeclMeta] = None

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

    return MeshHeaderScan(
        msh_path=str(msh_path),
        int_mode=str(int_mode),
        scaling_factor=float(scaling),
        node_decl=node_decl,
    )


def _read_header_line_only(fh, first_line: str, line_no: int) -> Tuple[str, int]:
    """
    Read enough lines to complete a Fluent section header, but stop before body coords.
    Matches style from your pipeline parser.
    """
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
            # likely body starts; include once then stop (same strategy as existing code)
            hdr += " " + s2
            break
        hdr += " " + s2
        toks_try = _strip_parens(hdr).split()
        if len(toks_try) >= 6:
            break
    return hdr, line_no


# =============================================================================
# Transform loading / application
# =============================================================================

@dataclass
class RigidTransform:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply_points(self, xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3], got {pts.shape}")
        # Row-vector convention matching previous alignment script exports
        out = pts @ self.R.T + self.t.reshape(1, 3)
        return out.astype(np.float64)

    def summary(self) -> Dict[str, float]:
        detR = float(np.linalg.det(self.R))
        ortho_err = float(np.linalg.norm(self.R.T @ self.R - np.eye(3)))
        return {"det_R": detR, "orthogonality_error_fro": ortho_err}


def load_transform_npz(npz_path: Path, logger: Logger) -> RigidTransform:
    npz_path = Path(npz_path)
    z = np.load(npz_path)

    if "R" not in z or "t" not in z:
        raise KeyError(f"Transform file missing R and/or t: {npz_path}")

    R = np.asarray(z["R"], dtype=np.float64)
    t = np.asarray(z["t"], dtype=np.float64).reshape(-1)

    if R.shape != (3, 3):
        raise ValueError(f"R must be shape (3,3), got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"t must be shape (3,), got {t.shape}")

    tfm = RigidTransform(R=R, t=t)
    s = tfm.summary()
    logger.log(f"[TFM] Loaded transform: {npz_path}")
    logger.log(f"[TFM] det(R)={s['det_R']:.12g}, orthogonality_error={s['orthogonality_error_fro']:.6g}")
    logger.log(f"[TFM] t={t}")

    return tfm


# =============================================================================
# Core .msh rewrite (transform nodes only, preserve everything else)
# =============================================================================

@dataclass
class RewriteStats:
    input_msh: str
    output_msh: str
    int_mode: str
    scaling_factor_detected: float
    node_decl_first: Optional[int]
    node_decl_last: Optional[int]
    node_decl_n: Optional[int]
    node_sections_rewritten: int
    nodes_written_total: int
    coord_lines_replaced: int
    dim_detected_values: List[int]
    transform_det_R: float
    transform_orthogonality_error_fro: float
    transform_t_norm: float
    format_style: str


def rewrite_fluent_msh_with_transform(
    msh_in: Path,
    msh_out: Path,
    tfm: RigidTransform,
    int_mode: str,
    logger: Logger,
    float_fmt: str = "{:.12e}",
) -> RewriteStats:
    """
    Stream-rewrite .msh:
    - Copy everything verbatim EXCEPT actual node coordinate bodies in (10 zone!=0 ...) sections
    - Replace those coordinates with transformed values
    - Preserve section headers and all non-node data
    """
    msh_in = Path(msh_in)
    msh_out = Path(msh_out)
    msh_out.parent.mkdir(parents=True, exist_ok=True)

    scan = scan_fluent_msh_headers(msh_in, int_mode=int_mode)
    if scan.node_decl is None:
        raise RuntimeError(f"Could not find node declaration in {msh_in}")
    node_decl = scan.node_decl

    node_first_global = int(node_decl.first)
    node_last_global = int(node_decl.last)
    n_nodes_global = int(node_decl.n_nodes)

    logger.log(f"[MSH] Input: {msh_in}")
    logger.log(f"[MSH] Output: {msh_out}")
    logger.log(f"[MSH] scaling_factor (header scan): {scan.scaling_factor}")
    logger.log(f"[MSH] node_decl first={node_first_global} last={node_last_global} n={n_nodes_global}")

    # We count nodes written as sanity check
    nodes_written_total = 0
    node_sections_rewritten = 0
    coord_lines_replaced = 0
    dim_detected_values: List[int] = []

    def node_to_local(nid_1based: int) -> int:
        idx = nid_1based - node_first_global
        if idx < 0 or idx >= n_nodes_global:
            raise IndexError(f"Node id {nid_1based} outside declared range [{node_first_global},{node_last_global}]")
        return idx

    with msh_in.open("r", errors="ignore") as fin, msh_out.open("w", encoding="utf-8", newline="\n") as fout:
        ln = 0
        while True:
            line = fin.readline()
            if not line:
                break
            ln += 1
            s = line.strip()

            # Only intercept node sections
            if not s.startswith("(10"):
                fout.write(line)
                continue

            # Parse enough header to know section semantics
            hdr, ln = _read_header_line_only(fin, s, ln)
            toks = _strip_parens(hdr).split()

            # Write header back (normalized line, safe)
            fout.write(hdr.rstrip() + "\n")

            # If malformed header or not actual node section, continue
            if len(toks) < 6 or toks[0] != "10":
                continue

            try:
                zone = parse_int_token(toks[1], int_mode=int_mode)
            except Exception:
                zone = None

            # Declaration section (zone 0): no coords to rewrite
            if zone == 0:
                continue

            # Try parse node section metadata
            try:
                start = parse_int_token(toks[2], int_mode=int_mode)
                end = parse_int_token(toks[3], int_mode=int_mode)
                dim = parse_int_token(toks[5], int_mode=int_mode)
            except Exception as e:
                raise RuntimeError(f"Failed parsing node section header near line {ln}: {hdr}") from e

            dim_detected_values.append(int(dim))
            if dim != 3:
                raise RuntimeError(f"Expected 3D node section (dim=3), got dim={dim} at line {ln}")

            expected = end - start + 1
            if expected <= 0:
                # Still need to pass through body until ')'
                while True:
                    body_line = fin.readline()
                    if not body_line:
                        break
                    ln += 1
                    fout.write(body_line)
                    if body_line.strip().startswith(")"):
                        break
                continue

            node_sections_rewritten += 1
            logger.log(f"[REWRITE] Node zone={zone} start={start} end={end} n={expected}")

            nid = start
            # Consume original body lines and emit transformed coords in same body-line grouping counts
            while True:
                body_line = fin.readline()
                if not body_line:
                    raise RuntimeError(f"Unexpected EOF inside node section zone={zone} (started at line ~{ln}).")
                ln += 1
                raw = body_line.strip()

                if raw.startswith(")"):
                    # Sanity: must have emitted all nodes in this section
                    if nid <= end:
                        raise RuntimeError(
                            f"Node section zone={zone} closed before all nodes emitted "
                            f"(last emitted node-id {nid-1}, expected end {end})."
                        )
                    fout.write(body_line)  # preserve closing paren line
                    break

                t = _strip_parens(raw)
                if t == "":
                    # preserve blank-ish lines
                    fout.write(body_line)
                    continue

                # Parse floats from the original line to determine how many coordinates this line carried
                vals: List[float] = []
                for a in t.split():
                    try:
                        vals.append(float(a))
                    except Exception:
                        pass

                if len(vals) == 0:
                    # Non-numeric content (rare); preserve
                    fout.write(body_line)
                    continue

                if len(vals) % dim != 0:
                    raise RuntimeError(
                        f"Node coord line at input line {ln} has {len(vals)} floats, not multiple of dim={dim}."
                    )

                n_nodes_on_this_line = len(vals) // dim
                out_vals: List[str] = []

                for _ in range(n_nodes_on_this_line):
                    if nid > end:
                        # Original line has more coords than expected; malformed or parser mismatch
                        raise RuntimeError(
                            f"Node section zone={zone} has extra coordinates after expected end node {end}."
                        )
                    local_idx = node_to_local(nid)
                    # Reconstruct original unscaled coordinate basis from file line values?
                    # IMPORTANT:
                    # We should transform the ACTUAL coordinates as present in file (raw body values), not header-scaled values.
                    # Since we're reading raw vals line-by-line, use them directly for exact rewrite.
                    #
                    # Extract one node from raw vals at the same line position:
                    k0 = (_ * dim)
                    xyz_raw = np.array([vals[k0], vals[k0 + 1], vals[k0 + 2]], dtype=np.float64).reshape(1, 3)

                    xyz_new = tfm.apply_points(xyz_raw)[0]
                    out_vals.extend([
                        float_fmt.format(float(xyz_new[0])),
                        float_fmt.format(float(xyz_new[1])),
                        float_fmt.format(float(xyz_new[2])),
                    ])

                    nid += 1
                    nodes_written_total += 1

                # Write one line preserving body style as simple space-separated floats
                fout.write(" ".join(out_vals) + "\n")
                coord_lines_replaced += 1

            # Sanity after closing
            if nid - start != expected:
                raise RuntimeError(
                    f"Rewrote {nid-start} nodes in zone={zone}, expected {expected}. "
                    f"Check parser/int_mode."
                )

    # Final sanity
    if nodes_written_total != n_nodes_global:
        # Some Fluent meshes can have node zones not covering full declaration range, but typically they do.
        logger.log(f"[WARN] nodes_written_total={nodes_written_total} != node_decl_n={n_nodes_global}.")
    else:
        logger.log(f"[OK] Rewrote all declared nodes: {nodes_written_total}")

    tfm_summary = tfm.summary()
    stats = RewriteStats(
        input_msh=str(msh_in),
        output_msh=str(msh_out),
        int_mode=str(int_mode),
        scaling_factor_detected=float(scan.scaling_factor),
        node_decl_first=int(node_first_global),
        node_decl_last=int(node_last_global),
        node_decl_n=int(n_nodes_global),
        node_sections_rewritten=int(node_sections_rewritten),
        nodes_written_total=int(nodes_written_total),
        coord_lines_replaced=int(coord_lines_replaced),
        dim_detected_values=sorted(set(int(x) for x in dim_detected_values)),
        transform_det_R=float(tfm_summary["det_R"]),
        transform_orthogonality_error_fro=float(tfm_summary["orthogonality_error_fro"]),
        transform_t_norm=float(np.linalg.norm(tfm.t)),
        format_style=str(float_fmt),
    )
    return stats


# =============================================================================
# Lightweight verification (header-level + transform file sanity)
# =============================================================================

def verify_rewritten_mesh_headers(msh_path: Path, int_mode: str, logger: Logger) -> Dict[str, Any]:
    scan = scan_fluent_msh_headers(msh_path, int_mode=int_mode)
    out = {
        "msh_path": str(msh_path),
        "int_mode": str(int_mode),
        "scaling_factor": float(scan.scaling_factor),
        "node_decl_present": bool(scan.node_decl is not None),
        "node_decl_first": int(scan.node_decl.first) if scan.node_decl else None,
        "node_decl_last": int(scan.node_decl.last) if scan.node_decl else None,
        "node_decl_n": int(scan.node_decl.n_nodes) if scan.node_decl else None,
    }
    logger.log(f"[VERIFY] Header scan OK for {msh_path}")
    if scan.node_decl:
        logger.log(f"[VERIFY] node_decl first={scan.node_decl.first} last={scan.node_decl.last} n={scan.node_decl.n_nodes}")
    return out


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rewrite Fluent legacy ASCII .msh by applying rigid transform to node coordinates only."
    )
    ap.add_argument("--msh_in", type=str, default=DEFAULT_MSH_IN, help="Input Fluent legacy ASCII .msh")
    ap.add_argument("--transform_npz", type=str, default=DEFAULT_TRANSFORM_NPZ, help="Transform NPZ (contains R, t)")
    ap.add_argument("--msh_out", type=str, default=DEFAULT_MSH_OUT, help="Output .msh path")
    ap.add_argument("--int_mode", type=str, default="hex", choices=["hex", "dec", "auto"], help="Fluent integer parsing mode")
    ap.add_argument("--float_fmt", type=str, default="{:.12e}", help='Float format for rewritten coords, e.g. "{:.12e}"')
    ap.add_argument("--save_json", action="store_true", help="Save rewrite summary JSON next to output mesh")
    ap.add_argument("--verify_headers", action="store_true", help="Re-scan rewritten mesh headers after writing")
    args = ap.parse_args()

    logger = Logger()

    msh_in = Path(args.msh_in)
    tfm_npz = Path(args.transform_npz)
    msh_out = Path(args.msh_out)

    if not msh_in.exists():
        raise FileNotFoundError(f"Input mesh not found: {msh_in}")
    if not tfm_npz.exists():
        raise FileNotFoundError(f"Transform NPZ not found: {tfm_npz}")

    logger.log("=" * 96)
    logger.log("FLUENT .MSH RIGID-TRANSFORM REWRITER")
    logger.log("=" * 96)
    logger.log(f"MSH_IN        : {msh_in}")
    logger.log(f"TRANSFORM_NPZ : {tfm_npz}")
    logger.log(f"MSH_OUT       : {msh_out}")
    logger.log(f"INT_MODE      : {args.int_mode}")
    logger.log(f"FLOAT_FMT     : {args.float_fmt}")

    tfm = load_transform_npz(tfm_npz, logger)

    stats = rewrite_fluent_msh_with_transform(
        msh_in=msh_in,
        msh_out=msh_out,
        tfm=tfm,
        int_mode=str(args.int_mode),
        logger=logger,
        float_fmt=str(args.float_fmt),
    )

    logger.log("[DONE] Mesh rewrite completed.")
    logger.log(f"[DONE] Output mesh: {msh_out}")

    verify_summary = None
    if args.verify_headers:
        verify_summary = verify_rewritten_mesh_headers(msh_out, int_mode=str(args.int_mode), logger=logger)

    if args.save_json:
        out_json = msh_out.with_suffix(msh_out.suffix + ".rewrite_summary.json")
        payload = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rewrite_stats": dataclasses.asdict(stats),
            "verify_headers": verify_summary,
            "transform_npz": str(tfm_npz),
        }
        write_json(out_json, payload)
        logger.log(f"[OK] Wrote summary JSON: {out_json}")


if __name__ == "__main__":
    main()