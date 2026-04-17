#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_utils.py
Shared utilities: path constants, column name definitions, probe output parsing, prime_data timing parsing, hardware.txt reading.
"""

import re
import subprocess
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Paths (all relative to the code tree that will be open-sourced)
# ═══════════════════════════════════════════════════════════════════════════════

CODE_ROOT = Path("/home/stu1/donghangcheng/code/FlexSpGEMM")
DATA_DIR = CODE_ROOT / "data"
PREP_DIR = DATA_DIR / "data_prepare"

PROBE9_EXEC = DATA_DIR / "tile_probe"
PROBEC_EXEC = DATA_DIR / "probeC"

PRIME_DATA = PREP_DIR / "prime_data"
HARDWARE_FILE = PRIME_DATA / "hardware.txt"
# Note: the A100_AA directory has a leading space in its name
A100_AA_DIR = PRIME_DATA / " A100_AA"
A100_AAT_DIR = PRIME_DATA / "A100_AAT"
H200_AA_DIR = PRIME_DATA / "H200_AA"
H200_AAT_DIR = PRIME_DATA / "H200_AAT"

MTX_DIRS = {
    "test": DATA_DIR / "test",
    "train": DATA_DIR / "train",
    "val": DATA_DIR / "val",
}
MTX_T_DIR = DATA_DIR / "mtx_T"
OUTPUT_DIR = PREP_DIR / "data_get"
STAGE1_DIR = OUTPUT_DIR / "stage1_intermediate"

TIMEOUT_SEC = 600

# ═══════════════════════════════════════════════════════════════════════════════
# Column name definitions — must match reference all_data/*.csv exactly
# ═══════════════════════════════════════════════════════════════════════════════

# Tile order for probe9 (matches tile_probe CSV output order)
# tile_probe outputs "8x8" but reference CSV uses "8_8"
P9_TILE_ORDER = ["8_8", "16_8", "8_16", "16_16", "16_32", "32_16", "32_32", "8_32", "32_8"]
# Map from tile_probe output name to our column name fragment
P9_TILE_MAP = {
    "8x8": "8_8", "16x8": "16_8", "8x16": "8_16",
    "16x16": "16_16", "16x32": "16_32", "32x16": "32_16",
    "32x32": "32_32", "8x32": "8_32", "32x8": "32_8",
}

P9_FIELDS = [
    "numtile", "tile_density", "nnz_per_tile_avg", "nnz_per_tile_max",
    "nnz_per_tile_min", "nnz_per_tile_std", "nnz_per_tile_cv",
    "tile_fill_avg", "tile_fill_max",
    "tiles_per_row_avg", "tiles_per_row_max", "tiles_per_row_min",
    "tiles_per_row_std", "tiles_per_row_cv", "empty_row_ratio",
    "hist_1", "hist_2_4", "hist_4_8", "hist_8_16", "hist_16_32",
    "hist_32_64", "hist_64_128", "hist_128_plus",
]

PC_TILE_ORDER = ["m8", "m16", "m32"]
PC_TILE_MAP = {"8": "m8", "16": "m16", "32": "m32"}

PC_FIELDS = [
    "sml", "lrg", "dns", "ful", "numblkC", "total_flops",
    "avg_matchedcnt", "max_matchedcnt", "max_flops_per_tile",
]

HARDWARE_COLS = [
    "mem_bandwidth_gbs", "sm_count", "l2_cache_mb",
    "shared_mem_per_sm_kb", "fp64_tc_tflops",
]

STAGE1_TIMING_COLS = [
    "probe9_A_wall_ms", "probe9_A_probe_ms",
    "probe9_AT_wall_ms", "probe9_AT_probe_ms",
    "probeC_AA_wall_ms", "probeC_AA_load_ms", "probeC_AA_build_ms", "probeC_AA_estimate_ms",
    "probeC_AAT_wall_ms", "probeC_AAT_load_ms", "probeC_AAT_build_ms", "probeC_AAT_estimate_ms",
]


def build_final_header():
    """Return the 454-column header matching the reference dataset."""
    cols = ["matrix_name", "rows", "cols", "nnz", "density", "avg_nnz_per_row"]
    for tile in P9_TILE_ORDER:
        for f in P9_FIELDS:
            cols.append("p9_{}_{}".format(tile, f))
    for tile in P9_TILE_ORDER:
        for f in P9_FIELDS:
            cols.append("p9_AT_{}_{}".format(tile, f))
    for tile in PC_TILE_ORDER:
        for f in PC_FIELDS:
            cols.append("pC_{}_{}".format(tile, f))
    cols.extend(HARDWARE_COLS)
    cols.extend(["best_tile", "best_tc"])
    return cols


# pC columns use a different prefix in the reference CSV:
# pC_m8_sml, pC_m16_sml, pC_m32_sml ...
def build_pc_col_name(tile_key, field):
    """tile_key is 'm8'/'m16'/'m32'; field is one of PC_FIELDS."""
    return "pC_{}_{}".format(tile_key, field)


# ═══════════════════════════════════════════════════════════════════════════════
# Probe execution and output parsing
# ═══════════════════════════════════════════════════════════════════════════════

def run_command(cmd, timeout_sec=TIMEOUT_SEC):
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout_sec,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        return result.returncode, result.stdout, result.stderr, wall_ms
    except subprocess.TimeoutExpired:
        wall_ms = (time.perf_counter() - started) * 1000.0
        return -1, "", "TIMEOUT", wall_ms


def _extract(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def _parse_csv_block(text, marker):
    if marker not in text:
        return []
    tail = text.split(marker, 1)[1]
    csv_lines = []
    for raw in tail.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("===") or line.startswith("="):
            break
        if "," in line:
            csv_lines.append(line)
    if len(csv_lines) < 2:
        return []
    header = [x.strip() for x in csv_lines[0].split(",")]
    rows = []
    for line in csv_lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    return rows


def parse_probe9(stdout):
    """Parse tile_probe stdout -> dict with p9_{tile}_{field} keys."""
    data = {}
    rows = _parse_csv_block(stdout, "=== CSV Format Features ===")
    by_tile = {r.get("tile_size", ""): r for r in rows}
    for raw_tile, col_tile in P9_TILE_MAP.items():
        row = by_tile.get(raw_tile, {})
        for f in P9_FIELDS:
            data["p9_{}_{}".format(col_tile, f)] = row.get(f, "")
    info = {
        "rows": _extract(r"Rows \(m\): (\d+)", stdout),
        "cols": _extract(r"Cols \(n\): (\d+)", stdout),
        "nnz": _extract(r"Nonzeros \(nnz\): (\d+)", stdout),
        "density": _extract(r"Density: ([\d.eE\-+]+)", stdout),
        "avg_nnz_per_row": _extract(r"Avg nnz/row: ([\d.]+)", stdout),
        "symmetric": _extract(r"Symmetric: (\w+)", stdout),
        "probe_time_ms": _extract(r"Probe time: ([\d.]+) ms", stdout),
    }
    return data, info


def parse_probe9_at(stdout):
    """Same as parse_probe9 but prefixed with p9_AT_."""
    data = {}
    rows = _parse_csv_block(stdout, "=== CSV Format Features ===")
    by_tile = {r.get("tile_size", ""): r for r in rows}
    for raw_tile, col_tile in P9_TILE_MAP.items():
        row = by_tile.get(raw_tile, {})
        for f in P9_FIELDS:
            data["p9_AT_{}_{}".format(col_tile, f)] = row.get(f, "")
    return data


def parse_probec(stdout, prefix="pC"):
    """Parse probeC stdout -> dict with {prefix}_m{8|16|32}_{field} keys."""
    data = {}
    rows = _parse_csv_block(stdout, "=== CSV ===")
    by_tile = {r.get("tile_m", ""): r for r in rows}
    for raw_tile, col_tile in PC_TILE_MAP.items():
        row = by_tile.get(raw_tile, {})
        for f in PC_FIELDS:
            data["{}_{}_{}".format(prefix, col_tile, f)] = row.get(f, "")
    timing = {
        "load_ms": _extract(r"Load time:\s*([\d.]+)\s*ms", stdout),
        "build_ms": _extract(r"Build time:\s*([\d.]+)\s*ms", stdout),
        "estimate_ms": _extract(r"Estimate time:\s*([\d.]+)\s*ms", stdout),
    }
    return data, timing


# ═══════════════════════════════════════════════════════════════════════════════
# prime_data timing file parsers — extract best_tile / best_tc from 81 combos
# ═══════════════════════════════════════════════════════════════════════════════

TILES_9 = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS_9 = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]


def _best_from_gflops(gflops_dict):
    """Given {combo_str: gflops_float}, return (best_tile, best_tc)."""
    if not gflops_dict:
        return "", ""
    best_combo = max(gflops_dict, key=gflops_dict.get)
    # combo is like "8x8_0/8"
    parts = best_combo.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return best_combo, ""


def parse_a100_aa_file(filepath):
    """Parse A100_AA flat txt (tabular 81-row format). Returns (best_tile, best_tc) or None."""
    gflops = {}
    try:
        with open(str(filepath), "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                if line.startswith("tile_m"):
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                left = parts[0].split()
                if len(left) < 4:
                    continue
                tile_m, tile_n = left[0], left[1]
                tc_frc = left[3]  # "0/8"
                mid = parts[1].split()
                if len(mid) < 6:
                    continue
                try:
                    gf = float(mid[5])
                except (ValueError, IndexError):
                    continue
                combo = "{}x{}".format(tile_m, tile_n) + "_" + tc_frc
                gflops[combo] = gf
    except Exception:
        return None
    return _best_from_gflops(gflops)


def parse_a100_aat_file(filepath):
    """Parse A100_AAT verbose log. Returns (best_tile, best_tc) or None (symmetric)."""
    gflops = {}
    content = ""
    try:
        with open(str(filepath), "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    # Newer A100_AAT files may use the same tabular format as A100_AA.
    if "|" in content and "tile_m tile_n tc_frc tc" in content:
        try:
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                if line.startswith("tile_m"):
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                left = parts[0].split()
                if len(left) < 4:
                    continue
                tile_m, tile_n = left[0], left[1]
                tc_frc = left[3]
                mid = parts[1].split()
                if len(mid) < 6:
                    continue
                try:
                    gf = float(mid[5])
                except (ValueError, IndexError):
                    continue
                combo = "{}x{}_{}".format(tile_m, tile_n, tc_frc)
                gflops[combo] = gf
        except Exception:
            return None

        if not gflops:
            return None
        return _best_from_gflops(gflops)

    if "SYMMETRIC_SKIP" in content or "does not do symmetric" in content:
        # Check if ALL configs are symmetric-skipped
        gf_matches = re.findall(r"gflops\s*=\s*([\d.]+)", content)
        if not gf_matches:
            return None  # truly symmetric, no AAT data

    # Extract per-config blocks
    blocks = re.split(r"配置\s+(\d+)/81:\s*tile=(\d+)x(\d+),\s*TC=(\d+/\d+)", content)
    # blocks: [preamble, idx, m, n, tc, block_content, idx, m, n, tc, block_content, ...]
    i = 1
    while i + 4 < len(blocks):
        tile_m = blocks[i + 1]
        tile_n = blocks[i + 2]
        tc = blocks[i + 3]
        block = blocks[i + 4]
        gf_match = re.search(r"gflops\s*=\s*([\d.]+)", block)
        if gf_match:
            combo = "{}x{}_{}".format(tile_m, tile_n, tc)
            gflops[combo] = float(gf_match.group(1))
        i += 5

    if not gflops:
        return None
    return _best_from_gflops(gflops)


def parse_h200_log_dir(dir_path):
    """Parse H200_AA or H200_AAT directory of 81 .log files. Returns (best_tile, best_tc) or None."""
    gflops = {}
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return None

    for logfile in dir_path.glob("*.log"):
        # Filename: aat0_m16_n16_tc0.log
        m = re.match(r"aat\d+_m(\d+)_n(\d+)_tc(\d+)\.log", logfile.name)
        if not m:
            continue
        tile_m, tile_n, tc_idx = m.group(1), m.group(2), int(m.group(3))
        tc_str = "{}/8".format(tc_idx)

        try:
            with logfile.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

        if "does not do symmetric" in content:
            return None  # symmetric matrix

        gf_match = re.search(r"gflops\s*=\s*([\d.]+)", content)
        if gf_match:
            combo = "{}x{}_{}".format(tile_m, tile_n, tc_str)
            gflops[combo] = float(gf_match.group(1))

    if not gflops:
        return None
    return _best_from_gflops(gflops)


# ═══════════════════════════════════════════════════════════════════════════════
# Lookup helpers for prime_data
# ═══════════════════════════════════════════════════════════════════════════════

def build_a100_aa_index():
    """Build {matrix_name: filepath} index for A100_AA."""
    idx = {}
    if not A100_AA_DIR.is_dir():
        return idx
    for f in A100_AA_DIR.glob("*.txt"):
        # Filename: category__matrix_name.txt
        parts = f.stem.split("__", 1)
        if len(parts) == 2:
            idx[parts[1]] = f
    return idx


def build_a100_aat_index():
    """Build {matrix_name: filepath} index for A100_AAT."""
    idx = {}
    if not A100_AAT_DIR.is_dir():
        return idx
    # Flat SYMMETRIC_SKIP files
    for f in A100_AAT_DIR.glob("*.txt"):
        if f.is_file():
            parts = f.stem.split("__", 1)
            if len(parts) == 2:
                idx[parts[1]] = f
    # Subdirectory files
    for subdir in A100_AAT_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.txt"):
                idx[f.stem] = f
    return idx


def build_h200_index(base_dir):
    """Build {matrix_name: dir_path} for H200_AA or H200_AAT."""
    idx = {}
    base = Path(base_dir)
    if not base.is_dir():
        return idx
    for d in base.iterdir():
        if d.is_dir():
            idx[d.name] = d
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware parameter reader
# ═══════════════════════════════════════════════════════════════════════════════

def read_hardware():
    """Read hardware.txt -> {"A100": {col: val}, "H200": {col: val}}."""
    hw = {}
    current_gpu = None
    with HARDWARE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                current_gpu = None
                continue
            if "=" not in line:
                current_gpu = line
                hw[current_gpu] = {}
            elif current_gpu:
                k, v = line.split("=", 1)
                hw[current_gpu][k.strip()] = v.strip()
    return hw
