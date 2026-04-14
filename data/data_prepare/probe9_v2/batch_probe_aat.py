#!/usr/bin/env python3
"""
Batch probe script for transposed matrices (AT) in training_dataset_t.
Results go to probe9_v2/results_AAT.
Skips matrices that failed to transpose (missing or malformed files).
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime

# Configuration
DATASET_DIR = "/home/stu1/Dataset/training_dataset_t"
PROBE_EXEC = "/home/stu1/marui/probe9_v2/tile_probe"
OUTPUT_DIR = "/home/stu1/marui/probe9_v2/results_AAT"
TIMEOUT_SEC = 600  # 10 minutes per matrix


def run_probe(matrix_path):
    """Run tile_probe on a single matrix."""
    try:
        result = subprocess.run(
            [PROBE_EXEC, matrix_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=TIMEOUT_SEC
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def is_valid_mtx(filepath):
    """Check if a .mtx file is valid (not empty, has proper header)."""
    try:
        if os.path.getsize(filepath) < 50:
            return False
        with open(filepath, 'r') as f:
            line = f.readline()
            if '%%MatrixMarket' not in line:
                return False
        return True
    except:
        return False


def parse_output(output_text):
    """Parse tile_probe output. Returns (info_dict, csv_header, csv_rows)."""
    info = {}
    for key, pattern in [
        ('rows',            r'Rows \(m\): (\d+)'),
        ('cols',            r'Cols \(n\): (\d+)'),
        ('nnz',             r'Nonzeros \(nnz\): (\d+)'),
        ('symmetric',       r'Symmetric: (\w+)'),
        ('density',         r'Density: ([\d.e\-+]+)'),
        ('avg_nnz_per_row', r'Avg nnz/row: ([\d.]+)'),
        ('probe_time_ms',   r'Probe time: ([\d.]+) ms'),
    ]:
        m = re.search(pattern, output_text)
        info[key] = m.group(1) if m else ''

    csv_header = ''
    csv_rows = []
    csv_match = re.search(
        r'=== CSV Format Features ===\s*\n(.*?)(?:\n===|\n\s*$|\Z)',
        output_text, re.DOTALL
    )
    if csv_match:
        lines = [l.strip() for l in csv_match.group(1).strip().split('\n') if l.strip()]
        if lines:
            csv_header = lines[0]
            csv_rows = lines[1:]

    return info, csv_header, csv_rows


def process_folder(folder_path):
    """Process all matrices in a folder, writing results incrementally."""

    folder_name = Path(folder_path).name
    mtx_files = sorted(Path(folder_path).glob("*.mtx"))

    if not mtx_files:
        return

    # Filter out invalid files
    valid_files = []
    skip_count = 0
    for f in mtx_files:
        if is_valid_mtx(str(f)):
            valid_files.append(f)
        else:
            skip_count += 1
            print("  SKIP (invalid): {}".format(f.stem))

    print("[{f}] {n} matrices ({s} skipped)".format(
        f=folder_name, n=len(valid_files), s=skip_count))

    if not valid_files:
        return

    output_file = Path(OUTPUT_DIR) / "{}_probe_results.txt".format(folder_name)
    tile_csv_header = ''
    ok_count = 0
    fail_count = 0

    with open(str(output_file), 'w') as f:
        f.write("=" * 160 + "\n")
        f.write("Tile Probe Results (AT) - {}\n".format(folder_name))
        f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("Total matrices in folder: {}\n".format(len(valid_files)))
        f.write("=" * 160 + "\n\n")

        for i, mtx_file in enumerate(valid_files, 1):
            name = mtx_file.stem
            sys.stdout.write("  [{}/{}] {} ... ".format(i, len(valid_files), name))
            sys.stdout.flush()

            output = run_probe(str(mtx_file))

            if output is None:
                print("TIMEOUT")
                f.write("### {} - TIMEOUT\n\n".format(name))
                f.flush()
                fail_count += 1
                continue

            info, csv_hdr, csv_rows = parse_output(output)

            if not csv_rows:
                print("PARSE_ERROR")
                f.write("### {} - PARSE_ERROR\n\n".format(name))
                f.flush()
                fail_count += 1
                continue

            if csv_hdr and not tile_csv_header:
                tile_csv_header = csv_hdr

            ok_count += 1
            print("OK  ({r}x{c}, nnz={z})".format(
                r=info.get('rows', '?'), c=info.get('cols', '?'),
                z=info.get('nnz', '?')))

            f.write("### {name}  |  {r}x{c}  |  nnz={z}  |  density={d}  |  sym={s}  |  avg_nnz/row={a}  |  probe={p}ms\n".format(
                name=name,
                r=info.get('rows', ''),
                c=info.get('cols', ''),
                z=info.get('nnz', ''),
                d=info.get('density', ''),
                s=info.get('symmetric', ''),
                a=info.get('avg_nnz_per_row', ''),
                p=info.get('probe_time_ms', '')))

            if tile_csv_header:
                f.write("  [header] {}\n".format(tile_csv_header))
            for row in csv_rows:
                f.write("  {}\n".format(row))
            f.write("\n")
            f.flush()

        f.write("=" * 160 + "\n")
        f.write("Summary: {ok} OK, {fail} failed, {total} total\n".format(
            ok=ok_count, fail=fail_count, total=len(valid_files)))
        f.write("=" * 160 + "\n\n")

        f.write("=== CSV FORMAT (all matrices, all tile sizes) ===\n\n")
        f.write("matrix_name,rows,cols,nnz,density,symmetric,avg_nnz_per_row,{}\n".format(
            tile_csv_header))
        f.flush()

    _append_csv_section(output_file, valid_files, tile_csv_header)

    print("  -> {f}  ({ok}/{total} OK)\n".format(
        f=output_file, ok=ok_count, total=len(valid_files)))


def _append_csv_section(output_file, mtx_files, tile_csv_header):
    """Re-read the result file and append a combined CSV section."""
    lines = []
    with open(str(output_file), 'r') as f:
        lines = f.readlines()

    csv_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if '=== CSV FORMAT' in lines[i]:
            csv_start = i
            break

    if csv_start == -1:
        return

    with open(str(output_file), 'a') as f:
        current_name = ''
        current_info = ''
        current_rows = []

        for line in lines:
            line_s = line.strip()

            if line_s.startswith('### ') and '|' in line_s:
                if current_name and current_rows:
                    for row in current_rows:
                        f.write("{},{}\n".format(current_info, row))
                    current_rows = []

                parts = line_s[4:].split('|')
                if len(parts) >= 6:
                    name = parts[0].strip()
                    rc = parts[1].strip()
                    nnz = parts[2].strip().replace('nnz=', '')
                    density = parts[3].strip().replace('density=', '')
                    sym = parts[4].strip().replace('sym=', '')
                    avg = parts[5].strip().replace('avg_nnz/row=', '')
                    r, c = rc.split('x') if 'x' in rc else ('', '')
                    current_name = name
                    current_info = "{},{},{},{},{},{},{}".format(
                        name, r.strip(), c.strip(), nnz, density, sym, avg)

            elif line_s and not line_s.startswith('#') and not line_s.startswith('=') and not line_s.startswith('[') and not line_s.startswith('(') and not line_s.startswith('Tile') and not line_s.startswith('Generated') and not line_s.startswith('Total') and not line_s.startswith('Summary') and not line_s.startswith('matrix_name') and not line_s.startswith('---') and ',' in line_s and line_s[0].isdigit():
                current_rows.append(line_s)

        if current_name and current_rows:
            for row in current_rows:
                f.write("{},{}\n".format(current_info, row))

        f.write("\n" + "=" * 160 + "\nEnd of report\n")


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("=" * 70)
    print("Batch Tile Probe - Transposed Dataset (AT for AAT)")
    print("=" * 70)
    print("Dataset:  " + DATASET_DIR)
    print("Output:   " + OUTPUT_DIR)
    print("Timeout:  {} sec/matrix".format(TIMEOUT_SEC))
    print("=" * 70 + "\n")

    if not Path(PROBE_EXEC).exists():
        print("ERROR: {} not found!".format(PROBE_EXEC))
        return

    subfolders = sorted([
        d for d in Path(DATASET_DIR).iterdir()
        if d.is_dir() and list(d.glob("*.mtx"))
    ])

    print("Found {} non-empty subfolders\n".format(len(subfolders)))

    for folder in subfolders:
        process_folder(folder)

    print("=" * 70)
    print("All done! Results in: " + OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
