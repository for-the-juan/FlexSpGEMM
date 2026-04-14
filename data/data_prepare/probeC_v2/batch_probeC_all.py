#!/usr/bin/env python3
"""
Batch probeC script for all matrices in training_dataset.
Writes results incrementally (after each matrix) so partial results are always available.
"""

import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime

# Configuration
DATASET_DIR = "/home/stu1/Dataset/training_dataset"
PROBE_EXEC = "/home/stu1/marui/ada/probeC/probeC"
OUTPUT_DIR = "/home/stu1/marui/ada/probeC/result"
TIMEOUT_SEC = 600  # 10 minutes per matrix


def run_probe(matrix_path):
    """Run probeC on a single matrix."""
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


def parse_output(output_text):
    """Parse probeC output. Returns (info_dict, csv_header, csv_rows)."""
    info = {}

    # Size: 1138 x 1138, nnz: 4054, Symmetric: Yes
    m = re.search(r'Size:\s*(\d+)\s*x\s*(\d+),\s*nnz:\s*(\d+),\s*Symmetric:\s*(\w+)', output_text)
    if m:
        info['rows'] = m.group(1)
        info['cols'] = m.group(2)
        info['nnz'] = m.group(3)
        info['symmetric'] = m.group(4)

    # Load time
    m = re.search(r'Load time:\s*([\d.]+)\s*ms', output_text)
    if m:
        info['load_ms'] = m.group(1)

    # Build time
    m = re.search(r'Build time:\s*([\d.]+)\s*ms', output_text)
    if m:
        info['build_ms'] = m.group(1)

    # Estimate time
    m = re.search(r'Estimate time:\s*([\d.]+)\s*ms', output_text)
    if m:
        info['estimate_ms'] = m.group(1)

    # CSV section
    csv_header = ''
    csv_rows = []
    csv_match = re.search(
        r'=== CSV ===\s*\n(.*?)(?:\n===|\n\s*$|\Z)',
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

    print("[{f}] {n} matrices".format(f=folder_name, n=len(mtx_files)))

    output_file = Path(OUTPUT_DIR) / "{}_probeC_results.txt".format(folder_name)
    tile_csv_header = ''
    ok_count = 0
    fail_count = 0

    # For combined CSV at end
    all_csv_entries = []  # (matrix_info_str, csv_rows)

    with open(str(output_file), 'w') as f:
        # Write file header
        f.write("=" * 160 + "\n")
        f.write("ProbeC Results - {}\n".format(folder_name))
        f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("Total matrices in folder: {}\n".format(len(mtx_files)))
        f.write("=" * 160 + "\n\n")

        for i, mtx_file in enumerate(mtx_files, 1):
            name = mtx_file.stem
            sys.stdout.write("  [{}/{}] {} ... ".format(i, len(mtx_files), name))
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
            print("OK  ({r}x{c}, nnz={z}, build={b}ms, est={e}ms)".format(
                r=info.get('rows', '?'), c=info.get('cols', '?'),
                z=info.get('nnz', '?'),
                b=info.get('build_ms', '?'), e=info.get('estimate_ms', '?')))

            # Write this matrix's results immediately
            f.write("### {name}  |  {r}x{c}  |  nnz={z}  |  sym={s}  |  load={ld}ms  |  build={bd}ms  |  estimate={es}ms\n".format(
                name=name,
                r=info.get('rows', ''),
                c=info.get('cols', ''),
                z=info.get('nnz', ''),
                s=info.get('symmetric', ''),
                ld=info.get('load_ms', ''),
                bd=info.get('build_ms', ''),
                es=info.get('estimate_ms', '')))

            # Write CSV rows (3 tile sizes)
            if tile_csv_header:
                f.write("  [header] {}\n".format(tile_csv_header))
            for row in csv_rows:
                f.write("  {}\n".format(row))
            f.write("\n")
            f.flush()

            # Save for combined CSV
            info_str = "{},{},{},{},{}".format(
                name, info.get('rows', ''), info.get('cols', ''),
                info.get('nnz', ''), info.get('symmetric', ''))
            timing_str = "{},{},{}".format(
                info.get('load_ms', ''), info.get('build_ms', ''),
                info.get('estimate_ms', ''))
            all_csv_entries.append((info_str, timing_str, csv_rows))

        # Write summary at end
        f.write("=" * 160 + "\n")
        f.write("Summary: {ok} OK, {fail} failed, {total} total\n".format(
            ok=ok_count, fail=fail_count, total=len(mtx_files)))
        f.write("=" * 160 + "\n\n")

        # Write combined CSV section at end
        f.write("=== CSV FORMAT (all matrices, all tile sizes) ===\n")
        f.write("matrix_name,rows,cols,nnz,symmetric,load_ms,build_ms,estimate_ms,{}\n".format(
            tile_csv_header))
        for info_str, timing_str, csv_rows in all_csv_entries:
            for row in csv_rows:
                f.write("{},{},{}\n".format(info_str, timing_str, row))
        f.flush()

    print("  -> {f}  ({ok}/{total} OK)\n".format(
        f=output_file, ok=ok_count, total=len(mtx_files)))


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Batch ProbeC - Training Dataset")
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
