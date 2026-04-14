#!/usr/bin/env python3
"""
Task 4: Batch probeC for A*AT prediction on all matrices in training_dataset.
Uses --aat fast path (tile-level transpose, no AT file needed).
Skips matrices that have no corresponding AT in training_dataset_t
(i.e., missing folders or missing individual files like conf6_0-8x8-30).
Results go to probeC_v2/result_AAT.
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime

PROBE_EXEC = "/home/stu1/marui/probeC_v2/probeC"
DATASET_A = "/home/stu1/Dataset/training_dataset"
DATASET_AT = "/home/stu1/Dataset/training_dataset_t"
OUTPUT_DIR = "/home/stu1/marui/probeC_v2/result_AAT"
TIMEOUT_SEC = 600


def run_probeC_aat(matrix_a_path):
    """Run probeC --aat on a single matrix (fast tile-level transpose)."""
    try:
        result = subprocess.run(
            [PROBE_EXEC, '--aat', matrix_a_path],
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
    for key, pattern in [
        ('rows', r'Size: (\d+) x'),
        ('cols', r'Size: \d+ x (\d+)'),
        ('nnz', r'nnz: (\d+)'),
        ('symmetric', r'Symmetric: (\w+)'),
        ('load_ms', r'Load time: ([\d.]+) ms'),
        ('build_ms', r'Build time: ([\d.]+) ms'),
        ('estimate_ms', r'Estimate time: ([\d.]+) ms'),
    ]:
        m = re.search(pattern, output_text)
        info[key] = m.group(1) if m else ''

    csv_header = ''
    csv_rows = []
    csv_match = re.search(r'=== CSV ===\s*\n(.*?)$', output_text, re.DOTALL)
    if csv_match:
        lines = [l.strip() for l in csv_match.group(1).strip().split('\n') if l.strip()]
        if lines:
            csv_header = lines[0]
            csv_rows = lines[1:]

    return info, csv_header, csv_rows


def get_at_available(folder_name):
    """Get set of matrix names that have AT available in training_dataset_t."""
    at_dir = Path(DATASET_AT) / folder_name
    if not at_dir.exists():
        return None  # entire folder missing
    return set(f.stem for f in at_dir.glob('*.mtx'))


def process_folder(folder_path):
    """Process all matrices in a folder for AAT prediction."""
    folder_name = Path(folder_path).name
    mtx_files = sorted(Path(folder_path).glob("*.mtx"))

    if not mtx_files:
        return

    # Check which matrices have AT available
    at_available = get_at_available(folder_name)
    if at_available is None:
        print("[{}] SKIP - no corresponding AT folder in training_dataset_t".format(folder_name))
        return

    # Filter: only process matrices that have AT
    valid_files = []
    skip_count = 0
    for f in mtx_files:
        if f.stem in at_available:
            valid_files.append(f)
        else:
            skip_count += 1

    print("[{f}] {n} matrices ({s} skipped - no AT)".format(
        f=folder_name, n=len(valid_files), s=skip_count))

    if not valid_files:
        return

    output_file = Path(OUTPUT_DIR) / "{}_probeC_results.txt".format(folder_name)
    csv_header = ''
    ok_count = 0
    fail_count = 0

    with open(str(output_file), 'w') as f:
        f.write("=" * 160 + "\n")
        f.write("ProbeC AAT Results - {}\n".format(folder_name))
        f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("Total matrices: {}\n".format(len(valid_files)))
        f.write("=" * 160 + "\n\n")

        for i, mtx_file in enumerate(valid_files, 1):
            name = mtx_file.stem
            sys.stdout.write("  [{}/{}] {} ... ".format(i, len(valid_files), name))
            sys.stdout.flush()

            output = run_probeC_aat(str(mtx_file))

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

            if csv_hdr and not csv_header:
                csv_header = csv_hdr

            ok_count += 1
            print("OK  ({r}x{c}, nnz={z}, est={e}ms)".format(
                r=info.get('rows', '?'), c=info.get('cols', '?'),
                z=info.get('nnz', '?'), e=info.get('estimate_ms', '?')))

            f.write("### {name}  |  {r}x{c}  |  nnz={z}  |  sym={s}  |  load={l}ms  |  build={b}ms  |  estimate={e}ms\n".format(
                name=name,
                r=info.get('rows', ''),
                c=info.get('cols', ''),
                z=info.get('nnz', ''),
                s=info.get('symmetric', ''),
                l=info.get('load_ms', ''),
                b=info.get('build_ms', ''),
                e=info.get('estimate_ms', '')))

            if csv_header:
                f.write("  [header] {}\n".format(csv_header))
            for row in csv_rows:
                f.write("  {}\n".format(row))
            f.write("\n")
            f.flush()

        f.write("=" * 160 + "\n")
        f.write("Summary: {ok} OK, {fail} failed, {total} total\n".format(
            ok=ok_count, fail=fail_count, total=len(valid_files)))
        f.write("=" * 160 + "\n\n")

        # CSV section
        f.write("=== CSV FORMAT (all matrices) ===\n\n")
        f.write("matrix_name,rows,cols,nnz,symmetric,{}\n".format(csv_header))
        f.flush()

    # Append combined CSV
    _append_csv_section(output_file, csv_header)

    print("  -> {f}  ({ok}/{total} OK)\n".format(
        f=output_file, ok=ok_count, total=len(valid_files)))


def _append_csv_section(output_file, csv_header):
    """Re-read result file and append combined CSV section."""
    with open(str(output_file), 'r') as f:
        lines = f.readlines()

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
                if len(parts) >= 4:
                    name = parts[0].strip()
                    rc = parts[1].strip()
                    nnz = parts[2].strip().replace('nnz=', '')
                    sym = parts[3].strip().replace('sym=', '')
                    r, c = rc.split('x') if 'x' in rc else ('', '')
                    current_name = name
                    current_info = "{},{},{},{},{}".format(
                        name, r.strip(), c.strip(), nnz, sym)

            elif line_s and line_s[0].isdigit() and ',' in line_s:
                current_rows.append(line_s)

        if current_name and current_rows:
            for row in current_rows:
                f.write("{},{}\n".format(current_info, row))

        f.write("\n" + "=" * 160 + "\nEnd of report\n")


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("=" * 70)
    print("Batch ProbeC AAT - Training Dataset")
    print("=" * 70)
    print("A matrices:  " + DATASET_A)
    print("AT check:    " + DATASET_AT)
    print("Output:      " + OUTPUT_DIR)
    print("Timeout:     {} sec/matrix".format(TIMEOUT_SEC))
    print("=" * 70 + "\n")

    if not Path(PROBE_EXEC).exists():
        print("ERROR: {} not found!".format(PROBE_EXEC))
        return

    subfolders = sorted([
        d for d in Path(DATASET_A).iterdir()
        if d.is_dir() and list(d.glob("*.mtx"))
    ])

    print("Found {} subfolders in training_dataset\n".format(len(subfolders)))

    for folder in subfolders:
        process_folder(folder)

    print("=" * 70)
    print("All done! Results in: " + OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
