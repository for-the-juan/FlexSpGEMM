#!/usr/bin/env python3
"""
compare.py - Run probeC and AdaSpGEMM, compare C tile distribution results.
"""

import subprocess
import re
import os
import sys
from datetime import datetime

PROBEC = "/home/stu1/marui/ada/probeC/probeC"
ADA_BIN = "/home/stu1/marui/ada/AdaSpGEMM/bin"
DATASET = "/home/stu1/marui/ada/TileSpGEMMDataset"
OUTPUT = "/home/stu1/marui/ada/test/probeC_test.txt"
CSV_FILE = "/home/stu1/marui/ada/test/probeC_csv.txt"

TILE_CONFIGS = [
    (8, 8), (8, 16), (8, 32),
    (16, 8), (16, 16), (16, 32),
    (32, 8), (32, 16), (32, 32),
]

def run_cmd(cmd, timeout=300):
    """Run command and return (returncode, stdout)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)

def parse_probec_csv(output):
    """Parse probeC CSV output into dict keyed by (tile_m, tile_n)."""
    results = {}
    in_csv = False
    for line in output.split('\n'):
        if line.strip() == "tile_m,tile_n,total,tiny,sml,lrg,dns,ful":
            in_csv = True
            continue
        if in_csv and line.strip():
            parts = line.strip().split(',')
            if len(parts) == 8:
                try:
                    tm, tn, total, tiny, sml, lrg, dns, ful = [int(x) for x in parts]
                    results[(tm, tn)] = {
                        'total': total, 'tiny': tiny, 'sml': sml,
                        'lrg': lrg, 'dns': dns, 'ful': ful
                    }
                except ValueError:
                    pass
    return results

def parse_ada_output(output):
    """Parse AdaSpGEMM output for Number line and Non-empty tiles."""
    result = {}

    # Extract "Number: Tiny: X, Sml: Y, Lrg: Z, Dns: W, Ful: V"
    m = re.search(r'Number:\s*Tiny:\s*(\d+),\s*Sml:\s*(\d+),\s*Lrg:\s*(\d+),\s*Dns:\s*(\d+),\s*Ful:\s*(\d+)', output)
    if m:
        result['tiny'] = int(m.group(1))
        result['sml'] = int(m.group(2))
        result['lrg'] = int(m.group(3))
        result['dns'] = int(m.group(4))
        result['ful'] = int(m.group(5))

    # Extract "Non-empty tiles of C = XXXX"
    m2 = re.search(r'Non-empty tiles of C\s*=\s*(\d+)', output)
    if m2:
        result['total'] = int(m2.group(1))

    return result

def main():
    # Get list of matrix files
    mtx_files = sorted([f for f in os.listdir(DATASET) if f.endswith('.mtx')])

    out_lines = []
    csv_lines = ["matrix,tile_m,tile_n,pred_total,pred_tiny,pred_sml,pred_lrg,pred_dns,pred_ful,actual_total,actual_tiny,actual_sml,actual_lrg,actual_dns,actual_ful"]

    out_lines.append("=" * 64)
    out_lines.append("  probeC vs AdaSpGEMM Comparison Report")
    out_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out_lines.append("=" * 64)
    out_lines.append("")

    for mtx_name in mtx_files:
        base_name = mtx_name.replace('.mtx', '')
        mtx_path = os.path.join(DATASET, mtx_name)

        print(f"Processing: {base_name} ...")

        out_lines.append("=" * 64)
        out_lines.append(f"Matrix: {base_name}")
        out_lines.append("=" * 64)

        # Run probeC
        print("  Running probeC...")
        ret, probe_out = run_cmd([PROBEC, mtx_path], timeout=600)

        # Add first few info lines
        for line in probe_out.split('\n')[:5]:
            if line.strip():
                out_lines.append(line)
        out_lines.append("")

        # Parse probeC results
        pred = parse_probec_csv(probe_out)

        # Header for comparison table
        hdr = f"  {'TileSize':<10} {'P_Total':>8} {'P_Tiny':>8} {'P_Sml':>8} {'P_Lrg':>8} {'P_Dns':>8} {'P_Ful':>8} | {'A_Total':>8} {'A_Tiny':>8} {'A_Sml':>8} {'A_Lrg':>8} {'A_Dns':>8} {'A_Ful':>8}"
        sep = f"  {'----------':<10} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} | {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8}"
        out_lines.append(hdr)
        out_lines.append(sep)

        # Run each AdaSpGEMM tile size
        for tm, tn in TILE_CONFIGS:
            bin_path = os.path.join(ADA_BIN, f"test_m{tm}_n{tn}")

            if not os.path.isfile(bin_path):
                print(f"  Skipping {tm}x{tn} (binary not found)")
                continue

            print(f"  Running AdaSpGEMM {tm}x{tn}...")

            # Run with retry
            actual = {}
            for attempt in range(3):
                ret, ada_out = run_cmd([bin_path, "-d", "0", "-aat", "0", mtx_path], timeout=300)
                actual = parse_ada_output(ada_out)
                if 'tiny' in actual:
                    break
                print(f"    Attempt {attempt+1} failed (exit={ret}), retrying...")
                import time
                time.sleep(3)

            if 'tiny' not in actual:
                out_lines.append(f"    WARNING: AdaSpGEMM failed for {tm}x{tn} (exit={ret})")
                continue

            # Get predicted values
            p = pred.get((tm, tn), {'total': 0, 'tiny': 0, 'sml': 0, 'lrg': 0, 'dns': 0, 'ful': 0})
            a = actual

            line = f"  {tm}x{tn:<8} {p['total']:>8} {p['tiny']:>8} {p['sml']:>8} {p['lrg']:>8} {p['dns']:>8} {p['ful']:>8} | {a.get('total', 0):>8} {a.get('tiny', 0):>8} {a.get('sml', 0):>8} {a.get('lrg', 0):>8} {a.get('dns', 0):>8} {a.get('ful', 0):>8}"
            out_lines.append(line)

            # CSV line
            csv_lines.append(f"{base_name},{tm},{tn},{p['total']},{p['tiny']},{p['sml']},{p['lrg']},{p['dns']},{p['ful']},{a.get('total', 0)},{a.get('tiny', 0)},{a.get('sml', 0)},{a.get('lrg', 0)},{a.get('dns', 0)},{a.get('ful', 0)}")

        out_lines.append("")
        print(f"{base_name} Done!")

    # Accuracy summary
    out_lines.append("")
    out_lines.append("=" * 64)
    out_lines.append("  Accuracy Summary")
    out_lines.append("=" * 64)
    out_lines.append("")

    # Calculate from CSV data
    category_errors = {'tiny': 0, 'sml': 0, 'lrg': 0, 'dns': 0, 'ful': 0}
    category_counts = {'tiny': 0, 'sml': 0, 'lrg': 0, 'dns': 0, 'ful': 0}
    total_tile_err = 0
    total_tile_cnt = 0

    for csv_line in csv_lines[1:]:  # skip header
        parts = csv_line.split(',')
        if len(parts) != 15:
            continue
        # matrix,tile_m,tile_n,pred_total,pred_tiny,pred_sml,pred_lrg,pred_dns,pred_ful,actual_total,actual_tiny,actual_sml,actual_lrg,actual_dns,actual_ful
        p_total = int(parts[3])
        p_vals = {'tiny': int(parts[4]), 'sml': int(parts[5]), 'lrg': int(parts[6]), 'dns': int(parts[7]), 'ful': int(parts[8])}
        a_total = int(parts[9])
        a_vals = {'tiny': int(parts[10]), 'sml': int(parts[11]), 'lrg': int(parts[12]), 'dns': int(parts[13]), 'ful': int(parts[14])}

        for cat in ['tiny', 'sml', 'lrg', 'dns', 'ful']:
            category_errors[cat] += abs(p_vals[cat] - a_vals[cat])
            category_counts[cat] += a_vals[cat]

        if a_total > 0:
            total_tile_err += abs(p_total - a_total)
            total_tile_cnt += a_total

    if total_tile_cnt > 0:
        out_lines.append("Category-level accuracy (across all matrices and tile sizes):")
        out_lines.append("")
        out_lines.append(f"{'Category':<10} {'Total Actual':>12} {'Total |Error|':>14} {'Error Rate':>12}")
        out_lines.append(f"{'-'*10} {'-'*12} {'-'*14} {'-'*12}")

        for cat in ['tiny', 'sml', 'lrg', 'dns', 'ful']:
            actual = category_counts[cat]
            err = category_errors[cat]
            rate = (err / actual * 100) if actual > 0 else 0
            out_lines.append(f"{cat:<10} {actual:>12} {err:>14} {rate:>11.2f}%")

        out_lines.append("")
        out_lines.append(f"Total C tiles prediction error: {total_tile_err}/{total_tile_cnt} = {total_tile_err/total_tile_cnt*100:.2f}%")
    else:
        out_lines.append("No results to summarize.")

    out_lines.append("")

    # Write output files
    with open(OUTPUT, 'w') as f:
        f.write('\n'.join(out_lines))

    with open(CSV_FILE, 'w') as f:
        f.write('\n'.join(csv_lines))

    print()
    print(f"Results written to: {OUTPUT}")
    print(f"CSV data written to: {CSV_FILE}")

if __name__ == '__main__':
    main()
