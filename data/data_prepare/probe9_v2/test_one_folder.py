#!/usr/bin/env python3
"""
Test batch probe on ONE complete folder
"""

import subprocess
import re
from pathlib import Path
from datetime import datetime

DATASET_DIR = "/home/stu1/Dataset/training_dataset"
PROBE_EXEC = "/home/stu1/marui/ada/probe7/tile_probe"
OUTPUT_DIR = "/home/stu1/marui/ada/probe7/results"

# Choose a small folder for testing
TEST_FOLDER = Path(DATASET_DIR) / "10k-50k-0.1-1"

def parse_probe_output(output_text, matrix_name):
    """Parse tile_probe output"""
    data = {'matrix_name': matrix_name}

    patterns = {
        'rows': (r'Rows \(m\): (\d+)', int),
        'cols': (r'Cols \(n\): (\d+)', int),
        'nnz': (r'Nonzeros \(nnz\): (\d+)', int),
        'symmetric': (r'Symmetric: (\w+)', str),
        'density': (r'Density: ([\d.e\-+]+)', float),
        'avg_nnz_per_row': (r'Avg nnz/row: ([\d.]+)', float),
    }

    for key, (pattern, dtype) in patterns.items():
        m = re.search(pattern, output_text)
        data[key] = dtype(m.group(1)) if m else (0 if dtype in [int, float] else 'N/A')

    # Parse tile stats for square tiles
    for size_key, size_display in [('8x8', '8×8'), ('16x16', '16×16'), ('32x32', '32×32')]:
        m = re.search(r'\[Tile ' + re.escape(size_display) + r'\].*?numtile: (\d+)', output_text, re.DOTALL)
        data[f'numtile_{size_key}'] = int(m.group(1)) if m else 0

        m = re.search(r'\[Tile ' + re.escape(size_display) + r'\].*?density: ([\d.]+)%', output_text, re.DOTALL)
        data[f'tile_density_{size_key}'] = float(m.group(1)) if m else 0.0

        m = re.search(r'\[Tile ' + re.escape(size_display) + r'\].*?nnz_per_tile: avg=([\d.]+)', output_text, re.DOTALL)
        data[f'nnz_per_tile_avg_{size_key}'] = float(m.group(1)) if m else 0.0

    return data

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"Testing complete folder processing: {TEST_FOLDER.name}")
print(f"Output: {OUTPUT_DIR}/{TEST_FOLDER.name}_probe_results.txt")
print("="*80)

# Find all matrices
mtx_files = sorted(TEST_FOLDER.glob("*.mtx"))
print(f"Found {len(mtx_files)} matrices\n")

results = []
for i, mtx_file in enumerate(mtx_files, 1):
    matrix_name = mtx_file.stem
    print(f"  [{i:2d}/{len(mtx_files)}] {matrix_name:<40}", end=' ', flush=True)

    try:
        result = subprocess.run(
            [PROBE_EXEC, str(mtx_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300
        )
        data = parse_probe_output(result.stdout, matrix_name)
        results.append(data)
        print(f"✓ ({data['rows']:6d}×{data['cols']:<6d}, tile32={data['numtile_32x32']:6d})")
    except Exception as e:
        print(f"✗ {e}")

# Write results
output_file = Path(OUTPUT_DIR) / f"{TEST_FOLDER.name}_probe_results.txt"

with open(output_file, 'w') as f:
    f.write("="*150 + "\n")
    f.write(f"Tile Probe Results - {TEST_FOLDER.name}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total matrices: {len(results)}\n")
    f.write("="*150 + "\n\n")

    # Basic info
    f.write("=== BASIC MATRIX INFORMATION ===\n\n")
    f.write(f"{'Matrix Name':<40} {'Rows':>10} {'Cols':>10} {'NNZ':>12} {'Density':>12} {'Symmetric':>10} {'Avg NNZ/Row':>12}\n")
    f.write("-"*150 + "\n")

    for data in results:
        f.write(f"{data['matrix_name']:<40} "
               f"{data['rows']:>10} "
               f"{data['cols']:>10} "
               f"{data['nnz']:>12} "
               f"{data['density']:>12.6e} "
               f"{data['symmetric']:>10} "
               f"{data['avg_nnz_per_row']:>12.2f}\n")

    # Tile statistics for each size
    for size in ['8x8', '16x16', '32x32']:
        f.write(f"\n\n=== TILE STATISTICS [{size}] ===\n\n")
        f.write(f"{'Matrix Name':<40} {'NumTile':>12} {'Tile Density':>15} {'Avg NNZ/Tile':>15}\n")
        f.write("-"*150 + "\n")

        for data in results:
            f.write(f"{data['matrix_name']:<40} "
                   f"{data.get(f'numtile_{size}', 0):>12} "
                   f"{data.get(f'tile_density_{size}', 0.0):>15.6f}% "
                   f"{data.get(f'nnz_per_tile_avg_{size}', 0.0):>15.2f}\n")

    # CSV format
    f.write("\n\n=== CSV FORMAT ===\n\n")
    f.write("matrix_name,rows,cols,nnz,density,symmetric,avg_nnz_per_row")
    for size in ['8x8', '16x16', '32x32']:
        f.write(f",numtile_{size},tile_density_{size},nnz_per_tile_avg_{size}")
    f.write("\n")

    for data in results:
        line = (f"{data['matrix_name']},"
               f"{data['rows']},"
               f"{data['cols']},"
               f"{data['nnz']},"
               f"{data['density']:.6e},"
               f"{data['symmetric']},"
               f"{data['avg_nnz_per_row']:.2f}")

        for size in ['8x8', '16x16', '32x32']:
            line += (f",{data.get(f'numtile_{size}', 0)}"
                    f",{data.get(f'tile_density_{size}', 0.0):.6f}"
                    f",{data.get(f'nnz_per_tile_avg_{size}', 0.0):.2f}")

        f.write(line + "\n")

    f.write("\n" + "="*150 + "\n")

print(f"\n{'='*80}")
print(f"✓ Results written to: {output_file}")
print(f"  Processed: {len(results)}/{len(mtx_files)} matrices")
print(f"{'='*80}\n")

# Show sample
print("Sample results (first 5 matrices):")
print(f"{'Matrix':<40} {'Size':>15} {'NNZ':>12} {'Tile32x32':>12}")
print("-"*80)
for data in results[:5]:
    print(f"{data['matrix_name']:<40} {data['rows']:>7}×{data['cols']:<7} {data['nnz']:>12} {data['numtile_32x32']:>12}")

print(f"\nIf this looks good, run the full batch: ./run_batch_probe.sh")
