#!/bin/bash
# Test script - process only one small folder

echo "Testing batch probe on a small folder..."
echo "This will process only the '10k-50k-0.1-1' folder (28 matrices)"

cd /home/stu1/marui/ada/probe7

# Run the Python script with a single folder test
python3 - << 'PYTHON_EOF'
import os
import subprocess
import re
from pathlib import Path
from datetime import datetime

DATASET_DIR = "/home/stu1/Dataset/training_dataset"
PROBE_EXEC = "/home/stu1/marui/ada/probe7/tile_probe"
OUTPUT_DIR = "/home/stu1/marui/ada/probe7/results"

# Test with just one small folder
TEST_FOLDER = Path(DATASET_DIR) / "10k-50k-0.1-1"

def parse_probe_output(output_text, matrix_name):
    """Parse tile_probe output and extract key information"""
    data = {
        'matrix_name': matrix_name,
        'rows': 0,
        'cols': 0,
        'nnz': 0,
        'symmetric': 'N/A',
        'density': 0.0,
        'avg_nnz_per_row': 0.0,
    }

    m = re.search(r'Rows \(m\): (\d+)', output_text)
    if m: data['rows'] = int(m.group(1))

    m = re.search(r'Cols \(n\): (\d+)', output_text)
    if m: data['cols'] = int(m.group(1))

    m = re.search(r'Nonzeros \(nnz\): (\d+)', output_text)
    if m: data['nnz'] = int(m.group(1))

    m = re.search(r'Symmetric: (\w+)', output_text)
    if m: data['symmetric'] = m.group(1)

    m = re.search(r'Density: ([\d.e\-+]+)', output_text)
    if m: data['density'] = float(m.group(1))

    m = re.search(r'Avg nnz/row: ([\d.]+)', output_text)
    if m: data['avg_nnz_per_row'] = float(m.group(1))

    tile_sizes = ['8x8', '16x16', '32x32']
    for size in tile_sizes:
        pattern = rf'\[Tile {size}\].*?numtile: (\d+)'
        m = re.search(pattern, output_text, re.DOTALL)
        if m: data[f'numtile_{size}'] = int(m.group(1))

        pattern = rf'\[Tile {size}\].*?density: ([\d.]+)%'
        m = re.search(pattern, output_text, re.DOTALL)
        if m: data[f'tile_density_{size}'] = float(m.group(1))

        pattern = rf'\[Tile {size}\].*?nnz_per_tile: avg=([\d.]+)'
        m = re.search(pattern, output_text, re.DOTALL)
        if m: data[f'nnz_per_tile_avg_{size}'] = float(m.group(1))

        pattern = rf'\[Tile {size}\].*?nnz_per_tile: avg=[\d.]+, max=(\d+)'
        m = re.search(pattern, output_text, re.DOTALL)
        if m: data[f'nnz_per_tile_max_{size}'] = int(m.group(1))

    return data

# Find mtx files
mtx_files = sorted(TEST_FOLDER.glob("*.mtx"))[:3]  # Test with first 3 matrices

print(f"Testing with {len(mtx_files)} matrices from {TEST_FOLDER.name}")

results = []
for i, mtx_file in enumerate(mtx_files, 1):
    print(f"[{i}/{len(mtx_files)}] {mtx_file.stem}...", end=' ', flush=True)
    try:
        result = subprocess.run([PROBE_EXEC, str(mtx_file)],
                              capture_output=True, text=True, timeout=60)
        data = parse_probe_output(result.stdout, mtx_file.stem)
        results.append(data)
        print("✓")
    except Exception as e:
        print(f"✗ {e}")

# Show results
print(f"\nParsed data for {len(results)} matrices:")
for data in results:
    print(f"  {data['matrix_name']}: {data['rows']}×{data['cols']}, nnz={data['nnz']}, density={data['density']:.2e}")
    print(f"    Tile 32×32: numtile={data.get('numtile_32x32', 'N/A')}")

print("\n✓ Test completed successfully!")
print("If this looks good, run the full script: /home/stu1/marui/ada/probe7/batch_probe_all.py")

PYTHON_EOF
