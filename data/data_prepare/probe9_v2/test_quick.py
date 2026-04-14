#!/usr/bin/env python3
"""
Quick test of batch probe on 3 matrices
"""

import subprocess
import re
from pathlib import Path

PROBE_EXEC = "/home/stu1/marui/ada/probe7/tile_probe"
TEST_FOLDER = Path("/home/stu1/Dataset/training_dataset/10k-50k-0.1-1")

def parse_basic_info(output_text, matrix_name):
    """Parse basic matrix info from probe output"""
    data = {'matrix_name': matrix_name}

    patterns = {
        'rows': r'Rows \(m\): (\d+)',
        'cols': r'Cols \(n\): (\d+)',
        'nnz': r'Nonzeros \(nnz\): (\d+)',
        'density': r'Density: ([\d.e\-+]+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, output_text)
        data[key] = m.group(1) if m else 'N/A'

    # Parse tile stats for 32x32 (note: uses × symbol in output)
    m = re.search(r'\[Tile 32×32\].*?numtile: (\d+)', output_text, re.DOTALL)
    data['numtile_32x32'] = m.group(1) if m else 'N/A'

    return data

# Test with first 3 matrices
mtx_files = sorted(TEST_FOLDER.glob("*.mtx"))[:3]

print(f"Testing with {len(mtx_files)} matrices from {TEST_FOLDER.name}\n")

for i, mtx_file in enumerate(mtx_files, 1):
    print(f"[{i}/{len(mtx_files)}] {mtx_file.stem}...", end=' ', flush=True)
    try:
        result = subprocess.run(
            [PROBE_EXEC, str(mtx_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        data = parse_basic_info(result.stdout, mtx_file.stem)
        print(f"✓ ({data['rows']}×{data['cols']}, nnz={data['nnz']}, numtile_32x32={data['numtile_32x32']})")
    except Exception as e:
        print(f"✗ {e}")

print("\n✓ Test completed!")
print("To process all folders, run: python3 /home/stu1/marui/ada/probe7/batch_probe_all.py")
