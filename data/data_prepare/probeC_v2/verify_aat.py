#!/usr/bin/env python3
"""
Task 2: 验证probeC AAT预测正确性，统计AA与AAT预测时间
使用better_150.txt中的矩阵进行测试
"""

import subprocess
import re
import os
import time
import sys

PROBE_EXEC = "/home/stu1/marui/probeC_v2/probeC"
DATASET_DIR = "/home/stu1/Dataset/training_dataset"
DATASET_NOSQ = "/home/stu1/Dataset/training_dataset_nosquare"
BETTER_FILE = "/home/stu1/marui/lightGBM_A_AT/better_150.txt"
OUTPUT_FILE = "/home/stu1/marui/probeC_v2/probeAAT=C_time.txt"
TIMEOUT_SEC = 300

def find_matrix(name):
    """Find matrix .mtx file in dataset directories."""
    for base in [DATASET_DIR, DATASET_NOSQ]:
        for root, dirs, files in os.walk(base):
            for f in files:
                # Match exactly: name.mtx or name_suffix.mtx where suffix starts with _
                if f == name + '.mtx':
                    return os.path.join(root, f)
    # Also check TileSpGEMMDataset
    p = f"/home/stu1/marui/TileSpGEMMDataset/{name}.mtx"
    if os.path.exists(p):
        return p
    return None

def run_probe(matrix_path, mode='aa'):
    """Run probeC and extract CSV data + timing."""
    if mode == 'aat':
        cmd = [PROBE_EXEC, '--aat', matrix_path]
    else:
        cmd = [PROBE_EXEC, matrix_path]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=TIMEOUT_SEC)
        output = result.stdout

        # Extract estimate time
        est_match = re.search(r'Estimate time:\s+([\d.]+)\s*ms', output)
        est_time = float(est_match.group(1)) if est_match else -1

        # Extract build time
        build_match = re.search(r'Build time:\s+([\d.]+)\s*ms', output)
        build_time = float(build_match.group(1)) if build_match else -1

        # Extract transpose time (AAT only)
        trans_match = re.search(r'Transpose time:\s+([\d.]+)\s*ms', output)
        trans_time = float(trans_match.group(1)) if trans_match else 0

        # Extract CSV lines
        csv_lines = []
        in_csv = False
        for line in output.split('\n'):
            if '=== CSV ===' in line:
                in_csv = True
                continue
            if in_csv and line.strip() and not line.startswith('tile_m'):
                csv_lines.append(line.strip())

        return {
            'build_time': build_time,
            'estimate_time': est_time,
            'transpose_time': trans_time,
            'total_probe_time': build_time + est_time + trans_time,
            'csv_lines': csv_lines,
            'success': True,
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'TIMEOUT'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def parse_csv_line(line):
    """Parse a CSV line into dict."""
    parts = line.split(',')
    return {
        'tile_m': int(parts[0]),
        'sml': int(parts[1]),
        'lrg': int(parts[2]),
        'dns': int(parts[3]),
        'ful': int(parts[4]),
        'numblkC': int(parts[5]),
        'total_flops': int(parts[6]),
    }

# Parse better_150.txt
matrices = []
with open(BETTER_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-') or line.startswith('matrix_name'):
            continue
        parts = line.split()
        matrices.append({'name': parts[0], 'category': parts[1]})

print(f"=" * 80)
print(f"Task 2: probeC AAT 验证与计时")
print(f"=" * 80)
print(f"矩阵数: {len(matrices)}")

# Run tests
results = []
success_count = 0
fail_count = 0
sym_count = 0

for idx, m in enumerate(matrices):
    name = m['name']
    mtx_path = find_matrix(name)

    if mtx_path is None:
        print(f"  [{idx+1}/{len(matrices)}] {name}: 未找到mtx文件, 跳过")
        fail_count += 1
        continue

    # Run AA
    aa_result = run_probe(mtx_path, 'aa')
    if not aa_result['success']:
        print(f"  [{idx+1}/{len(matrices)}] {name}: AA失败 ({aa_result.get('error', 'unknown')})")
        fail_count += 1
        continue

    # Run AAT
    aat_result = run_probe(mtx_path, 'aat')
    if not aat_result['success']:
        print(f"  [{idx+1}/{len(matrices)}] {name}: AAT失败 ({aat_result.get('error', 'unknown')})")
        fail_count += 1
        continue

    # Check: for symmetric matrices, AA and AAT should be identical
    is_symmetric = (aa_result['csv_lines'] == aat_result['csv_lines'])
    if is_symmetric:
        sym_count += 1

    # Validate: both should have 3 CSV lines (tile_m=8,16,32)
    aa_ok = len(aa_result['csv_lines']) == 3
    aat_ok = len(aat_result['csv_lines']) == 3

    status = "OK"
    if not aa_ok or not aat_ok:
        status = "CSV_MISMATCH"

    results.append({
        'name': name,
        'category': m['category'],
        'aa_build': aa_result['build_time'],
        'aa_estimate': aa_result['estimate_time'],
        'aa_total': aa_result['total_probe_time'],
        'aat_transpose': aat_result['transpose_time'],
        'aat_build': aat_result['build_time'],
        'aat_estimate': aat_result['estimate_time'],
        'aat_total': aat_result['total_probe_time'],
        'symmetric': is_symmetric,
        'status': status,
        'aa_csv': aa_result['csv_lines'],
        'aat_csv': aat_result['csv_lines'],
    })
    success_count += 1

    if (idx + 1) % 20 == 0:
        print(f"  进度: {idx+1}/{len(matrices)} | 成功: {success_count} | 失败: {fail_count}")

print(f"\n完成: 成功={success_count}, 失败={fail_count}, 对称矩阵={sym_count}")

# Write output
with open(OUTPUT_FILE, 'w') as f:
    f.write(f"# probeC AA vs AAT 预测时间统计\n")
    f.write(f"# 总矩阵数: {len(matrices)}, 成功: {success_count}, 失败: {fail_count}\n")
    f.write(f"# 对称矩阵(AA==AAT): {sym_count}\n")
    f.write(f"#\n")
    f.write(f"{'matrix_name':<40} {'category':<25} {'sym':>3} | "
            f"{'aa_build':>10} {'aa_est':>10} {'aa_total':>10} | "
            f"{'aat_trans':>10} {'aat_build':>10} {'aat_est':>10} {'aat_total':>10} | {'status':>6}\n")
    f.write("-" * 170 + "\n")

    aa_totals = []
    aat_totals = []

    for r in results:
        sym_str = "Yes" if r['symmetric'] else "No"
        f.write(f"{r['name']:<40} {r['category']:<25} {sym_str:>3} | "
                f"{r['aa_build']:>10.2f} {r['aa_estimate']:>10.2f} {r['aa_total']:>10.2f} | "
                f"{r['aat_transpose']:>10.2f} {r['aat_build']:>10.2f} {r['aat_estimate']:>10.2f} {r['aat_total']:>10.2f} | "
                f"{r['status']:>6}\n")
        aa_totals.append(r['aa_total'])
        aat_totals.append(r['aat_total'])

    f.write("\n")
    f.write(f"# 统计摘要 (单位: ms)\n")
    if aa_totals:
        import numpy as np
        aa_arr = np.array(aa_totals)
        aat_arr = np.array(aat_totals)
        f.write(f"# AA  预测时间: 平均={aa_arr.mean():.2f}ms, 中位数={np.median(aa_arr):.2f}ms, "
                f"最大={aa_arr.max():.2f}ms, 最小={aa_arr.min():.2f}ms\n")
        f.write(f"# AAT 预测时间: 平均={aat_arr.mean():.2f}ms, 中位数={np.median(aat_arr):.2f}ms, "
                f"最大={aat_arr.max():.2f}ms, 最小={aat_arr.min():.2f}ms\n")
        f.write(f"# AAT/AA 比值:  平均={np.mean(aat_arr/aa_arr):.2f}x\n")
        f.write(f"# <10ms的矩阵: AA={np.sum(aa_arr<10)}/{len(aa_arr)}, "
                f"AAT={np.sum(aat_arr<10)}/{len(aat_arr)}\n")

print(f"\n结果已保存到: {OUTPUT_FILE}")

# Print summary
if aa_totals:
    import numpy as np
    aa_arr = np.array(aa_totals)
    aat_arr = np.array(aat_totals)
    print(f"\nAA  预测时间: 平均={aa_arr.mean():.2f}ms, 中位数={np.median(aa_arr):.2f}ms, 最大={aa_arr.max():.2f}ms")
    print(f"AAT 预测时间: 平均={aat_arr.mean():.2f}ms, 中位数={np.median(aat_arr):.2f}ms, 最大={aat_arr.max():.2f}ms")
    print(f"AAT/AA 比值:  平均={np.mean(aat_arr/aa_arr):.2f}x")
    print(f"<10ms的矩阵: AA={np.sum(aa_arr<10)}/{len(aa_arr)}, AAT={np.sum(aat_arr<10)}/{len(aat_arr)}")

# Spot check: verify non-symmetric matrices have different results
print(f"\n非对称矩阵验证 (AA != AAT):")
non_sym = [r for r in results if not r['symmetric']]
print(f"  非对称矩阵数: {len(non_sym)}")
for r in non_sym[:5]:
    aa_d = parse_csv_line(r['aa_csv'][0]) if r['aa_csv'] else {}
    aat_d = parse_csv_line(r['aat_csv'][0]) if r['aat_csv'] else {}
    print(f"  {r['name']:30s} AA_numblkC={aa_d.get('numblkC','?'):>10} AAT_numblkC={aat_d.get('numblkC','?'):>10}")
