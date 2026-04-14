#!/usr/bin/env python3
"""
Remove >200k matrices from probe9 and probeC result files,
EXCEPT for matrices in A100/H200 test datasets.
"""

import re
import os
from pathlib import Path

# 必须保留的>200k测试集矩阵
KEEP_MATRICES = {'mac_econ_fwd500', 'webbase-1M', 'mc2depi', 'af_shell10', 'pwtk'}

# 需要处理的>200k文件夹前缀
BIG_PREFIXES = ['200k-500k', '500k-1M', '1M-2M']

# 4个结果目录
DIRS = [
    ('/home/stu1/marui/probe9_v2/results_AA', '_probe_results.txt'),
    ('/home/stu1/marui/probe9_v2/results_AAT', '_probe_results.txt'),
    ('/home/stu1/marui/probeC_v2/result_AA', '_probeC_results.txt'),
    ('/home/stu1/marui/probeC_v2/result_AAT', '_probeC_results.txt'),
]


def is_big_folder(filename):
    """Check if filename corresponds to a >200k folder."""
    for prefix in BIG_PREFIXES:
        if filename.startswith(prefix):
            return True
    return False


def filter_probe9_file(filepath):
    """Filter probe9 result file: remove >200k matrices except KEEP list."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    skip_matrix = False
    removed = []
    kept = []
    current_name = ''

    # Track CSV section separately
    in_csv_section = False
    csv_header_line = ''
    csv_data_lines = []
    pre_csv_lines = []

    for line in lines:
        stripped = line.strip()

        if '=== CSV FORMAT' in stripped:
            in_csv_section = True
            pre_csv_lines = new_lines[:]
            new_lines.append(line)
            continue

        if in_csv_section:
            # In CSV section: filter data lines
            if stripped.startswith('matrix_name') or stripped.startswith('=') or stripped.startswith('End') or not stripped:
                new_lines.append(line)
            elif ',' in stripped:
                # CSV data line: first field is matrix_name
                name = stripped.split(',')[0]
                rows_str = stripped.split(',')[1] if len(stripped.split(',')) > 1 else '0'
                try:
                    rows = int(rows_str)
                except:
                    rows = 0
                if rows <= 200000 or name in KEEP_MATRICES:
                    new_lines.append(line)
                # else skip
            else:
                new_lines.append(line)
            continue

        # Pre-CSV section: filter matrix blocks
        m = re.match(r'### (\S+)', stripped)
        if m:
            current_name = m.group(1)
            # Check if matrix has rows > 200k
            rows_match = re.search(r'(\d+)x\d+', stripped)
            if rows_match:
                rows = int(rows_match.group(1))
                if rows > 200000 and current_name not in KEEP_MATRICES:
                    skip_matrix = True
                    removed.append(current_name)
                    continue
                else:
                    skip_matrix = False
                    if rows > 200000:
                        kept.append(current_name)
            else:
                skip_matrix = False

        if skip_matrix:
            continue

        # Update Summary line
        if stripped.startswith('Summary:'):
            # Will rewrite after counting
            pass

        new_lines.append(line)

    # Update total count in header and summary
    final_lines = []
    for line in new_lines:
        if line.strip().startswith('Total matrices'):
            # Recount from ### lines
            count = sum(1 for l in new_lines if l.strip().startswith('### ') and 'TIMEOUT' not in l and 'PARSE_ERROR' not in l)
            final_lines.append(re.sub(r'Total matrices.*: \d+', f'Total matrices: {count}', line))
        elif line.strip().startswith('Summary:'):
            ok = sum(1 for l in new_lines if l.strip().startswith('### ') and 'TIMEOUT' not in l and 'PARSE_ERROR' not in l)
            fail = sum(1 for l in new_lines if 'TIMEOUT' in l or 'PARSE_ERROR' in l)
            final_lines.append(f"Summary: {ok} OK, {fail} failed, {ok+fail} total\n")
        else:
            final_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(final_lines)

    return removed, kept


def main():
    total_removed = 0
    total_kept = 0

    for dir_path, suffix in DIRS:
        if not os.path.exists(dir_path):
            print(f"SKIP: {dir_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"处理目录: {dir_path}")
        print(f"{'='*60}")

        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(suffix):
                continue
            folder_name = fname.replace(suffix, '')

            if not is_big_folder(folder_name):
                continue

            filepath = os.path.join(dir_path, fname)
            removed, kept = filter_probe9_file(filepath)
            total_removed += len(removed)
            total_kept += len(kept)

            if removed or kept:
                print(f"  {fname}:")
                if removed:
                    print(f"    删除 {len(removed)} 个矩阵: {removed[:5]}{'...' if len(removed)>5 else ''}")
                if kept:
                    print(f"    保留(测试集) {len(kept)} 个矩阵: {kept}")
            else:
                print(f"  {fname}: 无>200k矩阵需删除")

    print(f"\n{'='*60}")
    print(f"总计: 删除 {total_removed} 个矩阵条目, 保留(测试集例外) {total_kept} 个")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
