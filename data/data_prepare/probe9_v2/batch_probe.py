#!/usr/bin/env python3
"""
batch_probe.py - 批量执行 tile_probe 并汇总结果到表格

用法:
    python batch_probe.py <矩阵文件夹路径> [输出文件.txt]

示例:
    python batch_probe.py /home/stu1/marui/ada/mtx results.txt
    python batch_probe.py /home/stu1/marui/ada/TileSpGEMMDataset
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

# tile_probe 可执行文件路径
TILE_PROBE = "/home/stu1/marui/ada/probe7/tile_probe"

# 7 种 tile 尺寸
TILE_SIZES = ["8x8", "16x8", "8x16", "16x16", "16x32", "32x16", "32x32"]


def run_tile_probe(mtx_path):
    """运行 tile_probe 并返回输出"""
    try:
        result = subprocess.run(
            [TILE_PROBE, mtx_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # Python 3.6 兼容写法 (等同于 text=True)
            timeout=300  # 5 分钟超时
        )
        if result.returncode != 0:
            return result.stderr, result.returncode
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "Timeout", -1
    except FileNotFoundError:
        return "tile_probe not found: {}".format(TILE_PROBE), -2
    except Exception as e:
        return "Exception: {}".format(str(e)), -3


def parse_matrix_info(output):
    """解析矩阵基本信息"""
    info = {}

    # Rows (m): 83334
    m = re.search(r'Rows \(m\): (\d+)', output)
    info['m'] = int(m.group(1)) if m else 0

    # Cols (n): 83334
    n = re.search(r'Cols \(n\): (\d+)', output)
    info['n'] = int(n.group(1)) if n else 0

    # Nonzeros (nnz): 6010480
    nnz = re.search(r'Nonzeros \(nnz\): (\d+)', output)
    info['nnz'] = int(nnz.group(1)) if nnz else 0

    # Density: 8.654281e-04 (0.086543%)
    density = re.search(r'Density: ([\d.e+-]+)', output)
    info['density'] = float(density.group(1)) if density else 0

    # Avg nnz/row: 72.12
    avg_nnz = re.search(r'Avg nnz/row: ([\d.]+)', output)
    info['avg_nnz_per_row'] = float(avg_nnz.group(1)) if avg_nnz else 0

    # Probe time: 5.43 ms
    probe_time = re.search(r'Probe time: ([\d.]+) ms', output)
    info['probe_time_ms'] = float(probe_time.group(1)) if probe_time else 0

    return info


def parse_tile_stats(output, tile_size):
    """解析特定 tile 尺寸的统计信息"""
    stats = {}

    # 找到对应 tile 尺寸的部分
    pattern = rf'\[Tile {tile_size.replace("x", "×")}\](.*?)(?=\[Tile|\n=== Cross-Size)'
    match = re.search(pattern, output, re.DOTALL)
    if not match:
        return None

    section = match.group(1)

    # Grid: 10417 × 10417 tiles
    grid = re.search(r'Grid: (\d+) × (\d+) tiles', section)
    stats['tilem'] = int(grid.group(1)) if grid else 0
    stats['tilen'] = int(grid.group(2)) if grid else 0

    # numtile: 272897 (density: 0.251486%)
    numtile = re.search(r'numtile: (\d+) \(density: ([\d.]+)%\)', section)
    stats['numtile'] = int(numtile.group(1)) if numtile else 0
    stats['tile_density'] = float(numtile.group(2)) if numtile else 0

    # nnz_per_tile: avg=22.02, max=64, min=1, std=17.48, cv=0.794
    nnz_stats = re.search(r'nnz_per_tile: avg=([\d.]+), max=(\d+), min=(\d+), std=([\d.]+), cv=([\d.]+)', section)
    if nnz_stats:
        stats['nnz_per_tile_avg'] = float(nnz_stats.group(1))
        stats['nnz_per_tile_max'] = int(nnz_stats.group(2))
        stats['nnz_per_tile_min'] = int(nnz_stats.group(3))
        stats['nnz_per_tile_std'] = float(nnz_stats.group(4))
        stats['nnz_per_tile_cv'] = float(nnz_stats.group(5))

    # tile_fill_ratio: avg=0.3440, max=1.0000
    fill = re.search(r'tile_fill_ratio: avg=([\d.]+), max=([\d.]+)', section)
    if fill:
        stats['tile_fill_avg'] = float(fill.group(1))
        stats['tile_fill_max'] = float(fill.group(2))

    # tiles_per_row: avg=26.20, max=67, min=1, std=10.29, cv=0.393
    tpr = re.search(r'tiles_per_row: avg=([\d.]+), max=(\d+), min=(\d+), std=([\d.]+), cv=([\d.]+)', section)
    if tpr:
        stats['tiles_per_row_avg'] = float(tpr.group(1))
        stats['tiles_per_row_max'] = int(tpr.group(2))
        stats['tiles_per_row_min'] = int(tpr.group(3))
        stats['tiles_per_row_std'] = float(tpr.group(4))
        stats['tiles_per_row_cv'] = float(tpr.group(5))

    # empty_tile_row_ratio: 0.0000
    empty = re.search(r'empty_tile_row_ratio: ([\d.]+)', section)
    stats['empty_row_ratio'] = float(empty.group(1)) if empty else 0

    # nnz_histogram: [1]=1952, [2-4)=35192, [4-8)=21482, [8-16)=78405, [16-32)=60602, [32-64)=75264, [64-128)=0, [128+]=0
    hist = re.search(r'nnz_histogram: \[1\]=(\d+), \[2-4\)=(\d+), \[4-8\)=(\d+), \[8-16\)=(\d+), \[16-32\)=(\d+), \[32-64\)=(\d+), \[64-128\)=(\d+), \[128\+\]=(\d+)', section)
    if hist:
        stats['hist_1'] = int(hist.group(1))
        stats['hist_2_4'] = int(hist.group(2))
        stats['hist_4_8'] = int(hist.group(3))
        stats['hist_8_16'] = int(hist.group(4))
        stats['hist_16_32'] = int(hist.group(5))
        stats['hist_32_64'] = int(hist.group(6))
        stats['hist_64_128'] = int(hist.group(7))
        stats['hist_128_plus'] = int(hist.group(8))

        # 计算直方图百分比
        total = sum([stats['hist_1'], stats['hist_2_4'], stats['hist_4_8'], stats['hist_8_16'],
                     stats['hist_16_32'], stats['hist_32_64'], stats['hist_64_128'], stats['hist_128_plus']])
        if total > 0:
            stats['hist_1_pct'] = stats['hist_1'] / total * 100
            stats['hist_2_4_pct'] = stats['hist_2_4'] / total * 100
            stats['hist_4_8_pct'] = stats['hist_4_8'] / total * 100
            stats['hist_8_16_pct'] = stats['hist_8_16'] / total * 100
            stats['hist_16_32_pct'] = stats['hist_16_32'] / total * 100
            stats['hist_32_64_pct'] = stats['hist_32_64'] / total * 100
            stats['hist_64_128_pct'] = stats['hist_64_128'] / total * 100
            stats['hist_128_plus_pct'] = stats['hist_128_plus'] / total * 100
        else:
            for k in ['hist_1_pct', 'hist_2_4_pct', 'hist_4_8_pct', 'hist_8_16_pct',
                      'hist_16_32_pct', 'hist_32_64_pct', 'hist_64_128_pct', 'hist_128_plus_pct']:
                stats[k] = 0.0

    return stats


def find_mtx_files(folder_path):
    """递归查找所有 .mtx 文件"""
    mtx_files = []
    folder = Path(folder_path)

    for f in folder.rglob("*.mtx"):
        mtx_files.append(str(f))

    return sorted(mtx_files)


def format_number(n, width=12):
    """格式化数字"""
    if isinstance(n, float):
        if abs(n) < 0.01 and n != 0:
            return f"{n:>{width}.2e}"
        return f"{n:>{width}.4f}"
    return f"{n:>{width}}"


def main():
    if len(sys.argv) < 2:
        print("用法: python batch_probe.py <矩阵文件夹路径> [输出文件.txt]")
        print("示例: python batch_probe.py /home/stu1/marui/ada/mtx results.txt")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "probe_results.txt"

    # 检查 tile_probe 是否存在
    if not os.path.exists(TILE_PROBE):
        print(f"错误: tile_probe 程序不存在: {TILE_PROBE}")
        sys.exit(1)

    # 查找所有矩阵文件
    mtx_files = find_mtx_files(folder_path)
    if not mtx_files:
        print(f"错误: 在 {folder_path} 中没有找到 .mtx 文件")
        sys.exit(1)

    print(f"找到 {len(mtx_files)} 个矩阵文件")
    print(f"结果将保存到: {output_file}")
    print("=" * 80)

    # 收集所有结果
    all_results = []

    for i, mtx_file in enumerate(mtx_files):
        matrix_name = os.path.basename(mtx_file)
        print(f"[{i+1}/{len(mtx_files)}] 处理: {matrix_name}...", end=" ", flush=True)

        output, retcode = run_tile_probe(mtx_file)

        if retcode != 0 or output is None:
            print(f"失败 (错误码: {retcode})")
            continue

        # 解析结果
        try:
            matrix_info = parse_matrix_info(output)
            matrix_info['name'] = matrix_name

            tile_stats = {}
            for ts in TILE_SIZES:
                stats = parse_tile_stats(output, ts)
                if stats:
                    tile_stats[ts] = stats

            if tile_stats:
                all_results.append((matrix_info, tile_stats))
                print(f"完成 (probe_time: {matrix_info['probe_time_ms']:.2f}ms)")
            else:
                print("解析失败")
        except Exception as e:
            print(f"解析错误: {e}")

    if not all_results:
        print("没有成功处理任何矩阵")
        sys.exit(1)

    # 生成输出文件
    print("\n" + "=" * 80)
    print(f"正在生成输出文件: {output_file}")

    with open(output_file, 'w') as f:
        # 写入标题
        f.write("=" * 120 + "\n")
        f.write("                    tile_probe 批量处理结果汇总表\n")
        f.write("=" * 120 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"矩阵文件夹: {folder_path}\n")
        f.write(f"处理矩阵数: {len(all_results)}\n")
        f.write("=" * 120 + "\n\n")

        # ========================================================================
        # 表1: 矩阵基本信息汇总
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表1: 矩阵基本信息汇总\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<30} {'行数(m)':>12} {'列数(n)':>12} {'非零元(nnz)':>14} {'密度':>14} {'avg_nnz/row':>12} {'探测时间(ms)':>14}\n"
        f.write(header)
        f.write("-" * 120 + "\n")

        for matrix_info, _ in all_results:
            line = f"{matrix_info['name']:<30} {matrix_info['m']:>12} {matrix_info['n']:>12} {matrix_info['nnz']:>14} {matrix_info['density']:>14.6e} {matrix_info['avg_nnz_per_row']:>12.2f} {matrix_info['probe_time_ms']:>14.2f}\n"
            f.write(line)

        f.write("\n\n")

        # ========================================================================
        # 表2: 各 tile 尺寸的 numtile 和 tile_density 汇总
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表2: numtile 汇总 (各 tile 尺寸的非空 tile 数量)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['numtile']:>12}"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        # ========================================================================
        # 表3: tile_density 汇总 (%)
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表3: tile_density 汇总 (非空 tile 占比 %)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['tile_density']:>11.4f}%"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        # ========================================================================
        # 表4: nnz_per_tile_avg 汇总
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表4: nnz_per_tile_avg 汇总 (每个非空 tile 的平均 nnz)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['nnz_per_tile_avg']:>12.2f}"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        # ========================================================================
        # 表5: tile_fill_avg 汇总 (tile 平均填充率)
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表5: tile_fill_avg 汇总 (tile 平均填充率 = nnz_per_tile_avg / tile_cap)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['tile_fill_avg']:>12.4f}"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        # ========================================================================
        # 表6-12: 各尺寸的直方图百分比
        # ========================================================================
        hist_bins = [
            ('hist_1_pct', '[1]'),
            ('hist_2_4_pct', '[2-4)'),
            ('hist_4_8_pct', '[4-8)'),
            ('hist_8_16_pct', '[8-16)'),
            ('hist_16_32_pct', '[16-32)'),
            ('hist_32_64_pct', '[32-64)'),
            ('hist_64_128_pct', '[64-128)'),
            ('hist_128_plus_pct', '[128+]'),
        ]

        for ts in TILE_SIZES:
            f.write("=" * 120 + "\n")
            f.write(f"表: {ts} tile 的 nnz 直方图百分比 (%)\n")
            f.write("=" * 120 + "\n")

            header = f"{'矩阵名称':<25}"
            for _, bin_name in hist_bins:
                header += f" {bin_name:>10}"
            f.write(header + "\n")
            f.write("-" * 120 + "\n")

            for matrix_info, tile_stats in all_results:
                if ts not in tile_stats:
                    continue
                line = f"{matrix_info['name']:<25}"
                for key, _ in hist_bins:
                    if key in tile_stats[ts]:
                        line += f" {tile_stats[ts][key]:>9.2f}%"
                    else:
                        line += f" {'N/A':>10}"
                f.write(line + "\n")

            f.write("\n\n")

        # ========================================================================
        # 表: 详细统计 (tiles_per_row, nnz_per_tile 的 cv/std)
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("表: tiles_per_row_cv 汇总 (每 tile 行非空 tile 数的变异系数)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['tiles_per_row_cv']:>12.4f}"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        f.write("=" * 120 + "\n")
        f.write("表: nnz_per_tile_cv 汇总 (tile 内 nnz 的变异系数)\n")
        f.write("=" * 120 + "\n")

        header = f"{'矩阵名称':<25}"
        for ts in TILE_SIZES:
            header += f" {ts:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for matrix_info, tile_stats in all_results:
            line = f"{matrix_info['name']:<25}"
            for ts in TILE_SIZES:
                if ts in tile_stats:
                    line += f" {tile_stats[ts]['nnz_per_tile_cv']:>12.4f}"
                else:
                    line += f" {'N/A':>12}"
            f.write(line + "\n")

        f.write("\n\n")

        # ========================================================================
        # CSV 格式输出 (便于后续处理)
        # ========================================================================
        f.write("=" * 120 + "\n")
        f.write("附录: CSV 格式数据 (便于导入 Excel 或 Python 进一步分析)\n")
        f.write("=" * 120 + "\n\n")

        # CSV header
        csv_header = "matrix,m,n,nnz,density,tile_size,numtile,tile_density,nnz_per_tile_avg,nnz_per_tile_max,nnz_per_tile_min,nnz_per_tile_std,nnz_per_tile_cv,tile_fill_avg,tile_fill_max,tiles_per_row_avg,tiles_per_row_max,tiles_per_row_min,tiles_per_row_std,tiles_per_row_cv,empty_row_ratio,hist_1,hist_2_4,hist_4_8,hist_8_16,hist_16_32,hist_32_64,hist_64_128,hist_128_plus,hist_1_pct,hist_2_4_pct,hist_4_8_pct,hist_8_16_pct,hist_16_32_pct,hist_32_64_pct,hist_64_128_pct,hist_128_plus_pct"
        f.write(csv_header + "\n")

        for matrix_info, tile_stats in all_results:
            for ts in TILE_SIZES:
                if ts not in tile_stats:
                    continue
                s = tile_stats[ts]
                csv_row = f"{matrix_info['name']},{matrix_info['m']},{matrix_info['n']},{matrix_info['nnz']},{matrix_info['density']:.6e},"
                csv_row += f"{ts},{s.get('numtile',0)},{s.get('tile_density',0):.6f},"
                csv_row += f"{s.get('nnz_per_tile_avg',0):.4f},{s.get('nnz_per_tile_max',0)},{s.get('nnz_per_tile_min',0)},"
                csv_row += f"{s.get('nnz_per_tile_std',0):.4f},{s.get('nnz_per_tile_cv',0):.4f},"
                csv_row += f"{s.get('tile_fill_avg',0):.6f},{s.get('tile_fill_max',0):.6f},"
                csv_row += f"{s.get('tiles_per_row_avg',0):.4f},{s.get('tiles_per_row_max',0)},{s.get('tiles_per_row_min',0)},"
                csv_row += f"{s.get('tiles_per_row_std',0):.4f},{s.get('tiles_per_row_cv',0):.4f},"
                csv_row += f"{s.get('empty_row_ratio',0):.6f},"
                csv_row += f"{s.get('hist_1',0)},{s.get('hist_2_4',0)},{s.get('hist_4_8',0)},{s.get('hist_8_16',0)},"
                csv_row += f"{s.get('hist_16_32',0)},{s.get('hist_32_64',0)},{s.get('hist_64_128',0)},{s.get('hist_128_plus',0)},"
                csv_row += f"{s.get('hist_1_pct',0):.4f},{s.get('hist_2_4_pct',0):.4f},{s.get('hist_4_8_pct',0):.4f},{s.get('hist_8_16_pct',0):.4f},"
                csv_row += f"{s.get('hist_16_32_pct',0):.4f},{s.get('hist_32_64_pct',0):.4f},{s.get('hist_64_128_pct',0):.4f},{s.get('hist_128_plus_pct',0):.4f}"
                f.write(csv_row + "\n")

        f.write("\n" + "=" * 120 + "\n")
        f.write("报告结束\n")
        f.write("=" * 120 + "\n")

    print(f"完成! 结果已保存到: {output_file}")
    print(f"共处理 {len(all_results)} 个矩阵")


if __name__ == "__main__":
    main()
