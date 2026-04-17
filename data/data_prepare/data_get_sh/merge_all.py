#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_all_data.py - Merge AA and AAT data to all_data directory
Data split: train 600, val 300, test 150
"""

import pandas as pd
import os
import shutil

BASE_DIR = "../data_get"

# 源数据目录
A100_AA_SRC = os.path.join(BASE_DIR, "A100_AA")
H200_AA_SRC = os.path.join(BASE_DIR, "H200_AA")
A100_AAT_SRC = os.path.join(BASE_DIR, "A100_AAT")
H200_AAT_SRC = os.path.join(BASE_DIR, "H200_AAT")

# 输出目录
ALL_DATA_AA = os.path.join(BASE_DIR, "all_data_AA")
ALL_DATA_AAT = os.path.join(BASE_DIR, "all_data_AAT")
ALL_DATA = os.path.join(BASE_DIR, "all_data")


def merge_aa_data():
    """合并A100和H200的AA数据到all_data_AA目录"""
    print("\n[1/3] 合并AA数据...")
    
    os.makedirs(ALL_DATA_AA, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        a100_path = os.path.join(A100_AA_SRC, f"{split}.csv")
        h200_path = os.path.join(H200_AA_SRC, f"{split}.csv")
        
        dfs = []
        if os.path.exists(a100_path):
            df_a100 = pd.read_csv(a100_path)
            dfs.append(df_a100)
            print(f"  A100 AA {split}: {len(df_a100)} 条")
        
        if os.path.exists(h200_path):
            df_h200 = pd.read_csv(h200_path)
            dfs.append(df_h200)
            print(f"  H200 AA {split}: {len(df_h200)} 条")
        
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(ALL_DATA_AA, f"{split}.csv")
            merged.to_csv(out_path, index=False)
            print(f"  -> 合并AA {split}: {len(merged)} 条 -> {out_path}")
    
    return True


def merge_aat_data():
    """合并A100和H200的AAT数据到all_data_AAT目录"""
    print("\n[2/3] 合并AAT数据...")
    
    os.makedirs(ALL_DATA_AAT, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        a100_path = os.path.join(A100_AAT_SRC, f"{split}.csv")
        h200_path = os.path.join(H200_AAT_SRC, f"{split}.csv")
        
        dfs = []
        if os.path.exists(a100_path):
            df_a100 = pd.read_csv(a100_path)
            dfs.append(df_a100)
            print(f"  A100 AAT {split}: {len(df_a100)} 条")
        
        if os.path.exists(h200_path):
            df_h200 = pd.read_csv(h200_path)
            dfs.append(df_h200)
            print(f"  H200 AAT {split}: {len(df_h200)} 条")
        
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(ALL_DATA_AAT, f"{split}.csv")
            merged.to_csv(out_path, index=False)
            print(f"  -> 合并AAT {split}: {len(merged)} 条 -> {out_path}")
    
    return True


def merge_all():
    """合并AA和AAT数据到all_data目录（AA在前，AAT在后）"""
    print("\n[3/3] 合并AA+AAT数据到all_data...")
    
    os.makedirs(ALL_DATA, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        aa_path = os.path.join(ALL_DATA_AA, f"{split}.csv")
        aat_path = os.path.join(ALL_DATA_AAT, f"{split}.csv")
        
        dfs = []
        aa_count = 0
        
        # AA数据在前
        if os.path.exists(aa_path):
            df_aa = pd.read_csv(aa_path)
            dfs.append(df_aa)
            aa_count = len(df_aa)
            print(f"  AA {split}: {len(df_aa)} 条")
        
        # AAT数据在后
        if os.path.exists(aat_path):
            df_aat = pd.read_csv(aat_path)
            dfs.append(df_aat)
            print(f"  AAT {split}: {len(df_aat)} 条")
        
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(ALL_DATA, f"{split}.csv")
            merged.to_csv(out_path, index=False)
            print(f"  -> 合并 {split}: {len(merged)} 条 (AA: {aa_count}, AAT: {len(merged)-aa_count}) -> {out_path}")
    
    return True


def main():
    print("=" * 70)
    print("  合并AA和AAT数据")
    print("=" * 70)
    
    # 1. 合并AA数据
    merge_aa_data()
    
    # 2. 合并AAT数据
    merge_aat_data()
    
    # 3. 合并AA+AAT数据
    merge_all()
    
    print("\n" + "=" * 70)
    print("  数据合并完成!")
    print("=" * 70)
    
    # 打印统计信息
    print("\n数据统计:")
    for split in ['train', 'val', 'test']:
        all_path = os.path.join(ALL_DATA, f"{split}.csv")
        aa_path = os.path.join(ALL_DATA_AA, f"{split}.csv")
        aat_path = os.path.join(ALL_DATA_AAT, f"{split}.csv")
        
        if os.path.exists(all_path):
            df_all = pd.read_csv(all_path)
            aa_cnt = len(pd.read_csv(aa_path)) if os.path.exists(aa_path) else 0
            aat_cnt = len(pd.read_csv(aat_path)) if os.path.exists(aat_path) else 0
            print(f"  {split}: 总计 {len(df_all)} 条 (AA: {aa_cnt}, AAT: {aat_cnt})")


if __name__ == '__main__':
    main()