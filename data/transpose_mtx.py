#!/usr/bin/env python3
"""
Matrix Transpose Script
Transpose all mtx files from source directory and its subfolders to target directory, preserving subfolder structure
"""

import os
from scipy.io import mmread, mmwrite
from scipy.sparse import issparse
import numpy as np

def transpose_mtx_files(src_dir, dst_dir):
    """
    Recursively transpose all mtx files from source directory and its subfolders to target directory, preserving subfolder structure
    
    Args:
        src_dir: Source directory path
        dst_dir: Target directory path
    """
    # Create target directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Recursively find all mtx files
    mtx_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.mtx'):
                mtx_files.append(os.path.join(root, file))
    
    if not mtx_files:
        print(f"No mtx files found in {src_dir} and its subfolders")
        return
    
    print(f"Found {len(mtx_files)} mtx files to process")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for mtx_file in sorted(mtx_files):
        # Calculate relative path
        rel_path = os.path.relpath(mtx_file, src_dir)
        dst_file = os.path.join(dst_dir, rel_path)
        
        # Create target subfolder
        dst_subdir = os.path.dirname(dst_file)
        os.makedirs(dst_subdir, exist_ok=True)
        
        try:
            print(f"Processing: {rel_path}...", end=" ", flush=True)
            
            # Read matrix
            matrix = mmread(mtx_file)
            
            # Transpose
            matrix_t = matrix.transpose()
            
            # Write to target file
            mmwrite(dst_file, matrix_t, comment='Transposed matrix')
            
            print("Done")
            success_count += 1
            
        except Exception as e:
            print(f"Failed: {e}")
            error_count += 1
    
    print("-" * 60)
    print(f"Processing complete: {success_count} succeeded, {error_count} failed")
    print(f"Transposed files saved in: {dst_dir}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source folders (train, test, val) using relative paths
    source_folders = ['train', 'test', 'val']
    
    # Output directory for transposed matrices
    output_dir = os.path.join(script_dir, 'mtx_T')
    
    # Process each source folder
    for folder in source_folders:
        src_dir = os.path.join(script_dir, folder)
        if os.path.exists(src_dir):
            print(f"\n{'='*60}")
            print(f"Processing folder: {folder}")
            print(f"{'='*60}")
            transpose_mtx_files(src_dir, output_dir)
        else:
            print(f"Folder {folder} does not exist, skipping...")