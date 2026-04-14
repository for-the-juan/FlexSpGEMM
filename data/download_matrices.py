#!/usr/bin/env python3
"""
Download matrices from SuiteSparse Matrix Collection based on matrix_split_new.json
Saves train, val, test matrices to separate folders

Supports fuzzy matching for variant names (e.g., case9_A_12 -> case9)
Uses wget with resume capability (-c flag) for robust downloads
Records download time for each matrix and calculates total time statistics
"""

import json
import os
import re
import glob
import shutil
import time
from datetime import timedelta
from ssgetpy import search

def load_matrix_names(json_file):
    """Load matrix names from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['train'], data['val'], data['test']

def try_fuzzy_match(matrix_name):
    """
    Try fuzzy matching when exact match fails
    Returns (matrix_object, matched_name) or (None, None)
    """
    # Common variant suffix patterns
    suffixes_to_try = [
        # _A_number, _B_number variants
        r'_[A-Z]_\d+$',
        # _E, _M single letter suffix
        r'_[A-Z]$',
        # _number suffix
        r'_\d+$',
    ]
    
    current_name = matrix_name
    
    for pattern in suffixes_to_try:
        if re.search(pattern, current_name):
            # Remove matching suffix
            new_name = re.sub(pattern, '', current_name)
            if new_name and new_name != current_name:
                # Try searching with new name
                results = search(name=new_name, limit=5)
                if results:
                    for m in results:
                        if m.name == new_name:
                            return m, new_name
                    # Return first result if no exact match
                    return results[0], new_name
                current_name = new_name
    
    return None, None

def search_matrix_by_name(matrix_name):
    """Search matrix info by name"""
    try:
        # Try exact match first
        results = search(name=matrix_name, limit=10)
        if results:
            for m in results:
                if m.name == matrix_name:
                    return m, matrix_name, 'exact'
            # Return first result if no exact match
            return results[0], results[0].name, 'partial'
        
        # Try fuzzy match if exact match fails
        mat, matched_name = try_fuzzy_match(matrix_name)
        if mat:
            return mat, matched_name, 'fuzzy'
        
        return None, None, None
    except Exception as e:
        print(f"    Error searching {matrix_name}: {e}")
        return None, None, None

def download_matrix(mat, output_dir, original_name):
    """
    Download and extract matrix to specified directory using wget
    Returns: (success: True/False, download_time: float in seconds)
    """
    matrix_name = mat.name
    matrix_group = mat.group
    
    # Check if target file already exists
    target_file = os.path.join(output_dir, f"{original_name}.mtx")
    if os.path.exists(target_file):
        print(f"    Already exists: {original_name}.mtx")
        return True, 0.0
    
    # Create temp directory
    temp_dir = os.path.join(output_dir, f"_temp_{matrix_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Build download URL
    matrix_url = f"https://suitesparse-collection-website.herokuapp.com/MM/{matrix_group}/{matrix_name}.tar.gz"
    temp_tar = os.path.join(temp_dir, f"{matrix_name}.tar.gz")
    
    # Start timing
    start_time = time.time()
    
    # Download with wget -c for resume capability
    max_retries = 3
    success = False
    for retry in range(max_retries):
        # Use wget -c for resume
        ret = os.system(f"wget -c -O {temp_tar} {matrix_url}")
        if ret == 0 and os.path.exists(temp_tar) and os.path.getsize(temp_tar) > 0:
            success = True
            break
        print(f"    Download failed (attempt {retry + 1}/{max_retries}), retrying...")
    
    if not success:
        download_time = time.time() - start_time
        print(f"    Download failed: {matrix_name}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False, download_time
    
    # Extract tar.gz
    ret = os.system(f"tar -zxf {temp_tar} -C {temp_dir}")
    if ret != 0:
        download_time = time.time() - start_time
        print(f"    Extract failed: {matrix_name}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False, download_time
    
    # Remove tar.gz file
    os.remove(temp_tar)
    
    # Find .mtx files
    mtx_files = glob.glob(os.path.join(temp_dir, "**/*.mtx"), recursive=True)
    
    if not mtx_files:
        download_time = time.time() - start_time
        print(f"    No .mtx file found")
        shutil.rmtree(temp_dir)
        return False, download_time
    
    # If only one file, move it directly
    if len(mtx_files) == 1:
        src_file = mtx_files[0]
        shutil.move(src_file, target_file)
        shutil.rmtree(temp_dir)
        download_time = time.time() - start_time
        print(f"    Done: {original_name}.mtx ({download_time:.2f}s)")
        return True, download_time
    
    # If multiple files, find the one matching original name
    found = False
    for mtx_file in mtx_files:
        mtx_basename = os.path.basename(mtx_file).replace('.mtx', '')
        if mtx_basename == original_name:
            shutil.move(mtx_file, target_file)
            found = True
            break
    
    if found:
        shutil.rmtree(temp_dir)
        download_time = time.time() - start_time
        print(f"    Done: {original_name}.mtx (selected from multiple files, {download_time:.2f}s)")
        return True, download_time
    
    # If no match found, use first file and rename
    src_file = mtx_files[0]
    shutil.move(src_file, target_file)
    shutil.rmtree(temp_dir)
    download_time = time.time() - start_time
    print(f"    Done: {original_name}.mtx (renamed from {os.path.basename(src_file)}, {download_time:.2f}s)")
    return True, download_time

def format_time(seconds):
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"

def process_split(matrix_names, split_name, output_base_dir):
    """Process one dataset split (train/val/test)"""
    print(f"\nProcessing {split_name} set ({len(matrix_names)} matrices)...")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, split_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_matrices = []
    total_download_time = 0.0
    download_times = []  # List of (matrix_name, time) for successful downloads
    
    split_start_time = time.time()
    
    for i, name in enumerate(matrix_names):
        print(f"  [{i+1}/{len(matrix_names)}] {name}", end="")
        
        # Check if file already exists
        target_file = os.path.join(output_dir, f"{name}.mtx")
        if os.path.exists(target_file):
            print(f" -> Already exists, skipping")
            skip_count += 1
            continue
        
        # Search matrix
        mat, matched_name, match_type = search_matrix_by_name(name)
        
        if mat is None:
            print(f" -> Not found")
            fail_count += 1
            failed_matrices.append(name)
            continue
        
        if match_type == 'fuzzy':
            print(f" -> Fuzzy match: {name} -> {matched_name}", end="")
        elif match_type == 'partial':
            print(f" -> Partial match: {name} -> {mat.name}", end="")
        
        # Download matrix
        success, download_time = download_matrix(mat, output_dir, name)
        if success:
            success_count += 1
            total_download_time += download_time
            download_times.append((name, download_time))
        else:
            fail_count += 1
            failed_matrices.append(name)
    
    split_elapsed_time = time.time() - split_start_time
    
    print(f"\n{split_name} processing complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total download time: {format_time(total_download_time)}")
    print(f"  Total elapsed time: {format_time(split_elapsed_time)}")
    if failed_matrices:
        print(f"  Failed list: {failed_matrices}")
    
    return {
        'success': success_count,
        'skip': skip_count,
        'fail': fail_count,
        'failed_matrices': failed_matrices,
        'total_download_time': total_download_time,
        'elapsed_time': split_elapsed_time,
        'download_times': download_times
    }

def main():
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'matrix_split_new.json')
    output_base_dir = script_dir  # Output to subdirectories under data/
    
    # Overall timing
    overall_start_time = time.time()
    
    # Load matrix names
    print(f"Loading matrix names from {json_file}...")
    train_names, val_names, test_names = load_matrix_names(json_file)
    print(f"Loaded: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")
    
    # Process each dataset
    results = {}
    
    results['train'] = process_split(train_names, 'train', output_base_dir)
    results['val'] = process_split(val_names, 'val', output_base_dir)
    results['test'] = process_split(test_names, 'test', output_base_dir)
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # Calculate totals
    total_success = sum(r['success'] for r in results.values())
    total_skip = sum(r['skip'] for r in results.values())
    total_fail = sum(r['fail'] for r in results.values())
    total_download_time = sum(r['total_download_time'] for r in results.values())
    all_failed = []
    for r in results.values():
        all_failed.extend(r['failed_matrices'])
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"\nDataset Statistics:")
    print(f"  Train: {results['train']['success']} success, {results['train']['skip']} skipped, {results['train']['fail']} failed")
    print(f"  Val:   {results['val']['success']} success, {results['val']['skip']} skipped, {results['val']['fail']} failed")
    print(f"  Test:  {results['test']['success']} success, {results['test']['skip']} skipped, {results['test']['fail']} failed")
    
    print(f"\nTime Statistics:")
    print(f"  Train download time: {format_time(results['train']['total_download_time'])}")
    print(f"  Val download time:   {format_time(results['val']['total_download_time'])}")
    print(f"  Test download time:  {format_time(results['test']['total_download_time'])}")
    print(f"  Total download time:  {format_time(total_download_time)}")
    print(f"  Total elapsed time:   {format_time(overall_elapsed_time)}")
    
    print(f"\nOverall Results:")
    print(f"  Total matrices: {len(train_names) + len(val_names) + len(test_names)}")
    print(f"  Success: {total_success}")
    print(f"  Skipped: {total_skip}")
    print(f"  Failed: {total_fail}")
    
    if all_failed:
        print(f"\nFailed matrices ({len(all_failed)}):")
        for name in all_failed:
            print(f"  - {name}")

if __name__ == '__main__':
    main()