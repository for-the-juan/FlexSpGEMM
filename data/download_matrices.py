#!/usr/bin/env python3
"""
Download matrices from SuiteSparse Matrix Collection based on matrix_split_new.json
Saves train, val, test matrices to separate folders

Supports fuzzy matching for variant names (e.g., case9_A_12 -> case9)
Uses wget with resume capability (-c flag) for robust downloads
"""

import json
import os
import re
import glob
import shutil
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
    Returns: True/False for success
    """
    matrix_name = mat.name
    matrix_group = mat.group
    
    # Check if target file already exists
    target_file = os.path.join(output_dir, f"{original_name}.mtx")
    if os.path.exists(target_file):
        print(f"    Already exists: {original_name}.mtx")
        return True
    
    # Create temp directory
    temp_dir = os.path.join(output_dir, f"_temp_{matrix_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Build download URL
    matrix_url = f"https://suitesparse-collection-website.herokuapp.com/MM/{matrix_group}/{matrix_name}.tar.gz"
    temp_tar = os.path.join(temp_dir, f"{matrix_name}.tar.gz")
    
    # Download with wget -c for resume capability
    max_retries = 3
    success = False
    for retry in range(max_retries):
        # Use wget -c for resume, -q for quiet
        ret = os.system(f"wget -c -q -O {temp_tar} {matrix_url}")
        if ret == 0 and os.path.exists(temp_tar) and os.path.getsize(temp_tar) > 0:
            success = True
            break
        print(f"    Download failed (attempt {retry + 1}/{max_retries}), retrying...")
    
    if not success:
        print(f"    Download failed: {matrix_name}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False
    
    # Extract tar.gz
    ret = os.system(f"tar -zxf {temp_tar} -C {temp_dir}")
    if ret != 0:
        print(f"    Extract failed: {matrix_name}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False
    
    # Remove tar.gz file
    os.remove(temp_tar)
    
    # Find .mtx files
    mtx_files = glob.glob(os.path.join(temp_dir, "**/*.mtx"), recursive=True)
    
    if not mtx_files:
        print(f"    No .mtx file found")
        shutil.rmtree(temp_dir)
        return False
    
    # If only one file, move it directly
    if len(mtx_files) == 1:
        src_file = mtx_files[0]
        shutil.move(src_file, target_file)
        shutil.rmtree(temp_dir)
        print(f"    Done: {original_name}.mtx")
        return True
    
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
        print(f"    Done: {original_name}.mtx (selected from multiple files)")
        return True
    
    # If no match found, use first file and rename
    src_file = mtx_files[0]
    shutil.move(src_file, target_file)
    shutil.rmtree(temp_dir)
    print(f"    Done: {original_name}.mtx (renamed from {os.path.basename(src_file)})")
    return True

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
        if download_matrix(mat, output_dir, name):
            success_count += 1
        else:
            fail_count += 1
            failed_matrices.append(name)
    
    print(f"\n{split_name} processing complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    if failed_matrices:
        print(f"  Failed list: {failed_matrices}")
    
    return success_count, skip_count, fail_count, failed_matrices

def main():
    # Path configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'matrix_split_new.json')
    output_base_dir = script_dir  # Output to subdirectories under data/
    
    # Load matrix names
    print(f"Loading matrix names from {json_file}...")
    train_names, val_names, test_names = load_matrix_names(json_file)
    print(f"Loaded: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")
    
    # Process each dataset
    total_success = 0
    total_skip = 0
    total_fail = 0
    all_failed = []
    
    success, skip, fail, failed = process_split(train_names, 'train', output_base_dir)
    total_success += success
    total_skip += skip
    total_fail += fail
    all_failed.extend(failed)
    
    success, skip, fail, failed = process_split(val_names, 'val', output_base_dir)
    total_success += success
    total_skip += skip
    total_fail += fail
    all_failed.extend(failed)
    
    success, skip, fail, failed = process_split(test_names, 'test', output_base_dir)
    total_success += success
    total_skip += skip
    total_fail += fail
    all_failed.extend(failed)
    
    print("\n" + "="*50)
    print("All processing complete!")
    print(f"Total: Success={total_success}, Skipped={total_skip}, Failed={total_fail}")
    if all_failed:
        print(f"Failed matrices ({len(all_failed)}):")
        for name in all_failed:
            print(f"  - {name}")

if __name__ == '__main__':
    main()