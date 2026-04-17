#!/bin/bash

# Record start time
START_TIME=$(date +%s)

# Relative Path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Project Root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="$PROJECT_ROOT/data/test/"
DATASET_T_DIR="$PROJECT_ROOT/data/mtx_T"
HSMU_LOG_DIR="$SCRIPT_DIR/logs/HSMU/"
spECK_LOG_DIR="$SCRIPT_DIR/logs/spECK/"
TileSpGEMM_LOG_DIR="$SCRIPT_DIR/logs/TileSpGEMM/"
mkdir -p "$HSMU_LOG_DIR"
mkdir -p "$spECK_LOG_DIR"
mkdir -p "$TileSpGEMM_LOG_DIR"
TIMEOUT_SECONDS=150

cd HSMU-SpGEMM_AA/evaluation/script
rm -rf ./test
make

echo "Running HSMU-SpGEMM_AA..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./test "$mtx_file" > "$HSMU_LOG_DIR/AA/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

cd "$SCRIPT_DIR/HSMU-SpGEMM_AAT/evaluation/script"
rm -rf ./test
make

echo "Running HSMU-SpGEMM_AAT..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./test "$mtx_file" > "$HSMU_LOG_DIR/AAT/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

cd "$SCRIPT_DIR/spECK"
rm -rf ./speck
make speck

echo "Running spECK_AA..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./speck "$mtx_file" > "$spECK_LOG_DIR/AA/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

# AAT
echo "Running spECK_AAT..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file_a; do
    base_name_=$(basename "$mtx_file_a" .mtx)
    
    mtx_file_b=$(find "$DATASET_T_DIR" -name "${base_name_}.mtx" -type f | head -n 1)

    if [ -z "$mtx_file_b" ] || [ ! -f "$mtx_file_b" ]; then
        echo "Warn: Matrix AT ${base_name_}.mtx not exist"
        continue
    fi

    echo "RUN: ${base_name_} (A * A^T)"
    timeout $TIMEOUT_SECONDS ./speck "$mtx_file_a" "$mtx_file_b" > "$spECK_LOG_DIR/AAT/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

cd "$SCRIPT_DIR/TileSpGEMM/src"
make

echo "Running TileSpGEMM_AA..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./test -d 0 -aat 0 "$mtx_file" > "$TileSpGEMM_LOG_DIR/AA/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

echo "Running TileSpGEMM_AAT..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./test -d 0 -aat 1 "$mtx_file" > "$TileSpGEMM_LOG_DIR/AAT/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

# Calculate and print total elapsed time
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
ELAPSED_REMAINING_SECONDS=$((ELAPSED_SECONDS % 60))

echo "All other SpGEMM methods finished!"
echo "You can find all logs in ./logs!"
echo "Total elapsed time: ${ELAPSED_MINUTES} minutes ${ELAPSED_REMAINING_SECONDS} seconds (${ELAPSED_SECONDS} seconds in total)"
