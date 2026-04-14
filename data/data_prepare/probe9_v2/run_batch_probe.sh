#!/bin/bash
# Convenient wrapper script to run batch probe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================================================"
echo "  Batch Tile Probe for Training Dataset"
echo "========================================================================"
echo ""
echo "This will process ALL matrices in /home/stu1/Dataset/training_dataset"
echo "and generate separate result files for each subfolder."
echo ""
echo "Output location: $SCRIPT_DIR/results/"
echo ""
echo "Estimated time: ~2-6 hours (depending on dataset size)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting batch probe..."
echo "Logs will be saved to: $SCRIPT_DIR/batch_probe.log"
echo ""

cd "$SCRIPT_DIR"
python3 batch_probe_all.py 2>&1 | tee batch_probe.log

echo ""
echo "========================================================================"
echo "  Batch probe completed!"
echo "  Results location: $SCRIPT_DIR/results/"
echo "  Log file: $SCRIPT_DIR/batch_probe.log"
echo "========================================================================"
