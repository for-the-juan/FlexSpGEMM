#!/bin/bash
# compare.sh - Run probeC and AdaSpGEMM, compare results
# Usage: bash compare.sh

PROBEC="/home/stu1/marui/ada/probeC/probeC"
ADA_BIN="/home/stu1/marui/ada/AdaSpGEMM/bin"
DATASET="/home/stu1/marui/ada/TileSpGEMMDataset"
OUTPUT="/home/stu1/marui/ada/test/probeC_test.txt"

TILE_M=(8 8 8 16 16 16 32 32 32)
TILE_N=(8 16 32 8 16 32 8 16 32)

# Header
echo "================================================================" > "$OUTPUT"
echo "  probeC vs AdaSpGEMM Comparison Report" >> "$OUTPUT"
echo "  Generated: $(date)" >> "$OUTPUT"
echo "================================================================" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# CSV summary header
CSV_FILE="/home/stu1/marui/ada/test/probeC_csv.txt"
echo "matrix,tile_m,tile_n,pred_total,pred_tiny,pred_sml,pred_lrg,pred_dns,pred_ful,actual_total,actual_tiny,actual_sml,actual_lrg,actual_dns,actual_ful" > "$CSV_FILE"

for mtx_file in "$DATASET"/*.mtx; do
    base_name=$(basename "$mtx_file" .mtx)
    echo "Processing: $base_name ..."

    echo "================================================================" >> "$OUTPUT"
    echo "Matrix: $base_name" >> "$OUTPUT"
    echo "================================================================" >> "$OUTPUT"

    # Run probeC
    echo "  Running probeC..."
    probe_output=$($PROBEC "$mtx_file" 2>&1)
    echo "$probe_output" | head -5 >> "$OUTPUT"
    echo "" >> "$OUTPUT"

    # Extract probeC CSV data (skip header)
    declare -A pred_tiny pred_sml pred_lrg pred_dns pred_ful pred_total
    while IFS=',' read -r tm tn total tiny sml lrg dns ful; do
        key="${tm}x${tn}"
        pred_total[$key]=$total
        pred_tiny[$key]=$tiny
        pred_sml[$key]=$sml
        pred_lrg[$key]=$lrg
        pred_dns[$key]=$dns
        pred_ful[$key]=$ful
    done < <(echo "$probe_output" | sed -n '/^[0-9]/p' | tail -9)

    # Run each AdaSpGEMM tile size
    printf "  %-10s %8s %8s %8s %8s %8s | %8s %8s %8s %8s %8s\n" \
        "TileSize" "P_Tiny" "P_Sml" "P_Lrg" "P_Dns" "P_Ful" \
        "A_Tiny" "A_Sml" "A_Lrg" "A_Dns" "A_Ful" >> "$OUTPUT"
    printf "  %-10s %8s %8s %8s %8s %8s | %8s %8s %8s %8s %8s\n" \
        "----------" "--------" "--------" "--------" "--------" "--------" \
        "--------" "--------" "--------" "--------" "--------" >> "$OUTPUT"

    for ((i=0; i<${#TILE_M[@]}; i++)); do
        tm=${TILE_M[i]}
        tn=${TILE_N[i]}
        key="${tm}x${tn}"
        bin="$ADA_BIN/test_m${tm}_n${tn}"

        if [ ! -f "$bin" ]; then
            echo "  Skipping ${tm}x${tn} (binary not found)"
            continue
        fi

        echo "  Running AdaSpGEMM ${tm}x${tn}..."
        # Run with retry logic and delay between GPU runs
        ada_output=""
        for attempt in 1 2 3; do
            ada_output=$($bin -d 0 -aat 0 "$mtx_file" 2>&1)
            ret=$?
            if [ $ret -eq 0 ] && echo "$ada_output" | grep -q "^Number:"; then
                break
            fi
            echo "    Attempt $attempt failed (exit=$ret), retrying after delay..."
            sleep 3
        done
        sleep 1

        # Extract "Number: Tiny: X, Sml: Y, Lrg: Z, Dns: W, Ful: V"
        number_line=$(echo "$ada_output" | grep "^Number:")
        if [ -z "$number_line" ]; then
            echo "    WARNING: No Number line found for ${tm}x${tn} (exit=$ret)" >> "$OUTPUT"
            continue
        fi

        a_tiny=$(echo "$number_line" | sed 's/.*Tiny: \([0-9]*\).*/\1/')
        a_sml=$(echo "$number_line" | sed 's/.*Sml: \([0-9]*\).*/\1/')
        a_lrg=$(echo "$number_line" | sed 's/.*Lrg: \([0-9]*\).*/\1/')
        a_dns=$(echo "$number_line" | sed 's/.*Dns: \([0-9]*\).*/\1/')
        a_ful=$(echo "$number_line" | sed 's/.*Ful: \([0-9]*\).*/\1/')

        # Extract Non-empty tiles of C
        a_total_line=$(echo "$ada_output" | grep "Non-empty tiles of C")
        a_total=$(echo "$a_total_line" | sed 's/.*= \([0-9]*\).*/\1/')

        p_tiny=${pred_tiny[$key]:-0}
        p_sml=${pred_sml[$key]:-0}
        p_lrg=${pred_lrg[$key]:-0}
        p_dns=${pred_dns[$key]:-0}
        p_ful=${pred_ful[$key]:-0}
        p_total=${pred_total[$key]:-0}

        printf "  %-10s %8s %8s %8s %8s %8s | %8s %8s %8s %8s %8s\n" \
            "${tm}x${tn}" "$p_tiny" "$p_sml" "$p_lrg" "$p_dns" "$p_ful" \
            "$a_tiny" "$a_sml" "$a_lrg" "$a_dns" "$a_ful" >> "$OUTPUT"

        # CSV line
        echo "$base_name,$tm,$tn,$p_total,$p_tiny,$p_sml,$p_lrg,$p_dns,$p_ful,$a_total,$a_tiny,$a_sml,$a_lrg,$a_dns,$a_ful" >> "$CSV_FILE"
    done

    echo "" >> "$OUTPUT"
    echo "${base_name} Done!"
done

# Add accuracy summary
echo "" >> "$OUTPUT"
echo "================================================================" >> "$OUTPUT"
echo "  Accuracy Summary" >> "$OUTPUT"
echo "================================================================" >> "$OUTPUT"

# Calculate accuracy from CSV
python3 - "$CSV_FILE" "$OUTPUT" << 'PYEOF'
import sys, csv

csv_file = sys.argv[1]
out_file = sys.argv[2]

results = []
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

if not results:
    with open(out_file, 'a') as f:
        f.write("No results to summarize.\n")
    sys.exit(0)

with open(out_file, 'a') as f:
    f.write("\n")

    # Per-matrix accuracy
    total_err = 0
    total_cnt = 0
    category_errors = {'tiny': 0, 'sml': 0, 'lrg': 0, 'dns': 0, 'ful': 0}
    category_counts = {'tiny': 0, 'sml': 0, 'lrg': 0, 'dns': 0, 'ful': 0}
    total_tile_err = 0
    total_tile_cnt = 0

    for row in results:
        for cat in ['tiny', 'sml', 'lrg', 'dns', 'ful']:
            pred = int(row.get(f'pred_{cat}', 0))
            actual = int(row.get(f'actual_{cat}', 0))
            err = abs(pred - actual)
            denom = max(actual, 1)
            category_errors[cat] += err
            category_counts[cat] += actual

        pred_total = int(row.get('pred_total', 0))
        actual_total = int(row.get('actual_total', 0))
        if actual_total > 0:
            total_tile_err += abs(pred_total - actual_total)
            total_tile_cnt += actual_total

    f.write("Category-level accuracy (across all matrices and tile sizes):\n\n")
    f.write(f"{'Category':<10} {'Total Actual':>12} {'Total |Error|':>14} {'Error Rate':>12}\n")
    f.write(f"{'-'*10} {'-'*12} {'-'*14} {'-'*12}\n")

    for cat in ['tiny', 'sml', 'lrg', 'dns', 'ful']:
        actual = category_counts[cat]
        err = category_errors[cat]
        rate = (err / actual * 100) if actual > 0 else 0
        f.write(f"{cat:<10} {actual:>12} {err:>14} {rate:>11.2f}%\n")

    f.write(f"\n")
    if total_tile_cnt > 0:
        f.write(f"Total C tiles prediction error: {total_tile_err}/{total_tile_cnt} = {total_tile_err/total_tile_cnt*100:.2f}%\n")

    f.write("\n")
PYEOF

echo ""
echo "Results written to: $OUTPUT"
echo "CSV data written to: $CSV_FILE"
