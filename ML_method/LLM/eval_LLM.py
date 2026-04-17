#!/usr/bin/env python3

import argparse
import csv
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VALID_M = {8, 16, 32}
VALID_N = {8, 16, 32}
VALID_T = set(range(9))
MAX_NEW_TOKENS = 32

TARGET_MATRICES = {
    "hangGlider_4",
    "webbase-1M",
    "pkustk12",
    "Goodwin_095",
    "af_shell10",
    "s3rmq4m1",
    "rma10",
    "nemeth12",
    "TSOPF_FS_b300_c2",
    "trans5",
    "heart3",
    "gupta3",
}

SOURCE_TAG_TO_CSV_ARG = {
    "A100_AAT": "a100_aat_csv",
    "A100_AA": "a100_aa_csv",
    "H200_AAT": "h200_aat_csv",
    "H200_AA": "h200_aa_csv",
}


def parse_cfg(text: str) -> tuple[int | None, int | None, int | None]:
    numbers = re.findall(r"\b(\d+)\b", text.strip())
    if len(numbers) < 3:
        return (None, None, None)
    try:
        m, n, t = int(numbers[0]), int(numbers[1]), int(numbers[2])
    except ValueError:
        return (None, None, None)
    if m in VALID_M and n in VALID_N and t in VALID_T:
        return (m, n, t)
    return (None, None, None)


def cfg_col_name(m: int, n: int, t: int) -> str:
    return f"{m}x{n}_{t}/8"


def build_prompt(instruction: str, input_text: str) -> str:
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ).format(instruction=instruction, input=input_text)


def load_gflops_table(csv_path: str) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_name = (row.get("matrix_name") or "").strip()
            if not matrix_name:
                continue
            cfg_map: dict[str, float] = {}
            for k, v in row.items():
                if not k or k == "matrix_name":
                    continue
                if v is None:
                    continue
                vv = v.strip()
                if not vv or vv.upper() == "N/A":
                    continue
                try:
                    cfg_map[k] = float(vv)
                except ValueError:
                    continue
            table[matrix_name] = cfg_map
    return table


def fmt_cfg(m: int | None, n: int | None, t: int | None) -> str:
    if m is None or n is None or t is None:
        return "-"
    return f"{m} {n} {t}"


def fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def parse_gflops_from_log(log_path: str) -> float | None:
    """Parse GFlops throughput from log file."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Throughput" in line and "GFlops" in line:
                    # Extract number before "GFlops"
                    match = re.search(r"(\d+\.?\d*)\s*GFlops", line)
                    if match:
                        return float(match.group(1))
    except FileNotFoundError:
        pass
    return None


def load_best_gflops_table(csv_path: str) -> dict[str, float]:
    """Load best_gflops from test12_result.csv, keyed by matrix_name (AA mode only)."""
    table: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_name = (row.get("matrix_name") or "").strip()
            mode = (row.get("mode") or "").strip()
            best_gflops_str = (row.get("best_gflops") or "").strip()
            if not matrix_name or mode != "AA":
                continue
            if best_gflops_str and best_gflops_str.upper() != "N/A":
                try:
                    table[matrix_name] = float(best_gflops_str)
                except ValueError:
                    continue
    return table


def main():
    parser = argparse.ArgumentParser(description="Evaluate test set and print markdown table")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint",
        help="model_checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./spgemm_finetune_test_best.json",
        help="test_json",
    )
    parser.add_argument(
        "--a100_aat_csv",
        type=str,
        default="../../data/data_prepare/prime_data/a100_aat_gflops_all.csv",
    )
    parser.add_argument(
        "--a100_aa_csv",
        type=str,
        default="../../data/data_prepare/prime_data/a100_gflops_all.csv",
    )
    parser.add_argument(
        "--h200_aat_csv",
        type=str,
        default="../../data/data_prepare/prime_data/h200_aat_gflops_all.csv",
    )
    parser.add_argument(
        "--h200_aa_csv",
        type=str,
        default="../../data/data_prepare/prime_data/h200_gflops_all.csv",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_csv",
        type=str,
        default="./LLM_12_results.csv",
        help="Optional: write CSV results to this file path",
    )
    parser.add_argument(
        "--test12_result_csv",
        type=str,
        default="../LightGBM/predictResult_test12/test12_result.csv",
        help="Path to test12_result.csv for best_gflops lookup",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../LightGBM/predictResult_test12/log",
        help="Directory containing FlexSpGEMM log files",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    gflops_tables = {
        "A100_AAT": load_gflops_table(args.a100_aat_csv),
        "A100_AA": load_gflops_table(args.a100_aa_csv),
        "H200_AAT": load_gflops_table(args.h200_aat_csv),
        "H200_AA": load_gflops_table(args.h200_aa_csv),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    rows: list[dict] = []
    valid_pcts: list[float] = []
    total_infer_sec = 0.0
    infer_count = 0

    for idx, item in enumerate(test_data):
        source_tag = item.get("source_tag", "")
        matrix_name = item.get("matrix_name", "")
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        gold_output = item.get("output", "")

        om, on, ot = parse_cfg(gold_output)
        if source_tag not in gflops_tables:
            rows.append(
                {
                    "index": idx,
                    "matrix_name": matrix_name,
                    "source_tag": source_tag,
                    "pred_cfg": "-",
                    "pred_gflops": None,
                    "optimal_cfg": fmt_cfg(om, on, ot),
                    "optimal_gflops": None,
                    "pct": None,
                    "note": "unknown_source_tag",
                }
            )
            continue

        prompt_text = build_prompt(instruction, input_text)
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        sync_cuda = model.device.type == "cuda"
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if sync_cuda:
            torch.cuda.synchronize()
        total_infer_sec += time.perf_counter() - t0
        infer_count += 1

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pm, pn, pt = parse_cfg(pred_text)

        note = ""
        pred_gf: float | None = None
        optimal_gf: float | None = None
        pct: float | None = None

        table = gflops_tables[source_tag].get(matrix_name)
        if table is None:
            note = "matrix_not_in_csv"
        else:
            if om is None:
                note = "invalid_gold_output"
            else:
                opt_col = cfg_col_name(om, on, ot)
                optimal_gf = table.get(opt_col)
                if optimal_gf is None:
                    note = "optimal_cfg_not_in_csv"
                elif optimal_gf <= 0:
                    note = "optimal_gflops_non_positive"
                else:
                    if pm is None:
                        note = "parse_failed"
                    else:
                        pred_col = cfg_col_name(pm, pn, pt)
                        pred_gf = table.get(pred_col)
                        if pred_gf is None:
                            note = "pred_cfg_not_in_csv"
                        else:
                            pct = 100.0 * pred_gf / optimal_gf
                            valid_pcts.append(pct)

        rows.append(
            {
                "index": idx,
                "matrix_name": matrix_name,
                "source_tag": source_tag,
                "pred_cfg": fmt_cfg(pm, pn, pt),
                "pred_gflops": pred_gf,
                "optimal_cfg": fmt_cfg(om, on, ot),
                "optimal_gflops": optimal_gf,
                "pct": pct,
                "note": note,
            }
        )

    avg_pct = sum(valid_pcts) / len(valid_pcts) if valid_pcts else None
    avg_infer_sec = (total_infer_sec / infer_count) if infer_count else None

    # Filter: only keep data where source_tag is A100_AA and matrix_name is in whitelist
    filtered_rows = [
        r for r in rows 
        if r["source_tag"] == "A100_AA" and r["matrix_name"] in TARGET_MATRICES
    ]

    # Load best_gflops from test12_result.csv
    best_gflops_table = load_best_gflops_table(args.test12_result_csv)

    # Enrich filtered rows with pred_gflops from log and best_gflops from csv
    output_rows = []
    for r in filtered_rows:
        matrix_name = r["matrix_name"]
        pred_cfg = r["pred_cfg"]
        
        # Parse pred_cfg to construct log filename
        pm, pn, pt = parse_cfg(pred_cfg)
        pred_gflops = None
        best_gflops = None
        gflops_ratio = None
        
        if pm is not None and pn is not None and pt is not None:
            # Construct log filename: {matrix_name}_AA_{m}x{n}_tc{t}.log
            log_filename = f"{matrix_name}_AA_{pm}x{pn}_tc{pt}.log"
            log_path = f"{args.log_dir}/{log_filename}"
            pred_gflops = parse_gflops_from_log(log_path)
        
        # Get best_gflops from table
        best_gflops = best_gflops_table.get(matrix_name)
        
        # Calculate ratio
        if pred_gflops is not None and best_gflops is not None and best_gflops > 0:
            gflops_ratio = pred_gflops / best_gflops
        
        output_rows.append({
            "matrix_name": matrix_name,
            "pred_cfg": pred_cfg,
            "pred_gflops": pred_gflops,
            "best_gflops": best_gflops,
            "gflops_ratio": gflops_ratio,
        })

    # Output to console
    print(f"Filtered {len(output_rows)} rows (source_tag=A100_AA, matrix_name in target list)")
    for r in output_rows:
        print(f"{r['matrix_name']},{r['pred_cfg']},{r['pred_gflops']},{r['best_gflops']},{r['gflops_ratio']}")

    # Save as CSV file
    if args.save_csv:
        with open(args.save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["matrix_name", "pred_cfg", "pred_gflops", "best_gflops", "gflops_ratio"])
            for r in output_rows:
                writer.writerow([
                    r["matrix_name"], 
                    r["pred_cfg"],
                    fmt_float(r["pred_gflops"], 2) if r["pred_gflops"] is not None else "-",
                    fmt_float(r["best_gflops"], 2) if r["best_gflops"] is not None else "-",
                    fmt_float(r["gflops_ratio"], 4) if r["gflops_ratio"] is not None else "-",
                ])
        print(f"\nResults saved to {args.save_csv}")


if __name__ == "__main__":
    main()
