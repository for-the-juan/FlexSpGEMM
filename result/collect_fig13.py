#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FIG8_CSV = SCRIPT_DIR / 'data' / 'Fig8.csv'
FIG11_CSV = SCRIPT_DIR / 'data' / 'Fig11.csv'
PROBE_CSV = PROJECT_ROOT / 'data' / 'data_prepare' / 'data_get' / 'probe.csv'
TEST100_CSV = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test100' / 'test100_result.csv'
TILE_AA_LOG_DIR = PROJECT_ROOT / 'other_spgemm_methods' / 'logs' / 'TileSpGEMM' / 'AA'

OUT_CSV = SCRIPT_DIR / 'data' / 'Fig13.csv'
OUT_FIG = SCRIPT_DIR / 'plots' / 'Fig13.png'

TILE_CONVERSION_RE = re.compile(
    r'CSR\s+to\s+Tile\s+conversion\s+uses\s*([0-9]*\.?[0-9]+)\s*ms',
    re.IGNORECASE,
)


def to_float_or_zero(value):
    if value is None:
        return 0.0
    text = str(value).strip()
    if text == '':
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def read_fig8_aa_matrix_order(path: Path):
    order = []
    seen = set()
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Type', '').strip().upper() != 'AA':
                continue
            name = row.get('Matrix Name', '').strip()
            if name and name not in seen:
                seen.add(name)
                order.append(name)
    return order


def read_probe_lookup(path: Path):
    lookup = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mode', '').strip().upper() != 'AA':
                continue
            if row.get('gpu', '').strip() != 'A100':
                continue
            name = row.get('matrix_name', '').strip()
            if not name:
                continue
            feature_extraction = (
                to_float_or_zero(row.get('A_probe_ms', ''))
                + to_float_or_zero(row.get('C_probe_total_ms', ''))
            )
            lightgbm_inference = to_float_or_zero(row.get('lightgbm_decision_ms', ''))
            lookup[name] = {
                'Feature Extraction': feature_extraction,
                'LightGBM Inference': lightgbm_inference,
            }
    return lookup


def read_test100_lookup(path: Path):
    lookup = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mode', '').strip().upper() != 'AA':
                continue
            if row.get('gpu', '').strip() != 'A100':
                continue
            name = row.get('matrix_name', '').strip()
            if not name:
                continue
            lookup[name] = {
                'Format Conversion': to_float_or_zero(row.get('csr2tile_ms', '')),
            }
    return lookup


def read_fig11_lookup(path: Path):
    lookup = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Matrix Name', '').strip()
            if not name:
                continue
            runtime = (
                to_float_or_zero(row.get('FlexSpGEMM_Numeric_Stage', ''))
                + to_float_or_zero(row.get('FlexSpGEMM_Symbolic_Stage', ''))
                + to_float_or_zero(row.get('FlexSpGEMM_Malloc', ''))
            )
            lookup[name] = runtime
    return lookup


def parse_tilespgemm_conversion_time(matrix_name: str):
    log_path = TILE_AA_LOG_DIR / f'{matrix_name}.log'
    if not log_path.exists() or log_path.stat().st_size == 0:
        return 0.0

    text = log_path.read_text(encoding='utf-8', errors='ignore')
    match = TILE_CONVERSION_RE.search(text)
    return float(match.group(1)) if match else 0.0


def build_rows(matrix_order, probe_lookup, test100_lookup, fig11_lookup):
    rows = []
    for name in matrix_order:
        feature_extraction = to_float_or_zero(probe_lookup.get(name, {}).get('Feature Extraction', 0.0))
        lightgbm_inference = to_float_or_zero(probe_lookup.get(name, {}).get('LightGBM Inference', 0.0))
        format_conversion = to_float_or_zero(test100_lookup.get(name, {}).get('Format Conversion', 0.0))
        flex_conversion = feature_extraction + lightgbm_inference + format_conversion
        tile_conversion = parse_tilespgemm_conversion_time(name)
        flex_runtime = to_float_or_zero(fig11_lookup.get(name, 0.0))

        rows.append({
            'Matrix Name': name,
            'Feature Extraction': feature_extraction,
            'LightGBM Inference': lightgbm_inference,
            'Format Conversion': format_conversion,
            'FlexSpGEMM Coversion Time': flex_conversion,
            'TileSpGEMM Conversion Time': tile_conversion,
            'FlexSpGEMM Runtime': flex_runtime,
        })
    return rows


def write_fig13_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'Feature Extraction',
            'LightGBM Inference',
            'Format Conversion',
            'FlexSpGEMM Coversion Time',
            'TileSpGEMM Conversion Time',
            'FlexSpGEMM Runtime',
        ])
        for row in rows:
            writer.writerow([
                row['Matrix Name'],
                f"{row['Feature Extraction']:.6f}",
                f"{row['LightGBM Inference']:.6f}",
                f"{row['Format Conversion']:.6f}",
                f"{row['FlexSpGEMM Coversion Time']:.6f}",
                f"{row['TileSpGEMM Conversion Time']:.6f}",
                f"{row['FlexSpGEMM Runtime']:.6f}",
            ])


def draw_fig13(rows, out_fig: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed. Skip plotting.')
        return

    matrices = [row['Matrix Name'] for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 1, figsize=(22, 12), constrained_layout=True)

    # (a) Conversion/runtime scatter plot with log10 scale.
    eps = 1e-2
    x_flex_conv = [xi - 0.18 for xi in x]
    x_tile_conv = x
    x_runtime = [xi + 0.18 for xi in x]

    y_flex_conv = [max(row['FlexSpGEMM Coversion Time'], eps) for row in rows]
    y_tile_conv = [max(row['TileSpGEMM Conversion Time'], eps) for row in rows]
    y_runtime = [max(row['FlexSpGEMM Runtime'], eps) for row in rows]

    axes[0].scatter(x_flex_conv, y_flex_conv, label='FlexSpGEMM Coversion Time', marker='o', s=52, color='#4C78A8')
    axes[0].scatter(x_tile_conv, y_tile_conv, label='TileSpGEMM Conversion Time', marker='s', s=52, color='#F58518')
    axes[0].scatter(x_runtime, y_runtime, label='FlexSpGEMM Runtime', marker='^', s=62, color='#54A24B')
    axes[0].set_yscale('log', base=10)
    axes[0].set_ylim(eps, 10000)
    axes[0].set_ylabel('Time (ms, log10)')
    axes[0].set_xlabel('Matrices')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(matrices, rotation=45, ha='right')
    axes[0].grid(axis='y', which='both', alpha=0.25)
    axes[0].legend(loc='upper left', ncol=3)
    axes[0].set_title('(a) Conversion Time and Runtime')

    # (b) Normalized conversion-time breakdown.
    feature_pct = []
    inference_pct = []
    format_pct = []
    for row in rows:
        total = row['FlexSpGEMM Coversion Time']
        if total <= 0:
            feature_pct.append(0.0)
            inference_pct.append(0.0)
            format_pct.append(0.0)
            continue
        feature_pct.append(100.0 * row['Feature Extraction'] / total)
        inference_pct.append(100.0 * row['LightGBM Inference'] / total)
        format_pct.append(100.0 * row['Format Conversion'] / total)

    axes[1].bar(x, feature_pct, width=0.95, label='Feature Extraction', color='#4C78A8')
    axes[1].bar(x, inference_pct, width=0.95, bottom=feature_pct, label='LightGBM Inference', color='#F58518')
    bottom_format = [a + b for a, b in zip(feature_pct, inference_pct)]
    axes[1].bar(x, format_pct, width=0.95, bottom=bottom_format, label='Format Conversion', color='#54A24B')
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Conversion Time Breakdown (%)')
    axes[1].set_xlabel('Matrices')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(matrices, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(loc='upper right', ncol=3)
    axes[1].set_title('(b) Normalized Conversion Time Breakdown')

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def main():
    matrix_order = read_fig8_aa_matrix_order(FIG8_CSV)
    probe_lookup = read_probe_lookup(PROBE_CSV)
    test100_lookup = read_test100_lookup(TEST100_CSV)
    fig11_lookup = read_fig11_lookup(FIG11_CSV)

    rows = build_rows(matrix_order, probe_lookup, test100_lookup, fig11_lookup)
    write_fig13_csv(OUT_CSV, rows)
    draw_fig13(rows, OUT_FIG)

    print(f'Wrote {len(rows)} rows to: {OUT_CSV}')
    print(f'Wrote figure to: {OUT_FIG}')


if __name__ == '__main__':
    main()
