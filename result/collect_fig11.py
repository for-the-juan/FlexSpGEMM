#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FIG8_CSV = SCRIPT_DIR / 'data' / 'Fig8.csv'
TEST100_CSV = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test100' / 'test100_result.csv'
TEST12_LIST = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test12' / 'test12mtx.txt'
TILE_AA_LOG_DIR = PROJECT_ROOT / 'other_spgemm_methods' / 'logs' / 'TileSpGEMM' / 'AA'

OUT_A_CSV = SCRIPT_DIR / 'data' / 'Fig11.csv'
OUT_B_CSV = SCRIPT_DIR / 'data' / 'Fig11_b.csv'
OUT_FIG = SCRIPT_DIR / 'plots' / 'Fig11.png'

STEP1_RE = re.compile(r'step1[\s\S]*?Runtime is\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)
STEP2_RE = re.compile(r'step2[\s\S]*?Runtime is\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)
STEP3_RE = re.compile(r'step3[\s\S]*?Runtime is\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)
MALLOC_RE = re.compile(r'Malloc uses\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)

B_MATRIX_ORDER_PREFIXES = [
    'goodwin',
    'nemeth',
    'rma',
    'af_',
    'heart',
    'web',
    'hang',
    's3r',
    'trans',
    'pku',
    'gupta',
    'tsopf',
]


def to_float_or_zero(v):
    if v is None:
        return 0.0
    t = str(v).strip()
    if t == '':
        return 0.0
    try:
        return float(t)
    except ValueError:
        return 0.0


def b_matrix_sort_key(name: str):
    lower = name.lower()
    for idx, prefix in enumerate(B_MATRIX_ORDER_PREFIXES):
        if lower.startswith(prefix):
            return (idx, lower)
    return (len(B_MATRIX_ORDER_PREFIXES), lower)


def read_fig8_aa_matrix_order(path: Path):
    order = []
    seen = set()
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Type', '').strip() != 'AA':
                continue
            name = row.get('Matrix Name', '').strip()
            if name and name not in seen:
                seen.add(name)
                order.append(name)
    return order


def read_test100_flex_stage(path: Path):
    lookup = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('gpu', '').strip() != 'A100':
                continue
            if row.get('mode', '').strip() != 'AA':
                continue
            name = row.get('matrix_name', '').strip()
            if not name:
                continue
            lookup[name] = {
                'FlexSpGEMM_Numeric_Stage': to_float_or_zero(row.get('Numeric_Stage', '')),
                'FlexSpGEMM_Symbolic_Stage': to_float_or_zero(row.get('Symbolic_Stage', '')),
                'FlexSpGEMM_Malloc': to_float_or_zero(row.get('Malloc', '')),
            }
    return lookup


def write_fig11_a_csv(matrix_order, flex_lookup):
    OUT_A_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUT_A_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'FlexSpGEMM_Numeric_Stage',
            'FlexSpGEMM_Symbolic_Stage',
            'FlexSpGEMM_Malloc',
        ])
        for m in matrix_order:
            info = flex_lookup.get(m, {})
            n = to_float_or_zero(info.get('FlexSpGEMM_Numeric_Stage', 0.0))
            s = to_float_or_zero(info.get('FlexSpGEMM_Symbolic_Stage', 0.0))
            ml = to_float_or_zero(info.get('FlexSpGEMM_Malloc', 0.0))
            writer.writerow([m, f'{n:.3f}', f'{s:.3f}', f'{ml:.3f}'])
            rows.append({'Matrix Name': m, 'FlexSpGEMM_Numeric_Stage': n, 'FlexSpGEMM_Symbolic_Stage': s, 'FlexSpGEMM_Malloc': ml})
    return rows


def parse_test12_matrix_list(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore').strip()
    return [x.strip() for x in text.replace('、', ',').split(',') if x.strip()]


def parse_tilespgemm_aa_stage(matrix_name: str):
    log_path = TILE_AA_LOG_DIR / f'{matrix_name}.log'
    if not log_path.exists() or log_path.stat().st_size == 0:
        return 0.0, 0.0, 0.0

    text = log_path.read_text(encoding='utf-8', errors='ignore')
    m1 = STEP1_RE.search(text)
    m2 = STEP2_RE.search(text)
    m3 = STEP3_RE.search(text)
    mm = MALLOC_RE.search(text)

    step1 = float(m1.group(1)) if m1 else 0.0
    step2 = float(m2.group(1)) if m2 else 0.0
    step3 = float(m3.group(1)) if m3 else 0.0
    malloc = float(mm.group(1)) if mm else 0.0

    symbolic = step1 + step2
    numeric = step3
    return numeric, symbolic, malloc


def write_fig11_b_csv(fig11_a_rows, test12_matrices):
    a_lookup = {r['Matrix Name']: r for r in fig11_a_rows}
    out_rows = []
    ordered_matrices = sorted(test12_matrices, key=b_matrix_sort_key)

    OUT_B_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_B_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'FlexSpGEMM_Numeric_Stage',
            'FlexSpGEMM_Symbolic_Stage',
            'FlexSpGEMM_Malloc',
            'TileSpGEMM_Numeric_Stage',
            'TileSpGEMM_Symbolic_Stage',
            'TileSpGEMM_Malloc',
        ])

        for m in ordered_matrices:
            a = a_lookup.get(m, {})
            fn = to_float_or_zero(a.get('FlexSpGEMM_Numeric_Stage', 0.0))
            fs = to_float_or_zero(a.get('FlexSpGEMM_Symbolic_Stage', 0.0))
            fm = to_float_or_zero(a.get('FlexSpGEMM_Malloc', 0.0))

            tn, ts, tm = parse_tilespgemm_aa_stage(m)

            writer.writerow([
                m,
                f'{fn:.3f}', f'{fs:.3f}', f'{fm:.3f}',
                f'{tn:.3f}', f'{ts:.3f}', f'{tm:.3f}',
            ])
            out_rows.append({
                'Matrix Name': m,
                'FlexSpGEMM_Numeric_Stage': fn,
                'FlexSpGEMM_Symbolic_Stage': fs,
                'FlexSpGEMM_Malloc': fm,
                'TileSpGEMM_Numeric_Stage': tn,
                'TileSpGEMM_Symbolic_Stage': ts,
                'TileSpGEMM_Malloc': tm,
            })

    return out_rows


def draw_fig11(fig11_a_rows, fig11_b_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed. Skip plotting.')
        return

    fig, axes = plt.subplots(2, 1, figsize=(26, 13), constrained_layout=True)

    # (a) FlexSpGEMM AA: normalize by FlexSpGEMM total stage time.
    names_a = [r['Matrix Name'] for r in fig11_a_rows]
    x_a = list(range(len(names_a)))
    width_a = 0.95

    flex_num_pct = []
    flex_sym_pct = []
    flex_malloc_pct = []
    for r in fig11_a_rows:
        n = r['FlexSpGEMM_Numeric_Stage']
        s = r['FlexSpGEMM_Symbolic_Stage']
        m = r['FlexSpGEMM_Malloc']
        total = n + s + m
        if total <= 0:
            flex_num_pct.append(0.0)
            flex_sym_pct.append(0.0)
            flex_malloc_pct.append(0.0)
        else:
            flex_num_pct.append(100.0 * n / total)
            flex_sym_pct.append(100.0 * s / total)
            flex_malloc_pct.append(100.0 * m / total)

    axes[0].bar(x_a, flex_sym_pct, width=width_a, label='FlexSpGEMM_Symbolic_Stage')
    axes[0].bar(x_a, flex_num_pct, width=width_a, bottom=flex_sym_pct, label='FlexSpGEMM_Numeric_Stage')
    bottom_malloc_a = [a + b for a, b in zip(flex_sym_pct, flex_num_pct)]
    axes[0].bar(x_a, flex_malloc_pct, width=width_a, bottom=bottom_malloc_a, label='FlexSpGEMM_Malloc')

    axes[0].set_title('(a) FlexSpGEMM AA Stage Breakdown (Normalized)')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(x_a)
    axes[0].set_xticklabels(names_a, rotation=90, fontsize=8)
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(loc='upper right')

    # (b) 12 matrices: normalize by TileSpGEMM total time; plot both Tile/Flex stacked bars per matrix.
    names_b = [r['Matrix Name'] for r in fig11_b_rows]
    group_gap = 0.70
    bar_w = 0.34
    x_tile = []
    x_flex = []
    group_centers = []

    for i in range(len(names_b)):
        # Explicitly enforce order in each matrix group:
        # left bar = FlexSpGEMM, right bar = TileSpGEMM.
        center = i * (2 * bar_w + group_gap)
        x_flex.append(center - bar_w / 2)
        x_tile.append(center + bar_w / 2)
        group_centers.append(center)

    tile_sym_pct = []
    tile_num_pct = []
    tile_malloc_pct = []
    flex_sym_pct_b = []
    flex_num_pct_b = []
    flex_malloc_pct_b = []

    for r in fig11_b_rows:
        tn = r['TileSpGEMM_Numeric_Stage']
        ts = r['TileSpGEMM_Symbolic_Stage']
        tm = r['TileSpGEMM_Malloc']
        fn = r['FlexSpGEMM_Numeric_Stage']
        fs = r['FlexSpGEMM_Symbolic_Stage']
        fm = r['FlexSpGEMM_Malloc']

        tile_total = tn + ts + tm
        if tile_total <= 0:
            tile_sym_pct.append(0.0)
            tile_num_pct.append(0.0)
            tile_malloc_pct.append(0.0)
            flex_sym_pct_b.append(0.0)
            flex_num_pct_b.append(0.0)
            flex_malloc_pct_b.append(0.0)
        else:
            tile_sym_pct.append(100.0 * ts / tile_total)
            tile_num_pct.append(100.0 * tn / tile_total)
            tile_malloc_pct.append(100.0 * tm / tile_total)

            flex_sym_pct_b.append(100.0 * fs / tile_total)
            flex_num_pct_b.append(100.0 * fn / tile_total)
            flex_malloc_pct_b.append(100.0 * fm / tile_total)

    # TileSpGEMM bars
    axes[1].bar(x_tile, tile_sym_pct, width=bar_w, label='TileSpGEMM_Symbolic_Stage', color='#4C78A8')
    axes[1].bar(x_tile, tile_num_pct, width=bar_w, bottom=tile_sym_pct, label='TileSpGEMM_Numeric_Stage', color='#F58518')
    tile_bottom_m = [a + b for a, b in zip(tile_sym_pct, tile_num_pct)]
    axes[1].bar(x_tile, tile_malloc_pct, width=bar_w, bottom=tile_bottom_m, label='TileSpGEMM_Malloc', color='#54A24B')

    # FlexSpGEMM bars (same normalization baseline)
    axes[1].bar(x_flex, flex_sym_pct_b, width=bar_w, label='FlexSpGEMM_Symbolic_Stage', color='#E45756')
    axes[1].bar(x_flex, flex_num_pct_b, width=bar_w, bottom=flex_sym_pct_b, label='FlexSpGEMM_Numeric_Stage', color='#72B7B2')
    flex_bottom_m = [a + b for a, b in zip(flex_sym_pct_b, flex_num_pct_b)]
    axes[1].bar(x_flex, flex_malloc_pct_b, width=bar_w, bottom=flex_bottom_m, label='FlexSpGEMM_Malloc', color='#B279A2')

    axes[1].set_title('(b) 12 Matrices, Normalized by TileSpGEMM Total Time')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_ylim(0, 160)
    axes[1].set_xticks(group_centers)
    axes[1].set_xticklabels(names_b, rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(loc='upper right', ncol=2)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)


def main():
    aa_matrix_order = read_fig8_aa_matrix_order(FIG8_CSV)
    flex_lookup = read_test100_flex_stage(TEST100_CSV)

    fig11_a_rows = write_fig11_a_csv(aa_matrix_order, flex_lookup)

    test12_mats = parse_test12_matrix_list(TEST12_LIST)
    fig11_b_rows = write_fig11_b_csv(fig11_a_rows, test12_mats)

    draw_fig11(fig11_a_rows, fig11_b_rows)

    print(f'Wrote {len(fig11_a_rows)} rows to: {OUT_A_CSV}')
    print(f'Wrote {len(fig11_b_rows)} rows to: {OUT_B_CSV}')
    print(f'Wrote figure to: {OUT_FIG}')


if __name__ == '__main__':
    main()
