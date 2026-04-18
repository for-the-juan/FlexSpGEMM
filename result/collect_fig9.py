#!/usr/bin/env python3
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST12_CSV = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test12' / 'test12_result.csv'
FIG8_CSV = SCRIPT_DIR / 'data' / 'Fig8.csv'
OUT_CSV = SCRIPT_DIR / 'data' / 'Fig9.csv'
OUT_FIG = SCRIPT_DIR / 'plots' / 'Fig9.png'

METHOD_COLUMNS = [
    'cuSPARSE GFLOPS',
    'spECK GFLOPS',
    'HSMU-SpGEMM GFLOPS',
    'TileSpGEMM GFLOPS',
    'FlexSpGEMM GFLOPS',
]

AA_MATRIX_ORDER_PREFIXES = [
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

AAT_MATRIX_ORDER_PREFIXES = [
    'goodwin',
    'rma',
    'heart',
    'web',
    'trans',
]


def to_float_or_none(value: str):
    if value is None:
        return None
    text = value.strip()
    if text == '':
        return None
    try:
        return float(text)
    except ValueError:
        return None


def matrix_sort_key(name: str, mode: str):
    prefixes = AA_MATRIX_ORDER_PREFIXES if mode == 'AA' else AAT_MATRIX_ORDER_PREFIXES
    lower = name.lower()
    for idx, prefix in enumerate(prefixes):
        if lower.startswith(prefix):
            return (idx, lower)
    return (len(prefixes), lower)


def value_or_zero(v):
    return 0.0 if v is None else v


def read_fig8_lookup(path: Path):
    lookup = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix = row['Matrix Name'].strip()
            mode = row['Type'].strip().upper()
            key = (matrix, mode)
            lookup[key] = {
                'cuSPARSE GFLOPS': to_float_or_none(row.get('cuSPARSE GFLOPS', '')),
                'spECK GFLOPS': to_float_or_none(row.get('spECK GFLOPS', '')),
                'HSMU-SpGEMM GFLOPS': to_float_or_none(row.get('HSMU-SpGEMM GFLOPS', '')),
                'TileSpGEMM GFLOPS': to_float_or_none(row.get('TileSpGEMM GFLOPS', '')),
            }
    return lookup


def build_fig9_rows(test12_path: Path, fig8_lookup):
    rows = []
    with test12_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix = row['matrix_name'].strip()
            mode = row['mode'].strip().upper()
            if mode not in ('AA', 'AAT'):
                continue

            flex_gflops = to_float_or_none(row.get('gflops', ''))
            other = fig8_lookup.get((matrix, mode), {})

            rows.append({
                'Matrix Name': matrix,
                'Type': mode,
                'cuSPARSE GFLOPS': other.get('cuSPARSE GFLOPS'),
                'spECK GFLOPS': other.get('spECK GFLOPS'),
                'HSMU-SpGEMM GFLOPS': other.get('HSMU-SpGEMM GFLOPS'),
                'TileSpGEMM GFLOPS': other.get('TileSpGEMM GFLOPS'),
                'FlexSpGEMM GFLOPS': flex_gflops,
            })

    rows.sort(key=lambda r: (r['Type'], matrix_sort_key(r['Matrix Name'], r['Type'])))
    return rows


def write_fig9_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'Type',
            'cuSPARSE GFLOPS',
            'spECK GFLOPS',
            'HSMU-SpGEMM GFLOPS',
            'TileSpGEMM GFLOPS',
            'FlexSpGEMM GFLOPS',
        ])

        for r in rows:
            writer.writerow([
                r['Matrix Name'],
                r['Type'],
                f"{value_or_zero(r['cuSPARSE GFLOPS']):.6f}",
                f"{value_or_zero(r['spECK GFLOPS']):.6f}",
                f"{value_or_zero(r['HSMU-SpGEMM GFLOPS']):.6f}",
                f"{value_or_zero(r['TileSpGEMM GFLOPS']):.6f}",
                f"{value_or_zero(r['FlexSpGEMM GFLOPS']):.6f}",
            ])


def plot_mode_bars(ax, mode_rows, panel_title):
    # Grouped bars with explicit gaps between matrix groups.
    bar_width = 0.16
    method_count = len(METHOD_COLUMNS)
    group_width = method_count * bar_width
    gap = 0.24

    group_starts = [i * (group_width + gap) for i in range(len(mode_rows))]
    group_centers = [s + (group_width - bar_width) / 2 for s in group_starts]
    mode = mode_rows[0]['Type'] if mode_rows else 'AA'
    ordered_rows = sorted(mode_rows, key=lambda r: matrix_sort_key(r['Matrix Name'], mode))
    matrix_labels = [r['Matrix Name'] for r in ordered_rows]

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']

    for mi, method in enumerate(METHOD_COLUMNS):
        xs = [s + mi * bar_width for s in group_starts]
        heights = []
        for r in ordered_rows:
            v = r.get(method)
            heights.append(value_or_zero(v))

        bars = ax.bar(xs, heights, width=bar_width, label=method, color=colors[mi], edgecolor='black', linewidth=0.3)

        # Add value labels on each bar. Missing values are labeled as N/A.
        for bar, r in zip(bars, ordered_rows):
            v = value_or_zero(r.get(method))
            x = bar.get_x() + bar.get_width() / 2
            y = min(v, 200)
            ax.text(x, y + 1.0, f'{v:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)

    # Put subfigure labels below each subplot to avoid overlapping with legend/caption.
    ax.text(0.5, -0.33, panel_title, transform=ax.transAxes, ha='center', va='top', fontsize=12, clip_on=False)
    ax.set_ylim(0, 200)
    ax.set_ylabel('GFLOPS')
    ax.set_xticks(group_centers)
    ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.25)


def draw_fig9(rows, out_fig: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed. Skip plotting.')
        return

    aa_rows = [r for r in rows if r['Type'] == 'AA']
    aat_rows = [r for r in rows if r['Type'] == 'AAT']

    fig, axes = plt.subplots(2, 1, figsize=(22, 12), constrained_layout=True)

    plot_mode_bars(axes[0], aa_rows, '(a) AA')
    plot_mode_bars(axes[1], aat_rows, '(b) AAT')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def main():
    fig8_lookup = read_fig8_lookup(FIG8_CSV)
    rows = build_fig9_rows(TEST12_CSV, fig8_lookup)

    write_fig9_csv(OUT_CSV, rows)
    draw_fig9(rows, OUT_FIG)

    print(f'Wrote {len(rows)} rows to: {OUT_CSV}')
    print(f'Wrote figure to: {OUT_FIG}')


if __name__ == '__main__':
    main()
