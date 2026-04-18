#!/usr/bin/env python3
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LOG_DIR = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test100' / 'log'
OUT_CSV = SCRIPT_DIR / 'data' / 'Fig12.csv'
OUT_FIG = SCRIPT_DIR / 'plots' / 'Fig12.png'

CSR_RE = re.compile(r'CSR Memory Cost\s*:\s*([0-9]*\.?[0-9]+)\s*MB', re.IGNORECASE)
DENSE_RE = re.compile(r'Dense Memory Cost\s*:\s*([0-9]*\.?[0-9]+)\s*MB', re.IGNORECASE)
TILE_RE = re.compile(r'TileSpGEMM Memory Cost\s*:\s*([0-9]*\.?[0-9]+)\s*MB', re.IGNORECASE)
FLEX_RE = re.compile(r'FlexSpGEMM Memory Cost\s*:\s*([0-9]*\.?[0-9]+)\s*MB', re.IGNORECASE)


def parse_aa_log(log_path: Path):
    name = log_path.name
    if '_AA_' not in name or '_AAT_' in name:
        return None

    matrix = name.split('_AA_')[0]
    text = log_path.read_text(encoding='utf-8', errors='ignore')

    csr_m = CSR_RE.search(text)
    dense_m = DENSE_RE.search(text)
    tile_m = TILE_RE.search(text)
    flex_m = FLEX_RE.search(text)

    if not (csr_m and dense_m and tile_m and flex_m):
        return None

    return {
        'Matrix Name': matrix,
        'Dense Memory Cost': float(dense_m.group(1)),
        'FlexSpGEMM Memory Cost': float(flex_m.group(1)),
        'TileSpGEMM Memory Cost': float(tile_m.group(1)),
        'CSR Memory Cost': float(csr_m.group(1)),
    }


def collect_rows():
    rows = []
    for log_file in sorted(LOG_DIR.glob('*.log')):
        row = parse_aa_log(log_file)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (r['CSR Memory Cost'], r['Matrix Name']))
    return rows


def write_csv(rows):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'Dense Memory Cost',
            'FlexSpGEMM Memory Cost',
            'TileSpGEMM Memory Cost',
            'CSR Memory Cost',
        ])
        for r in rows:
            writer.writerow([
                r['Matrix Name'],
                f"{r['Dense Memory Cost']:.2f}",
                f"{r['FlexSpGEMM Memory Cost']:.2f}",
                f"{r['TileSpGEMM Memory Cost']:.2f}",
                f"{r['CSR Memory Cost']:.2f}",
            ])


def draw_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed. Skip plotting.')
        return

    names = [r['Matrix Name'] for r in rows]
    x = list(range(len(names)))

    dense = [r['Dense Memory Cost'] for r in rows]
    flex = [r['FlexSpGEMM Memory Cost'] for r in rows]
    tile = [r['TileSpGEMM Memory Cost'] for r in rows]
    csr = [r['CSR Memory Cost'] for r in rows]

    fig, ax = plt.subplots(figsize=(26, 8), constrained_layout=True)

    ax.scatter(x, dense, s=14, label='Dense Memory Cost')
    ax.scatter(x, flex, s=14, label='FlexSpGEMM Memory Cost')
    ax.scatter(x, tile, s=14, label='TileSpGEMM Memory Cost')
    ax.scatter(x, csr, s=14, label='CSR Memory Cost')

    ax.set_xlabel('Matrices')
    ax.set_ylabel('Memory Cost (MB)')
    ax.set_yscale('log', base=10)
    ax.set_ylim(0.1, 10000)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.grid(True, which='both', axis='y', alpha=0.25)
    ax.legend(loc='upper left', ncol=2)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)


def main():
    rows = collect_rows()
    write_csv(rows)
    draw_plot(rows)

    print(f'Wrote {len(rows)} rows to: {OUT_CSV}')
    print(f'Wrote figure to: {OUT_FIG}')


if __name__ == '__main__':
    main()
