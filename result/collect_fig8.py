#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FLEX_LOG_DIR = PROJECT_ROOT / 'ML_method' / 'LightGBM' / 'predictResult_test100' / 'log'
HSMU_LOG_DIR = PROJECT_ROOT / 'other_spgemm_methods' / 'logs' / 'HSMU'
SPECK_LOG_DIR = PROJECT_ROOT / 'other_spgemm_methods' / 'logs' / 'spECK'
TILESPGEMM_LOG_DIR = PROJECT_ROOT / 'other_spgemm_methods' / 'logs' / 'TileSpGEMM'

DATA_DIR = SCRIPT_DIR / 'data'
PLOTS_DIR = SCRIPT_DIR / 'plots'
OUT_CSV = DATA_DIR / 'Fig8.csv'
OUT_FIG = PLOTS_DIR / 'Fig8.png'

FILE_RE = re.compile(r'^(?P<matrix>.+)_(?P<calc>AAT|AA)_[0-9]+x[0-9]+_tc\d+\.log$')
UPPER_RE = re.compile(r'NNZ Upper Bound \(nnzCub\)\s*:\s*(\d+)')
NNZC_RE = re.compile(r'NNZ \(C\)\s*:\s*(\d+)')
THROUGHPUT_RE = re.compile(r'Throughput\s*:\s*([0-9]*\.?[0-9]+)\s*GFlops', re.IGNORECASE)

CUSPARSE_GFLOPS_RE = re.compile(r'cusparse gflops\s*=\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
HSMU_GFLOPS_RE = re.compile(r'the gflops is\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
SPECK_GFLOPS_RE = re.compile(r'total gflops\s*=\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
TILESPGEMM_GFLOPS_RE = re.compile(r'gflops\s*=\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


def parse_flex_log(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        return None

    text = path.read_text(encoding='utf-8', errors='ignore')

    upper_m = UPPER_RE.search(text)
    nnzc_m = NNZC_RE.search(text)
    thr_m = THROUGHPUT_RE.search(text)
    if not (upper_m and nnzc_m and thr_m):
        return None

    upper = int(upper_m.group(1))
    nnzc = int(nnzc_m.group(1))
    throughput = float(thr_m.group(1))
    if nnzc == 0:
        return None

    return {
        'Matrix Name': m.group('matrix'),
        'Type': m.group('calc'),
        'Compression Rate': upper / nnzc,
        'FlexSpGEMM GFLOPS': throughput,
    }


def parse_hsmu_log(matrix_name: str, calc_type: str):
    hsmu_path = HSMU_LOG_DIR / calc_type / f'{matrix_name}.log'
    if not hsmu_path.exists() or hsmu_path.stat().st_size == 0:
        return None, None

    text = hsmu_path.read_text(encoding='utf-8', errors='ignore')
    lower = text.lower()

    if 'cusparse failed' in lower:
        cusparse_gflops = None
    else:
        m = CUSPARSE_GFLOPS_RE.search(text)
        cusparse_gflops = float(m.group(1)) if m else None

    cleaned = re.sub(r'cusparse\s+failed!?', ' ', lower, flags=re.IGNORECASE)
    has_hsmu_issue = re.search(r'fail|wrong', cleaned, re.IGNORECASE) is not None
    if has_hsmu_issue:
        hsmu_gflops = None
    else:
        m = HSMU_GFLOPS_RE.search(text)
        hsmu_gflops = float(m.group(1)) if m else None

    return cusparse_gflops, hsmu_gflops


def parse_optional_gflops(log_root: Path, matrix_name: str, calc_type: str, pattern: re.Pattern):
    path = log_root / calc_type / f'{matrix_name}.log'
    if not path.exists() or path.stat().st_size == 0:
        return None

    text = path.read_text(encoding='utf-8', errors='ignore')
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def format_optional_float(value, digits=6):
    if value is None:
        return ''
    return f'{value:.{digits}f}'


def fit_line_on_log_x(points):
    if len(points) < 2:
        return None

    xs = [math.log10(x) for x, _ in points if x > 0]
    ys = [y for x, y in points if x > 0]
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    a = cov_xy / var_x
    b = mean_y - a * mean_x
    return a, b


def make_fig8_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed. Skip plotting.')
        return

    panels = [
        ('cuSPARSE GFLOPS', 'cuSPARSE'),
        ('spECK GFLOPS', 'spECK'),
        ('HSMU-SpGEMM GFLOPS', 'HSMU-SpGEMM'),
        ('TileSpGEMM GFLOPS', 'TileSpGEMM'),
        ('FlexSpGEMM GFLOPS', 'FlexSpGEMM'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), constrained_layout=True)

    for ax, (column, title) in zip(axes, panels):
        points = []
        for row in rows:
            x = row['Compression Rate']
            y = row.get(column)
            if y is None:
                continue
            if x <= 0:
                continue
            points.append((x, y))

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, s=14, alpha=0.75)

            line_model = fit_line_on_log_x(points)
            if line_model is not None:
                a, b = line_model
                x_min = max(0.1, min(xs))
                x_max = min(1000.0, max(xs))
                if x_max > x_min:
                    steps = 200
                    lx_min = math.log10(x_min)
                    lx_max = math.log10(x_max)
                    fit_x = [10 ** (lx_min + (lx_max - lx_min) * i / (steps - 1)) for i in range(steps)]
                    fit_y = [a * math.log10(x) + b for x in fit_x]
                    ax.plot(fit_x, fit_y, linewidth=1.8)

        ax.set_title(title)
        ax.set_xscale('log', base=10)
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(0, 300)
        ax.grid(True, which='both', alpha=0.25)

    axes[0].set_ylabel('GFLOPS')
    for ax in axes:
        ax.set_xlabel('Compression Rate')

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)
    print(f'Wrote figure to: {OUT_FIG}')


def main():
    rows = []

    for flex_file in sorted(FLEX_LOG_DIR.glob('*.log')):
        row = parse_flex_log(flex_file)
        if row is None:
            continue

        matrix_name = row['Matrix Name']
        calc_type = row['Type']

        cusparse_gflops, hsmu_gflops = parse_hsmu_log(matrix_name, calc_type)
        speck_gflops = parse_optional_gflops(SPECK_LOG_DIR, matrix_name, calc_type, SPECK_GFLOPS_RE)
        tilespgemm_gflops = parse_optional_gflops(TILESPGEMM_LOG_DIR, matrix_name, calc_type, TILESPGEMM_GFLOPS_RE)

        row['cuSPARSE GFLOPS'] = cusparse_gflops
        row['HSMU-SpGEMM GFLOPS'] = hsmu_gflops
        row['spECK GFLOPS'] = speck_gflops
        row['TileSpGEMM GFLOPS'] = tilespgemm_gflops
        rows.append(row)

    rows.sort(key=lambda r: (r['Compression Rate'], r['Matrix Name']))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix Name',
            'Type',
            'Compression Rate',
            'FlexSpGEMM GFLOPS',
            'cuSPARSE GFLOPS',
            'HSMU-SpGEMM GFLOPS',
            'spECK GFLOPS',
            'TileSpGEMM GFLOPS',
        ])
        for r in rows:
            writer.writerow([
                r['Matrix Name'],
                r['Type'],
                f"{r['Compression Rate']:.10f}",
                f"{r['FlexSpGEMM GFLOPS']:.6f}",
                format_optional_float(r.get('cuSPARSE GFLOPS')),
                format_optional_float(r.get('HSMU-SpGEMM GFLOPS')),
                format_optional_float(r.get('spECK GFLOPS')),
                format_optional_float(r.get('TileSpGEMM GFLOPS')),
            ])

    print(f'Wrote {len(rows)} rows to: {OUT_CSV}')
    make_fig8_plot(rows)


if __name__ == '__main__':
    main()
