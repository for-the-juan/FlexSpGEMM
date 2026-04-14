# spECK 测试结果

生成时间: 2026-02-25T00:29:41+08:00
可执行文件: /home/stu1/guishuyun/Compare_SOTA_work/HSMU-SpGEMM/other_spgemm_code/spECK/speck
数据集目录: /home/stu1/Dataset/TileSpGEMMDataset, /home/stu1/Dataset/HYTEDataset
每次运行间隔: 5s

---

## `TileSpGEMMDataset/af_shell10.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/af_shell10.mtx
spECK     initial mallocs = 1.988608 ms
spECK  count computations = 2.319360 ms
spECK       load-balancer = 0.918528 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 23.122944 ms
spECK        malloc mat C = 2.992128 ms
spECK   num load-balancer = 0.094208 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 28.179457 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.524288 ms
--------------------------------------------------------------
time(ms):
    setup               4.308ms   7.15%
    symbolic_binning    0.919ms   1.52%
    symbolic           23.131ms  38.38%
    numeric_binning     0.094ms   0.16%
    prefix              0.000ms   0.00%
    allocate            2.992ms   4.96%
    numeric            28.194ms  46.78%
    cleanup             0.524ms   0.87%
total time =  60.270ms
perf(Gflops):
total gflops = 61.09
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/af_shell10.mtx```

## `TileSpGEMMDataset/cant.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/cant.mtx
spECK     initial mallocs = 0.024576 ms
spECK  count computations = 0.234496 ms
spECK       load-balancer = 0.713728 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 3.343360 ms
spECK        malloc mat C = 0.933888 ms
spECK   num load-balancer = 0.126976 ms
spECK     init GlobalMaps = 0.008192 ms
spECK      numeric kernel = 4.761600 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.010240 ms
--------------------------------------------------------------
time(ms):
    setup               0.259ms   2.52%
    symbolic_binning    0.714ms   6.95%
    symbolic            3.352ms  32.62%
    numeric_binning     0.127ms   1.24%
    prefix              0.000ms   0.00%
    allocate            0.934ms   9.09%
    numeric             4.777ms  46.49%
    cleanup             0.010ms   0.10%
total time =  10.275ms
perf(Gflops):
total gflops = 52.46
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/cant.mtx```

## `TileSpGEMMDataset/case39.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/case39.mtx
spECK     initial mallocs = 0.028672 ms
spECK  count computations = 0.841728 ms
spECK       load-balancer = 0.066560 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 29.369345 ms
spECK        malloc mat C = 5.491712 ms
spECK   num load-balancer = 0.079872 ms
spECK     init GlobalMaps = 2.044928 ms
spECK      numeric kernel = 70.147072 ms
spECK      Sorting kernel = 0.031744 ms
spECK             cleanup = 0.779264 ms
--------------------------------------------------------------
time(ms):
    setup               0.870ms   0.80%
    symbolic_binning    0.067ms   0.06%
    symbolic           29.377ms  26.95%
    numeric_binning     0.080ms   0.07%
    prefix              0.000ms   0.00%
    allocate            5.492ms   5.04%
    numeric            72.224ms  66.26%
    cleanup             0.779ms   0.71%
total time = 109.001ms
perf(Gflops):
total gflops = 74.29
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/case39.mtx```

## `TileSpGEMMDataset/conf5_4-8x8-05.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/conf5_4-8x8-05.mtx
spECK     initial mallocs = 0.025600 ms
spECK  count computations = 0.111616 ms
spECK       load-balancer = 0.849920 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 0.985088 ms
spECK        malloc mat C = 1.272832 ms
spECK   num load-balancer = 0.103424 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 2.163712 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.011264 ms
--------------------------------------------------------------
time(ms):
    setup               0.137ms   2.43%
    symbolic_binning    0.850ms  15.05%
    symbolic            0.992ms  17.57%
    numeric_binning     0.103ms   1.83%
    prefix              0.000ms   0.00%
    allocate            1.273ms  22.54%
    numeric             2.179ms  38.59%
    cleanup             0.011ms   0.20%
total time =   5.647ms
perf(Gflops):
total gflops = 26.48
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/conf5_4-8x8-05.mtx```

## `TileSpGEMMDataset/consph.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/consph.mtx
spECK     initial mallocs = 0.024576 ms
spECK  count computations = 0.321536 ms
spECK       load-balancer = 0.582656 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 6.482944 ms
spECK        malloc mat C = 0.787456 ms
spECK   num load-balancer = 0.070656 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 8.168448 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.010240 ms
--------------------------------------------------------------
time(ms):
    setup               0.346ms   2.09%
    symbolic_binning    0.583ms   3.52%
    symbolic            6.491ms  39.17%
    numeric_binning     0.071ms   0.43%
    prefix              0.000ms   0.00%
    allocate            0.787ms   4.75%
    numeric             8.184ms  49.38%
    cleanup             0.010ms   0.06%
total time =  16.573ms
perf(Gflops):
total gflops = 55.97
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/consph.mtx```

## `TileSpGEMMDataset/cop20k_A.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/cop20k_A.mtx
spECK     initial mallocs = 0.927744 ms
spECK  count computations = 0.147456 ms
spECK       load-balancer = 0.947200 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 1.302528 ms
spECK        malloc mat C = 1.767424 ms
spECK   num load-balancer = 0.070656 ms
spECK     init GlobalMaps = 0.008192 ms
spECK      numeric kernel = 3.286016 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.114688 ms
--------------------------------------------------------------
time(ms):
    setup               1.075ms  12.37%
    symbolic_binning    0.947ms  10.90%
    symbolic            1.310ms  15.07%
    numeric_binning     0.071ms   0.81%
    prefix              0.000ms   0.00%
    allocate            1.767ms  20.34%
    numeric             3.302ms  38.01%
    cleanup             0.115ms   1.32%
total time =   8.689ms
perf(Gflops):
total gflops = 18.39
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/cop20k_A.mtx```

## `TileSpGEMMDataset/gupta3.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/gupta3.mtx
spECK     initial mallocs = 0.029696 ms
spECK  count computations = 0.296960 ms
spECK       load-balancer = 0.060416 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 132.769791 ms
spECK        malloc mat C = 3.807232 ms
spECK   num load-balancer = 0.072704 ms
spECK     init GlobalMaps = 0.117760 ms
spECK      numeric kernel = 419.633148 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.164864 ms
--------------------------------------------------------------
time(ms):
    setup               0.327ms   0.06%
    symbolic_binning    0.060ms   0.01%
    symbolic          132.777ms  23.83%
    numeric_binning     0.073ms   0.01%
    prefix              0.000ms   0.00%
    allocate            3.807ms   0.68%
    numeric           419.759ms  75.35%
    cleanup             0.165ms   0.03%
total time = 557.073ms
perf(Gflops):
total gflops =110.28
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/gupta3.mtx```

## `TileSpGEMMDataset/mac_econ_fwd500.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/mac_econ_fwd500.mtx
spECK     initial mallocs = 1.878016 ms
spECK  count computations = 0.083968 ms
spECK       load-balancer = 0.202752 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 0.672768 ms
spECK        malloc mat C = 0.202752 ms
spECK   num load-balancer = 0.071680 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 0.784384 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.102400 ms
--------------------------------------------------------------
time(ms):
    setup               1.962ms  47.56%
    symbolic_binning    0.203ms   4.91%
    symbolic            0.680ms  16.48%
    numeric_binning     0.072ms   1.74%
    prefix              0.000ms   0.00%
    allocate            0.203ms   4.91%
    numeric             0.800ms  19.38%
    cleanup             0.102ms   2.48%
total time =   4.126ms
perf(Gflops):
total gflops =  3.66
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/mac_econ_fwd500.mtx```

## `TileSpGEMMDataset/mc2depi.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/mc2depi.mtx
spECK     initial mallocs = 1.873920 ms
spECK  count computations = 0.131072 ms
spECK       load-balancer = 0.150528 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 0.388096 ms
spECK        malloc mat C = 0.171008 ms
spECK   num load-balancer = 0.070656 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 0.564224 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.057344 ms
--------------------------------------------------------------
time(ms):
    setup               2.005ms  56.79%
    symbolic_binning    0.151ms   4.26%
    symbolic            0.395ms  11.19%
    numeric_binning     0.071ms   2.00%
    prefix              0.000ms   0.00%
    allocate            0.171ms   4.84%
    numeric             0.580ms  16.42%
    cleanup             0.057ms   1.62%
total time =   3.531ms
perf(Gflops):
total gflops =  4.75
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/mc2depi.mtx```

## `TileSpGEMMDataset/pdb1HYS.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/pdb1HYS.mtx
spECK     initial mallocs = 0.025600 ms
spECK  count computations = 0.200704 ms
spECK       load-balancer = 0.072704 ms
spECK      GlobalMaps Cnt = 0.507904 ms
spECK     counting kernel = 4.869120 ms
spECK        malloc mat C = 1.379328 ms
spECK   num load-balancer = 0.082944 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 9.119744 ms
spECK      Sorting kernel = 0.922624 ms
spECK             cleanup = 0.175104 ms
--------------------------------------------------------------
time(ms):
    setup               0.226ms   1.30%
    symbolic_binning    0.073ms   0.42%
    symbolic            5.377ms  30.77%
    numeric_binning     0.083ms   0.47%
    prefix              0.000ms   0.00%
    allocate            1.379ms   7.89%
    numeric            10.050ms  57.51%
    cleanup             0.175ms   1.00%
total time =  17.474ms
perf(Gflops):
total gflops = 63.56
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/pdb1HYS.mtx```

## `TileSpGEMMDataset/pkustk12.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/pkustk12.mtx
spECK     initial mallocs = 0.027648 ms
spECK  count computations = 0.387072 ms
spECK       load-balancer = 0.467968 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 23.171072 ms
spECK        malloc mat C = 5.122048 ms
spECK   num load-balancer = 0.091136 ms
spECK     init GlobalMaps = 0.955392 ms
spECK      numeric kernel = 117.696510 ms
spECK      Sorting kernel = 0.787456 ms
spECK             cleanup = 2.423808 ms
--------------------------------------------------------------
time(ms):
    setup               0.415ms   0.27%
    symbolic_binning    0.468ms   0.31%
    symbolic           23.178ms  15.32%
    numeric_binning     0.091ms   0.06%
    prefix              0.000ms   0.00%
    allocate            5.122ms   3.39%
    numeric           119.439ms  78.97%
    cleanup             2.424ms   1.60%
total time = 151.250ms
perf(Gflops):
total gflops = 35.44
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/pkustk12.mtx```

## `TileSpGEMMDataset/pwtk.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/pwtk.mtx
spECK     initial mallocs = 1.862656 ms
spECK  count computations = 0.545792 ms
spECK       load-balancer = 0.549888 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 7.044096 ms
spECK        malloc mat C = 0.596992 ms
spECK   num load-balancer = 0.074752 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 9.766912 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.056320 ms
--------------------------------------------------------------
time(ms):
    setup               2.408ms  11.68%
    symbolic_binning    0.550ms   2.67%
    symbolic            7.052ms  34.19%
    numeric_binning     0.075ms   0.36%
    prefix              0.000ms   0.00%
    allocate            0.597ms   2.89%
    numeric             9.781ms  47.42%
    cleanup             0.056ms   0.27%
total time =  20.627ms
perf(Gflops):
total gflops = 60.70
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/pwtk.mtx```

## `TileSpGEMMDataset/rma10.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/rma10.mtx
spECK     initial mallocs = 0.024576 ms
spECK  count computations = 0.145408 ms
spECK       load-balancer = 0.851968 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 2.012160 ms
spECK        malloc mat C = 0.235520 ms
spECK   num load-balancer = 0.067584 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 2.640896 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.010240 ms
--------------------------------------------------------------
time(ms):
    setup               0.170ms   2.78%
    symbolic_binning    0.852ms  13.94%
    symbolic            2.019ms  33.03%
    numeric_binning     0.068ms   1.11%
    prefix              0.000ms   0.00%
    allocate            0.236ms   3.85%
    numeric             2.655ms  43.43%
    cleanup             0.010ms   0.17%
total time =   6.113ms
perf(Gflops):
total gflops = 51.19
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/rma10.mtx```

## `TileSpGEMMDataset/scircuit.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/scircuit.mtx
spECK     initial mallocs = 0.859136 ms
spECK  count computations = 0.069632 ms
spECK       load-balancer = 0.745472 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 0.660480 ms
spECK        malloc mat C = 0.189440 ms
spECK   num load-balancer = 0.080896 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 0.836608 ms
spECK      Sorting kernel = 0.045056 ms
spECK             cleanup = 0.103424 ms
--------------------------------------------------------------
time(ms):
    setup               0.929ms  25.04%
    symbolic_binning    0.745ms  20.10%
    symbolic            0.668ms  18.00%
    numeric_binning     0.081ms   2.18%
    prefix              0.000ms   0.00%
    allocate            0.189ms   5.11%
    numeric             0.889ms  23.96%
    cleanup             0.103ms   2.79%
total time =   3.709ms
perf(Gflops):
total gflops =  4.68
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/scircuit.mtx```

## `TileSpGEMMDataset/shipsec1.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/shipsec1.mtx
spECK     initial mallocs = 0.898048 ms
spECK  count computations = 0.373760 ms
spECK       load-balancer = 0.716800 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 6.254592 ms
spECK        malloc mat C = 0.457728 ms
spECK   num load-balancer = 0.074752 ms
spECK     init GlobalMaps = 0.008192 ms
spECK      numeric kernel = 7.175168 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.059392 ms
--------------------------------------------------------------
time(ms):
    setup               1.272ms   7.88%
    symbolic_binning    0.717ms   4.44%
    symbolic            6.262ms  38.81%
    numeric_binning     0.075ms   0.46%
    prefix              0.000ms   0.00%
    allocate            0.458ms   2.84%
    numeric             7.191ms  44.57%
    cleanup             0.059ms   0.37%
total time =  16.134ms
perf(Gflops):
total gflops = 55.86
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/shipsec1.mtx```

## `TileSpGEMMDataset/SiO2.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/SiO2.mtx
spECK     initial mallocs = 0.036864 ms
spECK  count computations = 0.509952 ms
spECK       load-balancer = 1.048576 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 79.156227 ms
spECK        malloc mat C = 1.581056 ms
spECK   num load-balancer = 0.086016 ms
spECK     init GlobalMaps = 0.101376 ms
spECK      numeric kernel = 1045.691406 ms
spECK      Sorting kernel = 0.260096 ms
spECK             cleanup = 0.203776 ms
--------------------------------------------------------------
time(ms):
    setup               0.547ms   0.05%
    symbolic_binning    1.049ms   0.09%
    symbolic           79.163ms   7.01%
    numeric_binning     0.086ms   0.01%
    prefix              0.000ms   0.00%
    allocate            1.581ms   0.14%
    numeric          1046.053ms  92.67%
    cleanup             0.204ms   0.02%
total time =1128.795ms
perf(Gflops):
total gflops = 25.27
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/SiO2.mtx```

## `TileSpGEMMDataset/TSOPF_FS_b300_c2.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/TSOPF_FS_b300_c2.mtx
spECK     initial mallocs = 0.038912 ms
spECK  count computations = 0.427008 ms
spECK       load-balancer = 0.074752 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 298.682373 ms
spECK        malloc mat C = 11.111424 ms
spECK   num load-balancer = 0.087040 ms
spECK     init GlobalMaps = 0.169984 ms
spECK      numeric kernel = 910.288879 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.234496 ms
--------------------------------------------------------------
time(ms):
    setup               0.466ms   0.04%
    symbolic_binning    0.075ms   0.01%
    symbolic          298.690ms  24.46%
    numeric_binning     0.087ms   0.01%
    prefix              0.000ms   0.00%
    allocate           11.111ms   0.91%
    numeric           910.467ms  74.55%
    cleanup             0.234ms   0.02%
total time =1221.243ms
perf(Gflops):
total gflops = 88.35
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/TSOPF_FS_b300_c2.mtx```

## `TileSpGEMMDataset/webbase-1M.mtx`

```
/home/stu1/Dataset/TileSpGEMMDataset/webbase-1M.mtx
spECK     initial mallocs = 1.503232 ms
spECK  count computations = 1.587200 ms
spECK       load-balancer = 0.603136 ms
spECK      GlobalMaps Cnt = 0.944128 ms
spECK     counting kernel = 2.664448 ms
spECK        malloc mat C = 2.569216 ms
spECK   num load-balancer = 0.138240 ms
spECK     init GlobalMaps = 0.008192 ms
spECK      numeric kernel = 18.864128 ms
spECK      Sorting kernel = 4.528128 ms
spECK             cleanup = 0.349184 ms
--------------------------------------------------------------
time(ms):
    setup               3.090ms   9.12%
    symbolic_binning    0.603ms   1.78%
    symbolic            3.609ms  10.65%
    numeric_binning     0.138ms   0.41%
    prefix              0.000ms   0.00%
    allocate            2.569ms   7.58%
    numeric            23.400ms  69.08%
    cleanup             0.349ms   1.03%
total time =  33.873ms
perf(Gflops):
total gflops =  4.11
the cstr_matrix_name is /home/stu1/Dataset/TileSpGEMMDataset/webbase-1M.mtx```

## `HYTEDataset/af_shell9.mtx`

```
/home/stu1/Dataset/HYTEDataset/af_shell9.mtx
spECK     initial mallocs = 1.798144 ms
spECK  count computations = 0.803840 ms
spECK       load-balancer = 0.303104 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 7.677952 ms
spECK        malloc mat C = 1.674240 ms
spECK   num load-balancer = 0.086016 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 10.979328 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.219136 ms
--------------------------------------------------------------
time(ms):
    setup               2.602ms  10.99%
    symbolic_binning    0.303ms   1.28%
    symbolic            7.685ms  32.47%
    numeric_binning     0.086ms   0.36%
    prefix              0.000ms   0.00%
    allocate            1.674ms   7.07%
    numeric            10.995ms  46.45%
    cleanup             0.219ms   0.93%
total time =  23.672ms
perf(Gflops):
total gflops = 51.84
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/af_shell9.mtx```

## `HYTEDataset/cit-Patents.mtx`

```
/home/stu1/Dataset/HYTEDataset/cit-Patents.mtx
spECK     initial mallocs = 1.406976 ms
spECK  count computations = 1.012736 ms
spECK       load-balancer = 1.309696 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 9.389056 ms
spECK        malloc mat C = 1.872896 ms
spECK   num load-balancer = 0.283648 ms
spECK     init GlobalMaps = 0.060416 ms
spECK      numeric kernel = 20.026367 ms
spECK      Sorting kernel = 0.208896 ms
spECK             cleanup = 0.434176 ms
--------------------------------------------------------------
time(ms):
    setup               2.420ms   6.70%
    symbolic_binning    1.310ms   3.63%
    symbolic            9.397ms  26.01%
    numeric_binning     0.284ms   0.79%
    prefix              0.000ms   0.00%
    allocate            1.873ms   5.18%
    numeric            20.296ms  56.18%
    cleanup             0.434ms   1.20%
total time =  36.129ms
perf(Gflops):
total gflops =  4.55
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/cit-Patents.mtx```

## `HYTEDataset/CoupCons3D.mtx`

```
/home/stu1/Dataset/HYTEDataset/CoupCons3D.mtx
spECK     initial mallocs = 1.594368 ms
spECK  count computations = 1.038336 ms
spECK       load-balancer = 1.116160 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 16.060415 ms
spECK        malloc mat C = 2.747392 ms
spECK   num load-balancer = 0.081920 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 21.309441 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.356352 ms
--------------------------------------------------------------
time(ms):
    setup               2.633ms   5.93%
    symbolic_binning    1.116ms   2.51%
    symbolic           16.068ms  36.16%
    numeric_binning     0.082ms   0.18%
    prefix              0.000ms   0.00%
    allocate            2.747ms   6.18%
    numeric            21.325ms  48.00%
    cleanup             0.356ms   0.80%
total time =  44.430ms
perf(Gflops):
total gflops = 55.24
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/CoupCons3D.mtx```

## `HYTEDataset/crankseg_2.mtx`

```
/home/stu1/Dataset/HYTEDataset/crankseg_2.mtx
spECK     initial mallocs = 0.027648 ms
spECK  count computations = 0.412672 ms
spECK       load-balancer = 0.064512 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 23.027712 ms
spECK        malloc mat C = 2.110464 ms
spECK   num load-balancer = 0.084992 ms
spECK     init GlobalMaps = 0.961536 ms
spECK      numeric kernel = 80.196609 ms
spECK      Sorting kernel = 10.240000 ms
spECK             cleanup = 0.407552 ms
--------------------------------------------------------------
time(ms):
    setup               0.440ms   0.37%
    symbolic_binning    0.065ms   0.05%
    symbolic           23.036ms  19.58%
    numeric_binning     0.085ms   0.07%
    prefix              0.000ms   0.00%
    allocate            2.110ms   1.79%
    numeric            91.398ms  77.68%
    cleanup             0.408ms   0.35%
total time = 117.654ms
perf(Gflops):
total gflops = 63.28
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/crankseg_2.mtx```

## `HYTEDataset/dielFilterV2real.mtx`

```
/home/stu1/Dataset/HYTEDataset/dielFilterV2real.mtx
spECK     initial mallocs = 1.519616 ms
spECK  count computations = 2.597888 ms
spECK       load-balancer = 0.642048 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 40.429569 ms
spECK        malloc mat C = 4.589568 ms
spECK   num load-balancer = 0.094208 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 71.741440 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 1.141760 ms
--------------------------------------------------------------
time(ms):
    setup               4.118ms   3.35%
    symbolic_binning    0.642ms   0.52%
    symbolic           40.437ms  32.91%
    numeric_binning     0.094ms   0.08%
    prefix              0.000ms   0.00%
    allocate            4.590ms   3.73%
    numeric            71.756ms  58.39%
    cleanup             1.142ms   0.93%
total time = 122.888ms
perf(Gflops):
total gflops = 38.04
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/dielFilterV2real.mtx```

## `HYTEDataset/Fault_639.mtx`

```
/home/stu1/Dataset/HYTEDataset/Fault_639.mtx
spECK     initial mallocs = 2.044928 ms
spECK  count computations = 1.362944 ms
spECK       load-balancer = 0.805888 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 15.827968 ms
spECK        malloc mat C = 2.965504 ms
spECK   num load-balancer = 0.113664 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 29.338623 ms
spECK      Sorting kernel = 0.031744 ms
spECK             cleanup = 0.538624 ms
--------------------------------------------------------------
time(ms):
    setup               3.408ms   6.41%
    symbolic_binning    0.806ms   1.52%
    symbolic           15.835ms  29.79%
    numeric_binning     0.114ms   0.21%
    prefix              0.000ms   0.00%
    allocate            2.966ms   5.58%
    numeric            29.378ms  55.27%
    cleanup             0.539ms   1.01%
total time =  53.154ms
perf(Gflops):
total gflops = 48.87
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/Fault_639.mtx```

## `HYTEDataset/fem_hifreq_circuit.mtx`

```
/home/stu1/Dataset/HYTEDataset/fem_hifreq_circuit.mtx
spECK     initial mallocs = 1.991680 ms
spECK  count computations = 1.083392 ms
spECK       load-balancer = 1.080320 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 15.516672 ms
spECK        malloc mat C = 2.231296 ms
spECK   num load-balancer = 0.105472 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 29.326336 ms
spECK      Sorting kernel = 0.031744 ms
spECK             cleanup = 0.568320 ms
--------------------------------------------------------------
time(ms):
    setup               3.075ms   5.91%
    symbolic_binning    1.080ms   2.08%
    symbolic           15.524ms  29.82%
    numeric_binning     0.105ms   0.20%
    prefix              0.000ms   0.00%
    allocate            2.231ms   4.29%
    numeric            29.365ms  56.41%
    cleanup             0.568ms   1.09%
total time =  52.061ms
perf(Gflops):
total gflops = 37.01
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/fem_hifreq_circuit.mtx```

## `HYTEDataset/filter3D.mtx`

```
/home/stu1/Dataset/HYTEDataset/filter3D.mtx
spECK     initial mallocs = 0.922624 ms
spECK  count computations = 0.149504 ms
spECK       load-balancer = 0.941056 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 1.603584 ms
spECK        malloc mat C = 1.467392 ms
spECK   num load-balancer = 0.072704 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 3.596288 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.120832 ms
--------------------------------------------------------------
time(ms):
    setup               1.072ms  11.91%
    symbolic_binning    0.941ms  10.45%
    symbolic            1.611ms  17.89%
    numeric_binning     0.073ms   0.81%
    prefix              0.000ms   0.00%
    allocate            1.467ms  16.30%
    numeric             3.611ms  40.10%
    cleanup             0.121ms   1.34%
total time =   9.004ms
perf(Gflops):
total gflops = 19.09
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/filter3D.mtx```

## `HYTEDataset/kkt_power.mtx`

```
/home/stu1/Dataset/HYTEDataset/kkt_power.mtx
spECK     initial mallocs = 1.241088 ms
spECK  count computations = 0.683008 ms
spECK       load-balancer = 0.502784 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 9.528320 ms
spECK        malloc mat C = 2.885632 ms
spECK   num load-balancer = 0.183296 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 14.216192 ms
spECK      Sorting kernel = 0.046080 ms
spECK             cleanup = 0.161792 ms
--------------------------------------------------------------
time(ms):
    setup               1.924ms   6.51%
    symbolic_binning    0.503ms   1.70%
    symbolic            9.535ms  32.24%
    numeric_binning     0.183ms   0.62%
    prefix              0.000ms   0.00%
    allocate            2.886ms   9.76%
    numeric            14.269ms  48.24%
    cleanup             0.162ms   0.55%
total time =  29.578ms
perf(Gflops):
total gflops = 14.64
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/kkt_power.mtx```

## `HYTEDataset/kron_g500-logn18.mtx`

**（运行失败/崩溃，退出码: 134）**

```
terminate called after throwing an instance of 'std::exception'
  what():  std::exception
```

## `HYTEDataset/ldoor.mtx`

```
/home/stu1/Dataset/HYTEDataset/ldoor.mtx
spECK     initial mallocs = 1.337344 ms
spECK  count computations = 2.191360 ms
spECK       load-balancer = 1.034240 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 32.519169 ms
spECK        malloc mat C = 3.059712 ms
spECK   num load-balancer = 0.089088 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 41.786369 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.534528 ms
--------------------------------------------------------------
time(ms):
    setup               3.529ms   4.27%
    symbolic_binning    1.034ms   1.25%
    symbolic           32.526ms  39.34%
    numeric_binning     0.089ms   0.11%
    prefix              0.000ms   0.00%
    allocate            3.060ms   3.70%
    numeric            41.802ms  50.56%
    cleanup             0.535ms   0.65%
total time =  82.683ms
perf(Gflops):
total gflops = 58.27
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/ldoor.mtx```

## `HYTEDataset/mouse_gene.mtx`

```
/home/stu1/Dataset/HYTEDataset/mouse_gene.mtx
spECK     initial mallocs = 0.069632 ms
spECK  count computations = 0.919552 ms
spECK       load-balancer = 0.122880 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 324.390900 ms
spECK        malloc mat C = 6.849536 ms
spECK   num load-balancer = 0.111616 ms
spECK     init GlobalMaps = 0.201728 ms
spECK      numeric kernel = 1801.886719 ms
spECK      Sorting kernel = 0.606208 ms
spECK             cleanup = 0.268288 ms
--------------------------------------------------------------
time(ms):
    setup               0.989ms   0.05%
    symbolic_binning    0.123ms   0.01%
    symbolic          324.398ms  15.19%
    numeric_binning     0.112ms   0.01%
    prefix              0.000ms   0.00%
    allocate            6.850ms   0.32%
    numeric          1802.695ms  84.41%
    cleanup             0.268ms   0.01%
total time =2135.547ms
perf(Gflops):
total gflops = 48.42
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/mouse_gene.mtx```

## `HYTEDataset/mycielskian16.mtx`

```
/home/stu1/Dataset/HYTEDataset/mycielskian16.mtx
spECK     initial mallocs = 0.058368 ms
spECK  count computations = 1.047552 ms
spECK       load-balancer = 0.074752 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 439.015411 ms
spECK        malloc mat C = 26.013697 ms
spECK   num load-balancer = 0.093184 ms
spECK     init GlobalMaps = 0.974848 ms
spECK      numeric kernel = 2377.638916 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 7.790592 ms
--------------------------------------------------------------
time(ms):
    setup               1.106ms   0.04%
    symbolic_binning    0.075ms   0.00%
    symbolic          439.023ms  15.39%
    numeric_binning     0.093ms   0.00%
    prefix              0.000ms   0.00%
    allocate           26.014ms   0.91%
    numeric          2378.622ms  83.38%
    cleanup             7.791ms   0.27%
total time =2852.837ms
perf(Gflops):
total gflops = 56.05
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/mycielskian16.mtx```

## `HYTEDataset/nd24k.mtx`

```
/home/stu1/Dataset/HYTEDataset/nd24k.mtx
spECK     initial mallocs = 0.028672 ms
spECK  count computations = 0.590848 ms
spECK       load-balancer = 0.061440 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 64.586754 ms
spECK        malloc mat C = 2.676736 ms
spECK   num load-balancer = 0.078848 ms
spECK     init GlobalMaps = 0.092160 ms
spECK      numeric kernel = 250.231812 ms
spECK      Sorting kernel = 8.307712 ms
spECK             cleanup = 0.073728 ms
--------------------------------------------------------------
time(ms):
    setup               0.620ms   0.19%
    symbolic_binning    0.061ms   0.02%
    symbolic           64.595ms  19.76%
    numeric_binning     0.079ms   0.02%
    prefix              0.000ms   0.00%
    allocate            2.677ms   0.82%
    numeric           258.632ms  79.13%
    cleanup             0.074ms   0.02%
total time = 326.846ms
perf(Gflops):
total gflops = 72.69
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/nd24k.mtx```

## `HYTEDataset/pwtk.mtx`

```
/home/stu1/Dataset/HYTEDataset/pwtk.mtx
spECK     initial mallocs = 1.863680 ms
spECK  count computations = 0.543744 ms
spECK       load-balancer = 0.551936 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 7.044096 ms
spECK        malloc mat C = 0.601088 ms
spECK   num load-balancer = 0.074752 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 9.764864 ms
spECK      Sorting kernel = 0.008192 ms
spECK             cleanup = 0.057344 ms
--------------------------------------------------------------
time(ms):
    setup               2.407ms  11.67%
    symbolic_binning    0.552ms   2.67%
    symbolic            7.051ms  34.17%
    numeric_binning     0.075ms   0.36%
    prefix              0.000ms   0.00%
    allocate            0.601ms   2.91%
    numeric             9.780ms  47.40%
    cleanup             0.057ms   0.28%
total time =  20.634ms
perf(Gflops):
total gflops = 60.68
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/pwtk.mtx```

## `HYTEDataset/ship_001.mtx`

```
/home/stu1/Dataset/HYTEDataset/ship_001.mtx
spECK     initial mallocs = 0.024576 ms
spECK  count computations = 0.199680 ms
spECK       load-balancer = 0.070656 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 6.615040 ms
spECK        malloc mat C = 0.956416 ms
spECK   num load-balancer = 0.083968 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 12.432384 ms
spECK      Sorting kernel = 2.258944 ms
spECK             cleanup = 0.014336 ms
--------------------------------------------------------------
time(ms):
    setup               0.224ms   0.98%
    symbolic_binning    0.071ms   0.31%
    symbolic            6.623ms  29.08%
    numeric_binning     0.084ms   0.37%
    prefix              0.000ms   0.00%
    allocate            0.956ms   4.20%
    numeric            14.698ms  64.53%
    cleanup             0.014ms   0.06%
total time =  22.778ms
perf(Gflops):
total gflops = 63.58
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/ship_001.mtx```

## `HYTEDataset/TSOPF_FS_b300_c3.mtx`

```
/home/stu1/Dataset/HYTEDataset/TSOPF_FS_b300_c3.mtx
spECK     initial mallocs = 0.050176 ms
spECK  count computations = 0.629760 ms
spECK       load-balancer = 0.188416 ms
spECK      GlobalMaps Cnt = 0.007168 ms
spECK     counting kernel = 666.067993 ms
spECK        malloc mat C = 23.991297 ms
spECK   num load-balancer = 0.100352 ms
spECK     init GlobalMaps = 0.242688 ms
spECK      numeric kernel = 1918.063599 ms
spECK      Sorting kernel = 0.007168 ms
spECK             cleanup = 0.315392 ms
--------------------------------------------------------------
time(ms):
    setup               0.680ms   0.03%
    symbolic_binning    0.188ms   0.01%
    symbolic          666.075ms  25.52%
    numeric_binning     0.100ms   0.00%
    prefix              0.000ms   0.00%
    allocate           23.991ms   0.92%
    numeric          1918.313ms  73.50%
    cleanup             0.315ms   0.01%
total time =2609.779ms
perf(Gflops):
total gflops = 91.82
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/TSOPF_FS_b300_c3.mtx```

## `HYTEDataset/web-Google.mtx`

```
/home/stu1/Dataset/HYTEDataset/web-Google.mtx
spECK     initial mallocs = 1.699840 ms
spECK  count computations = 0.293888 ms
spECK       load-balancer = 0.833536 ms
spECK      GlobalMaps Cnt = 0.008192 ms
spECK     counting kernel = 3.750912 ms
spECK        malloc mat C = 2.427904 ms
spECK   num load-balancer = 0.129024 ms
spECK     init GlobalMaps = 0.007168 ms
spECK      numeric kernel = 5.266432 ms
spECK      Sorting kernel = 0.125952 ms
spECK             cleanup = 0.221184 ms
--------------------------------------------------------------
time(ms):
    setup               1.994ms  13.40%
    symbolic_binning    0.834ms   5.60%
    symbolic            3.759ms  25.27%
    numeric_binning     0.129ms   0.87%
    prefix              0.000ms   0.00%
    allocate            2.428ms  16.32%
    numeric             5.400ms  36.30%
    cleanup             0.221ms   1.49%
total time =  14.874ms
perf(Gflops):
total gflops =  8.16
the cstr_matrix_name is /home/stu1/Dataset/HYTEDataset/web-Google.mtx```

