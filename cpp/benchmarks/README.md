# C++ CPU backend benchmark

Two scripts compare the speed of SPECULA on the CPU with numpy and with the C++
backend (`specula_cpp`), at the scale of small SCAO simulations with several
wavefront sensors and science cameras:

- `bench_objects.py`: each main processing object on its own, with synthetic inputs.
- `bench_scao.py`: a whole open-loop simulation (`scao_multi.yml`), in three
  configurations: numpy, the C++ backend, and the C++ backend with the
  `psf_dt` approximation on the science cameras.

Each backend or configuration runs in its own Python process, so that its memory
and caches do not affect the others. Both scripts print comparison tables. The
reference results below were measured with them.

A third script, `bench_approx.py`, measures the error of the approximations that
reduce the time further (see [Approximations](#approximations)).

## What is measured

### `bench_objects.py`

Every object gets a 160 x 160 pixel pupil (8 m telescope, 0.05 m/pixel, 10%
central obstruction) with a 100 nm RMS random phase, or the equivalent synthetic
input. The time is the median of 10 steps (30 for the CCD) after one warm-up
step, for a whole step: `trigger()` plus `post_trigger()`.

| Object | Configuration |
|---|---|
| Pyramid | `ModulatedPyramid`, 750 nm, pupil diameter 30 and distance 36 subapertures, FoV 2", modulation 3 lambda/D (24 steps), 80 x 80 output; 480 x 480 FFTs |
| SH, NGS | `SH`, 20 x 20 subapertures, 750 nm, 6 x 6 pixels of 0.15", FoV 1.2" |
| SH, LGS | Same, with a `LaserLaunchTelescope` (Gaussian spot of 1"), so a convolution kernel per subaperture |
| PSF | `PSF` at 1650 nm, nd = 8 (1280 x 1280) and nd = 4 (640 x 640) |
| Propagation | `AtmoPropagation` of 1 on-axis and 3 off-axis (10") sources through 4 layers (0, 2, 6, 12 km) and a pupil stop |
| Atmosphere | `AtmoEvolution` of the same 4 layers, phase screens of 1024 x 1024, wind 10-25 m/s in 4 directions |
| CCD | `CCD` 120 x 120, photon and readout noise |

### `bench_scao.py`

`scao_multi.yml` is an open-loop simulation (no reconstructor, so no calibration
data is needed), 50 steps of 1 ms:

- 4 atmospheric layers (0, 2, 6, 12 km), seeing 0.8", screens of 1024 x 1024;
- an on-axis NGS at 750 nm, seen by a modulated pyramid with an 80 x 80 CCD and
  by a 20 x 20 SH with a 120 x 120 CCD (both as in `bench_objects.py`);
- 4 science cameras (`PSF`): on axis at 1650 nm with nd = 8, and three sources
  10" off axis at 1250, 1650 and 2200 nm with nd = 4.

Times come from the trace summary of each simulation (`--trace-file`), skipping
the first 5 steps: the time per step of the whole loop, and of each object (all
its phases). The `psf_dt` configuration computes the PSFs every 5 ms, which
approximates the long-exposure PSF; the Strehl ratios are still exact at every step.

## How to repeat it

1. Check out the commit to measure. The reference results below say which commit
   of the `cpp-cpu-backend` branch they are for.

2. Create a clean virtual environment with SPECULA's dependencies, and build the
   C++ backend (needs a C++17 compiler; CMake, nanobind and scikit-build-core are
   installed automatically for the build):

   ```bash
   python3 -m venv venv
   ./venv/bin/pip install numpy scipy astropy matplotlib pyyaml scikit-image \
       "symao>=1.0.1" "astro-seeing>=1.1" "synim==1.3.0"
   ./venv/bin/pip install ./cpp
   ```

   SPECULA itself does not need to be installed: the scripts import it from the
   repository.

3. Make the machine as quiet as possible: close other applications, wait for
   background indexing to finish, and check the load average (`uptime`). The C++
   backend uses all the cores, so it is the most sensitive to other load.

4. Run the two scripts from the repository root, one after the other:

   ```bash
   ./venv/bin/python cpp/benchmarks/bench_objects.py
   ./venv/bin/python cpp/benchmarks/bench_scao.py
   ```

   Each takes about one minute and uses less than 0.5 GB of memory.
   `cpp/benchmarks/bench_approx.py` (about 30 s) measures the approximations.

The number of threads of the C++ backend defaults to the CPUs available to the
process; set `SPECULA_CPP_THREADS` to change it. All the computations are in
single precision (`precision=1`), as in the default `specula` command line.

## Reference results

### AMD Ryzen 5 5600, Linux

Measured on 2026-11-03, on the commit after `a7b0531` (C++ CCD noise, see the notes
below), load average about 1.2:

- AMD Ryzen 5 5600 (6 cores, 12 threads), 16 GB, Ubuntu 26.04.1 (Linux 7.0.0)
- Python 3.14.4, numpy 2.5.4 (scipy-openblas 0.3.34), scipy 1.18.1, astropy 8.0.2
- GCC 15.2.0; portable build (no `SPECULA_CPP_NATIVE`)
- C++ backend with 12 threads

`bench_objects.py`:

| Object (whole step, median) | numpy ms | C++ ms | Speed-up |
|---|---|---|---|
| Pyramid (24 modulation steps) | 94.28 | 5.35 | 17.6x |
| SH 20x20, NGS | 47.33 | 0.72 | 65.3x |
| SH 20x20, LGS kernel | 92.65 | 1.59 | 58.3x |
| PSF nd=8 | 40.63 | 1.08 | 37.7x |
| PSF nd=4 | 5.27 | 0.44 | 11.9x |
| Propagation, 4 layers x 4 sources | 23.07 | 0.65 | 35.4x |
| Atmosphere evolution, 4 layers | 1.12 | 0.62 | 1.8x |
| CCD 120x120 with noise | 0.75 | 0.19 | 4.0x |

`bench_scao.py`:

| Configuration | ms/step | Speed-up | Peak memory |
|---|---|---|---|
| numpy | 228.2 | 1.0x | 367 MB |
| C++ backend | 13.2 | 17.3x | 257 MB |
| C++ backend, `psf_dt` = 5 ms on the cameras | 10.6 | 21.4x | 257 MB |

| Object (ms/step) | numpy | C++ | C++ psf_dt |
|---|---|---|---|
| pyramid | 94.28 | 6.60 | 6.35 |
| sh | 47.61 | 0.70 | 0.71 |
| psf_ngs | 41.95 | 1.96 | 0.65 |
| prop | 23.31 | 0.69 | 0.72 |
| psf_sci1 | 6.90 | 0.60 | 0.27 |
| psf_sci2 | 5.68 | 0.59 | 0.25 |
| psf_sci3 | 5.65 | 0.58 | 0.25 |
| atmo | 1.16 | 0.69 | 0.69 |
| sh_ccd | 0.74 | 0.17 | 0.18 |
| pyr_ccd | 0.43 | 0.17 | 0.15 |

With `SPECULA_CPP_NATIVE=ON` (`-march=native`, AVX2), measured before the C++ CCD
noise, the C++ backend took 12.7 instead of 14.1 ms/step: pyramid 5.5 ms (3.8 ms in
`bench_objects.py`).

### Apple M1, macOS

Measured on 2026-11-03, commit `de3be45`, load average about 2. That build was
still optimized for size (`-Os`, nanobind's default) and SH used BLAS for its DFT
products: both were slower than with the current code, and these results should
be measured again.

- Apple M1 (4 performance + 4 efficiency cores), 8 GB, macOS 26.7.2
- Python 3.14.7, numpy 2.5.4 (Apple Accelerate BLAS), scipy 1.18.1, astropy 8.0.2
- Apple clang 21.0.0; build with scikit-build-core 1.1.2, nanobind 3.1.0, CMake 4.4.4, Ninja 1.13.2
- C++ backend with 8 threads

`bench_objects.py`:

| Object (whole step, median) | numpy ms | C++ ms | Speed-up |
|---|---|---|---|
| Pyramid (24 modulation steps) | 99.22 | 10.95 | 9.1x |
| SH 20x20, NGS | 48.44 | 0.93 | 52.1x |
| SH 20x20, LGS kernel | 99.16 | 2.72 | 36.5x |
| PSF nd=8 | 24.14 | 3.00 | 8.0x |
| PSF nd=4 | 5.02 | 0.93 | 5.4x |
| Propagation, 4 layers x 4 sources | 25.46 | 1.00 | 25.5x |
| Atmosphere evolution, 4 layers | 2.19 | 0.48 | 4.6x |
| CCD 120x120 with noise | 0.78 | 0.78 | 1.0x |

`bench_scao.py`:

| Configuration | ms/step | Speed-up | Peak memory |
|---|---|---|---|
| numpy | 216.8 | 1.0x | 405 MB |
| C++ backend | 21.6 | 10.0x | 300 MB |
| C++ backend, `psf_dt` = 5 ms on the cameras | 17.5 | 12.4x | 300 MB |

| Object (ms/step) | numpy | C++ | C++ psf_dt |
|---|---|---|---|
| pyramid | 99.10 | 11.48 | 11.59 |
| sh | 46.80 | 1.08 | 1.06 |
| prop | 25.67 | 0.95 | 0.94 |
| psf_ngs | 24.42 | 3.04 | 0.79 |
| psf_sci2 | 6.59 | 0.96 | 0.35 |
| psf_sci1 | 5.47 | 1.08 | 0.38 |
| psf_sci3 | 5.16 | 0.93 | 0.33 |
| atmo | 1.92 | 0.54 | 0.53 |
| sh_ccd | 0.83 | 0.83 | 0.84 |
| pyr_ccd | 0.43 | 0.42 | 0.39 |

## Approximations

Two parameters trade some accuracy for speed. Measured with `bench_approx.py` on the
Ryzen above, against the exact computation with the C++ backend:

| Approximation | Time | Error |
|---|---|---|
| `ModulatedPyramid` `mod_step: 20` instead of the default 24 (`mod_amp` = 3) | pyramid 4.86 -> 4.16 ms | signal 1.0% / 0.8% / 0.5% |
| `PSF` `psf_dt: 0.002` (steps of 1 ms) | simulation 14.1 -> 11.7 ms/step | integrated PSF 0.24%, SR exact |
| `PSF` `psf_dt: 0.005` | simulation 10.8 ms/step | integrated PSF 1.1-1.5%, SR exact |
| `PSF` `psf_dt: 0.010` | simulation 10.4 ms/step | integrated PSF 2.3-2.6%, SR exact |

- Pyramid: relative RMS change of the signal I(phase) - I(flat), for residual-like
  phases (Kolmogorov, piston and tip-tilt removed) of 50 / 150 / 300 nm RMS at 750 nm.
  `mod_step` must be at least 2 pi `mod_amp` (19 here); the time is proportional to it.
- PSF: relative RMS change of the integrated PSF of the four cameras of `scao_multi.yml`
  over 300 steps. This simulation is open loop (seeing-limited PSFs, SR below 2%):
  check the error in closed loop, where the PSF changes differently between steps.

(`bench_approx.py` times the simulation without trace skipping, so its times are a bit
longer than `bench_scao.py`'s.) With both (`mod_step: 20` and `psf_dt: 0.005` on the four
cameras), `bench_scao.py`'s simulation takes 9.8 ms/step instead of 13.2, 23 times
faster than numpy.

Other approximations were tried and rejected (pyramid errors at 50 / 150 / 300 nm):

| Approximation | Pyramid time | Error | Why not |
|---|---|---|---|
| Modulation points on the FFT pixel grid (a single forward FFT per frame) | about -45% (estimated) | 11.3% / 7.8% / 4.1% | too large for small residuals |
| Pupil sampled with 120 pixels instead of 160 (4 per subaperture) | 4.86 -> 2.99 ms | 5.9% / 5.6% / 5.5% | above 5%; 128 pixels is worse |
| `fft_res: 2.5` instead of 3 (minimum 2.42 here) | 6.64 ms | 3.4% / 3.7% / 4.4% | slower: FFT size 406 = 2 x 7 x 29 |
| `fov: 1.5` instead of 2 | 4.67 ms | 0.7% / 0.9% / 1.5% | the FFT size does not change |

Without approximation, building the backend for the CPU of the machine
(`pip install ./cpp -Ccmake.define.SPECULA_CPP_NATIVE=ON`) made the simulation about 10%
faster on the Ryzen (AVX2), but the module may not run on other CPUs.

## Notes for comparing results

- Timings vary by several percent between runs. The same benchmark with a load
  average of about 9 (background indexing) gave 23.9 ms/step for the C++ backend
  and 220.4 ms/step for numpy: compare runs made in similar conditions, and repeat
  a run if in doubt.
- The CCD photon and readout noise are drawn by the C++ backend, multithreaded, with
  the same distributions as numpy but not the same random numbers: the noisy pixels
  differ from a numpy run with the same seeds (the other results are the same up to
  rounding). They do not depend on the number of threads.
- In `bench_objects.py`, the CCD runs right after the other objects, which use all
  the cores: with numpy, on the Ryzen, its first few tens of milliseconds were up to
  40% slower, an effect of the CPU state that is not seen in the simulation.
- The C++ results are the same as the numpy ones up to floating point rounding, except
  the CCD noise (`test/test_cpp_backend.py`), and do not depend on the number of threads. The
  `psf_dt` configuration is an approximation (see the `PSF` documentation).
- The peak memory of `bench_objects.py` includes the setup of all the objects in
  one process, with temporary arrays; the `bench_scao.py` values are the meaningful
  ones for a simulation.
- On other machines, the speed-up depends mostly on the number of cores (the
  numpy path is mainly single threaded) and on the memory bandwidth.
- With OpenBLAS (numpy on Linux and Windows), any BLAS call in the loop leaves
  OpenBLAS threads spinning for a while, which takes the cores of the C++ thread
  pool: before the SH DFT products were moved to C++, this made the C++ backend
  more than twice as slow on the Ryzen. Objects that add BLAS calls to the loop
  (e.g. a reconstructor) can be checked by comparing with `OPENBLAS_THREAD_TIMEOUT=4`
  or `OPENBLAS_NUM_THREADS=1` in the environment.
