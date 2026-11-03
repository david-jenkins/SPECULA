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

The number of threads of the C++ backend defaults to the CPUs available to the
process; set `SPECULA_CPP_THREADS` to change it. All the computations are in
single precision (`precision=1`), as in the default `specula` command line.

## Reference results

### AMD Ryzen 5 5600, Linux

Measured on 2026-11-03, on the commit after `d0e7bdd` that builds the backend
with `-O3` (nanobind's `NOMINSIZE`) and computes the SH DFT products without BLAS,
load average about 0.7:

- AMD Ryzen 5 5600 (6 cores, 12 threads), 16 GB, Ubuntu 26.04.1 (Linux 7.0.0)
- Python 3.14.4, numpy 2.5.4 (scipy-openblas 0.3.34), scipy 1.18.1, astropy 8.0.2
- GCC 15.2.0; portable build (no `SPECULA_CPP_NATIVE`)
- C++ backend with 12 threads

`bench_objects.py`:

| Object (whole step, median) | numpy ms | C++ ms | Speed-up |
|---|---|---|---|
| Pyramid (24 modulation steps) | 94.01 | 8.95 | 10.5x |
| SH 20x20, NGS | 47.15 | 0.91 | 51.7x |
| SH 20x20, LGS kernel | 93.19 | 1.80 | 51.7x |
| PSF nd=8 | 39.74 | 2.65 | 15.0x |
| PSF nd=4 | 5.29 | 0.64 | 8.3x |
| Propagation, 4 layers x 4 sources | 23.14 | 1.51 | 15.3x |
| Atmosphere evolution, 4 layers | 1.11 | 0.67 | 1.7x |
| CCD 120x120 with noise | 0.75 | 0.75 | 1.0x |

`bench_scao.py`:

| Configuration | ms/step | Speed-up | Peak memory |
|---|---|---|---|
| numpy | 228.3 | 1.0x | 366 MB |
| C++ backend | 22.1 | 10.3x | 263 MB |
| C++ backend, `psf_dt` = 5 ms on the cameras | 17.1 | 13.4x | 265 MB |

| Object (ms/step) | numpy | C++ | C++ psf_dt |
|---|---|---|---|
| pyramid | 94.31 | 10.36 | 10.21 |
| sh | 47.70 | 0.88 | 0.86 |
| psf_ngs | 42.19 | 4.10 | 1.16 |
| prop | 23.30 | 1.65 | 1.63 |
| psf_sci1 | 6.82 | 1.02 | 0.35 |
| psf_sci2 | 5.62 | 0.86 | 0.29 |
| psf_sci3 | 5.58 | 0.86 | 0.29 |
| atmo | 1.17 | 0.78 | 0.75 |
| sh_ccd | 0.72 | 0.74 | 0.74 |
| pyr_ccd | 0.43 | 0.42 | 0.38 |

With `SPECULA_CPP_NATIVE=ON` (`-march=native`, AVX2) the C++ backend takes
18.3 ms/step: pyramid 7.8 ms, propagation 1.0 ms.

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

## Notes for comparing results

- Timings vary by several percent between runs. The same benchmark with a load
  average of about 9 (background indexing) gave 23.9 ms/step for the C++ backend
  and 220.4 ms/step for numpy: compare runs made in similar conditions, and repeat
  a run if in doubt.
- The CCD is not accelerated: its time is the random number generation of numpy,
  kept so that the noise is the same with both backends.
- The C++ results are the same as the numpy ones up to floating point rounding
  (`test/test_cpp_backend.py`), and do not depend on the number of threads. The
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
