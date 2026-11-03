# specula-cpp

Optional C++ CPU backend for [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA).

When this package is installed, SPECULA objects running on the CPU
(`target_device_idx=-1`) use multithreaded C++ kernels for interpolation,
rebinning, electric field computation and the per-step work of the PSF,
pyramid and Shack-Hartmann objects, instead of the numpy/scipy code.
Results are the same up to floating point rounding.

The pyramid uses the FFTs of [pocketfft](https://github.com/mreineck/pocketfft)
(BSD 3-clause license), included unchanged in `third_party/pocketfft`.

## Installation

From the root of the SPECULA repository:

```bash
pip install ./cpp
```

A C++17 compiler is needed. CMake, nanobind and scikit-build-core are
installed automatically during the build. To optimize for the CPU of the
build machine (the result may not run on other CPUs):

```bash
pip install ./cpp -Ccmake.define.SPECULA_CPP_NATIVE=ON
```

## Usage

The backend is enabled automatically when the module can be imported.
It can be disabled with the `SPECULA_DISABLE_CPP` environment variable,
with `specula.init(..., cpp_backend=False)`, or with `specula --no-cpp`.

The number of threads defaults to the CPUs available to the process and can
be set with the `SPECULA_CPP_THREADS` environment variable or with
`specula_cpp.set_num_threads()`.
