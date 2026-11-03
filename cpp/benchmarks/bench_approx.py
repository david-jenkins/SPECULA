"""
Error of the approximations that speed up the CPU simulation, against the
exact computation (both with the C++ backend if installed). See README.md.

- ModulatedPyramid with fewer modulation steps (mod_step): relative RMS change
  of the signal I(phase) - I(flat) of the pyramid in bench_objects.py, for
  residual-like phases (Kolmogorov, piston and tip-tilt removed) of 50, 150
  and 300 nm RMS;
- PSF psf_dt: relative RMS change of the integrated PSF of the four cameras of
  scao_multi.yml, and change of their integrated SR, over 300 steps.

Usage: python cpp/benchmarks/bench_approx.py
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_DIR)
os.environ.setdefault('SPECULA_DISABLE_GPU', 'TRUE')

import specula
specula.init(-1, precision=1)

import numpy as np
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.simul import Simul

N = 160  # pupil pixels (8 m)
PSF_STEPS = 300
CAMS = ('psf_ngs', 'psf_sci1', 'psf_sci2', 'psf_sci3')


def residual_phase(rms, seed=3):
    rng = np.random.default_rng(seed)
    m = 1024
    f = np.fft.fftfreq(m)
    fr = np.hypot(*np.meshgrid(f, f))
    fr[0, 0] = 1
    noise = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    screen = np.fft.ifft2(noise * fr ** (-11 / 6)).real[:N, :N]
    mask = make_mask(N, obsratio=0.1, xp=np) > 0
    y, x = np.mgrid[0:N, 0:N] - (N - 1) / 2
    a = np.c_[np.ones(mask.sum()), x[mask], y[mask]]
    screen[mask] -= a @ np.linalg.lstsq(a, screen[mask], rcond=None)[0]
    screen *= rms / screen[mask].std()
    return (screen * mask).astype(np.float32)


def pyramid_output(phase, mod_step, n_timed=0):
    pyr = ModulatedPyramid(SimulParams(N, 0.05), wavelengthInNm=750, fov=2.0, pup_diam=30,
                           output_resolution=80, mod_amp=3.0, pup_dist=36, mod_step=mod_step)
    ef = ElectricField(N, N, 0.05, S0=100.0)
    ef.A = make_mask(N, obsratio=0.1, xp=np)
    ef.phaseInNm = phase
    pyr.inputs['in_ef'].set(ef)
    ef.generation_time = 1
    pyr.setup()
    pyr.check_ready(1)
    pyr.trigger()
    pyr.post_trigger()
    out = pyr.outputs['out_i'].i.copy()
    times = []
    for i in range(n_timed):
        ef.generation_time = 2 + i
        pyr.check_ready(2 + i)
        t0 = time.perf_counter()
        pyr.trigger()
        times.append(time.perf_counter() - t0)
    return out, (np.median(times) * 1e3 if times else None)


def pyramid_errors():
    flat = np.zeros((N, N), dtype=np.float32)
    phases = {rms: residual_phase(rms) for rms in (50, 150, 300)}
    ref = None
    print(f'{"Pyramid mod_step":16s} {"trigger ms":>10s}  signal error at 50 / 150 / 300 nm RMS')
    for mod_step in (24, 20):
        i0, ms = pyramid_output(flat, mod_step, n_timed=15)
        signal = {rms: pyramid_output(ph, mod_step)[0] - i0 for rms, ph in phases.items()}
        ref = ref or signal
        errs = [np.linalg.norm(signal[r] - ref[r]) / np.linalg.norm(ref[r]) for r in phases]
        print(f'{mod_step:16d} {ms:10.2f}  ' + ' / '.join(f'{100 * e:.2f}%' for e in errs))


def run_scao(psf_dt):
    overrides = f'{{main.total_time: {PSF_STEPS * 0.001}'
    overrides += ''.join(f', {c}.psf_dt: {psf_dt}' for c in CAMS if psf_dt) + '}'
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)  # root_dir of the scenario (phase screens)
        try:
            os.mkdir('calib')
            simul = Simul(os.path.join(HERE, 'scao_multi.yml'), overrides=overrides,
                          speed_report=False, log_level='WARNING')
            t0 = time.perf_counter()
            simul.run()
            ms = (time.perf_counter() - t0) / PSF_STEPS * 1e3
        finally:
            os.chdir(cwd)
    return {c: (simul.objs[c].int_psf.value.copy(), float(simul.objs[c].int_sr.value)) for c in CAMS}, ms


def psf_errors():
    ref, ms = run_scao(None)
    print(f'\n{"PSF psf_dt":16s} {"ms/step":>10s}  integrated PSF error / integrated SR change ({", ".join(CAMS)})')
    print(f'{"exact":16s} {ms:10.1f}')
    for psf_dt in (0.002, 0.005, 0.010):
        res, ms = run_scao(psf_dt)
        errs = [np.linalg.norm(res[c][0] - ref[c][0]) / np.linalg.norm(ref[c][0]) for c in CAMS]
        sr = [res[c][1] / ref[c][1] - 1 for c in CAMS]
        print(f'{f"{psf_dt * 1e3:.0f} ms":16s} {ms:10.1f}  ' +
              ' '.join(f'{100 * e:.2f}%' for e in errs) + '  /  ' + ' '.join(f'{100 * s:+.2f}%' for s in sr))


if __name__ == '__main__':
    pyramid_errors()
    psf_errors()
