"""
Per-object CPU benchmark of SPECULA at SCAO scale, numpy vs the C++ backend.

Each processing object gets synthetic inputs and is timed over whole steps
(trigger + post_trigger, median of several steps after a warm-up step).
Each backend runs in its own process. See README.md.

Usage: python cpp/benchmarks/bench_objects.py
"""
import os
import resource
import subprocess
import sys
import tempfile
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 1_000_000  # 1 ms in SPECULA time units


def peak_memory_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == 'darwin' else rss / 2**10  # bytes on macOS, kB on Linux


def worker(backend):
    sys.path.insert(0, REPO_DIR)
    import specula
    specula.init(-1, precision=1, cpp_backend=(backend == 'cpp'))

    import numpy as np
    from specula.base_value import BaseValue
    from specula.data_objects.electric_field import ElectricField
    from specula.data_objects.intensity import Intensity
    from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.data_objects.source import Source
    from specula.lib.make_mask import make_mask
    from specula.processing_objects.atmo_evolution import AtmoEvolution
    from specula.processing_objects.atmo_propagation import AtmoPropagation
    from specula.processing_objects.ccd import CCD
    from specula.processing_objects.modulated_pyramid import ModulatedPyramid
    from specula.processing_objects.psf import PSF
    from specula.processing_objects.sh import SH

    rng = np.random.default_rng(1)

    def pupil_ef():
        # 160 px pupil (8 m), 10% central obstruction, 100 nm RMS random phase
        ef = ElectricField(160, 160, 0.05, S0=100.0)
        mask = make_mask(160, obsratio=0.1, xp=np)
        ef.A = mask
        ef.phaseInNm = rng.standard_normal((160, 160)) * 100 * mask
        return ef

    def step_time(obj, inputs, n_iter):
        t = DT
        for v in inputs:
            v.generation_time = t
        obj.setup()
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()
        times = []
        for _ in range(n_iter):
            t += DT
            for v in inputs:
                v.generation_time = t
            obj.check_ready(t)
            t0 = time.perf_counter()
            obj.trigger()
            obj.post_trigger()
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1e3

    def pyramid():
        pyr = ModulatedPyramid(SimulParams(160, 0.05), wavelengthInNm=750, fov=2.0, pup_diam=30,
                               output_resolution=80, mod_amp=3.0, pup_dist=36)
        ef = pupil_ef()
        pyr.inputs['in_ef'].set(ef)
        return step_time(pyr, [ef], 10)

    def sh(lgs, data_dir):
        obj = SH(wavelengthInNm=750, subap_wanted_fov=1.2, sensor_pxscale=0.15, subap_on_diameter=20,
                 subap_npx=6, laser_launch_tel=LaserLaunchTelescope(spot_size=1.0) if lgs else None,
                 data_dir=data_dir)
        ef = pupil_ef()
        obj.inputs['in_ef'].set(ef)
        return step_time(obj, [ef], 10)

    def psf(nd):
        obj = PSF(SimulParams(160, 0.05), wavelengthInNm=1650, nd=nd)
        ef = pupil_ef()
        obj.inputs['in_ef'].set(ef)
        return step_time(obj, [ef], 10)

    def ccd():
        obj = CCD(SimulParams(160, 0.05, time_step=0.001), size=[120, 120], dt=0.001, bandw=300,
                  photon_noise=True, readout_noise=True, readout_level=1.0, quantum_eff=0.32)
        i = Intensity(120, 120)
        i.i[:] = rng.random((120, 120)) * 1e3
        obj.inputs['in_i'].set(i)
        return step_time(obj, [i], 30)

    def atmosphere(data_dir):
        sp = SimulParams(160, 0.05, time_step=0.001)
        atmo = AtmoEvolution(sp, L0=25, heights=[0, 2000, 6000, 12000], Cn2=[0.25] * 4, fov=60.0,
                             pixel_phasescreens=1024, data_dir=data_dir)
        values = [BaseValue(value=np.array([0.8], dtype=np.float32)),
                  BaseValue(value=np.array([10, 15, 20, 25], dtype=np.float32)),
                  BaseValue(value=np.array([0, 45, 90, 135], dtype=np.float32))]
        for name, v in zip(('seeing', 'wind_speed', 'wind_direction'), values):
            atmo.inputs[name].set(v)
        t_atmo = step_time(atmo, values, 10)

        # 1 on-axis and 3 off-axis sources (10 arcsec), through the 4 layers and a pupil stop
        sources = {f's{k}': Source(polar_coordinates=[0.0 if k == 0 else 10.0, 120.0 * k],
                                   magnitude=8, wavelengthInNm=750) for k in range(4)}
        prop = AtmoPropagation(sp, source_dict=sources)
        layers = atmo.outputs['layer_list']
        pupilstop = Pupilstop(sp, obs_diam=0.1)
        prop.inputs['atmo_layer_list'].set(layers)
        prop.inputs['common_layer_list'].set([pupilstop])
        return t_atmo, step_time(prop, layers + [pupilstop], 10)

    with tempfile.TemporaryDirectory() as data_dir:
        t_atmo, t_prop = atmosphere(data_dir)
        results = [
            ('Pyramid (24 modulation steps)', pyramid()),
            ('SH 20x20, NGS', sh(False, data_dir)),
            ('SH 20x20, LGS kernel', sh(True, data_dir)),
            ('PSF nd=8', psf(8)),
            ('PSF nd=4', psf(4)),
            ('Propagation, 4 layers x 4 sources', t_prop),
            ('Atmosphere evolution, 4 layers', t_atmo),
            ('CCD 120x120 with noise', ccd()),
        ]
    for name, ms in results:
        print(f'{name}\t{ms:.3f}')
    threads = specula.cpp.get_num_threads() if specula.cpp is not None else 1
    print(f'threads\t{threads}')
    print(f'peak_mb\t{peak_memory_mb():.0f}')


def run(backend):
    env = dict(os.environ, SPECULA_DISABLE_GPU='TRUE')
    env.pop('SPECULA_DISABLE_CPP', None)
    out = subprocess.run([sys.executable, __file__, '--worker', backend], env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'{backend} run failed:\n{out.stderr}')
    return dict(line.split('\t') for line in out.stdout.splitlines() if '\t' in line)


def main():
    numpy_res, cpp_res = run('numpy'), run('cpp')
    print(f'{"Object (whole step, median)":36s} {"numpy ms":>9s} {"C++ ms":>8s} {"speed-up":>9s}')
    for name in numpy_res:
        if name in ('threads', 'peak_mb'):
            continue
        a, b = float(numpy_res[name]), float(cpp_res[name])
        print(f'{name:36s} {a:9.2f} {b:8.2f} {a / b:8.1f}x')
    print(f'Peak memory: numpy {numpy_res["peak_mb"]} MB, C++ {cpp_res["peak_mb"]} MB;'
          f' C++ threads: {cpp_res["threads"]}')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
