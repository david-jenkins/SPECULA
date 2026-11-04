import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
from unittest import mock

from specula import np

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.lib.affine_transform import affine_transform
from specula.lib.calc_psf import calc_psf
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.interp2d import Interp2D
from specula.lib.make_mask import make_mask
from specula.lib.toccd import toccd
from specula.data_objects.intensity import Intensity
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.ccd import CCD
from specula.processing_objects import modulated_pyramid
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.psf import PSF
from specula.processing_objects import sh
from specula.processing_objects.sh import SH

from test.specula_testlib import cpp_backend, cpp_available


def numpy_and_cpp(f):
    '''Return f() computed without and with the C++ backend'''
    with cpp_backend(False):
        ref = f()
    with cpp_backend(True):
        new = f()
    return ref, new


def pupil_field(n, pitch, rms_nm, seed, precision=1, S0=100.0):
    ef = ElectricField(n, n, pitch, S0=S0, target_device_idx=-1, precision=precision)
    mask = make_mask(n, obsratio=0.1, xp=np)
    rng = np.random.default_rng(seed)
    ef.A = mask
    ef.phaseInNm = rng.standard_normal((n, n)) * rms_nm * mask
    return ef


def run_steps(obj, inputs, n_steps):
    '''setup() and n_steps loop iterations, 1 ms apart'''
    t = 0
    obj.setup()
    for _ in range(n_steps):
        t += 1_000_000
        for v in inputs:
            v.generation_time = t
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()


@unittest.skipUnless(cpp_available(), 'specula_cpp is not installed')
class TestCppBackend(unittest.TestCase):
    '''
    Results with the C++ CPU backend must be the same as with numpy,
    up to floating point rounding.
    '''

    def assert_close(self, new, ref, rtol):
        new, ref = np.asarray(new), np.asarray(ref)
        self.assertEqual(new.shape, ref.shape)
        self.assertEqual(new.dtype, ref.dtype)
        err = np.max(np.abs(new - ref)) / max(np.max(np.abs(ref)), np.finfo(ref.dtype).tiny)
        self.assertLessEqual(err, rtol)

    def test_backend_selection(self):
        with cpp_backend(True):
            self.assertIsNotNone(ElectricField(4, 4, 0.1, target_device_idx=-1).cpp)
        with cpp_backend(False):
            self.assertIsNone(ElectricField(4, 4, 0.1, target_device_idx=-1).cpp)

    def test_blas_threads(self):
        # One BLAS thread while the backend is enabled (busy-waiting BLAS threads slow it down),
        # unless the number of BLAS threads is set in the environment
        import iteras
        import threadpoolctl

        def blas_threads():
            return {p['num_threads'] for p in threadpoolctl.threadpool_info() if p['user_api'] == 'blas'}

        env = {k: v for k, v in os.environ.items() if k not in iteras._BLAS_THREADS_ENV}
        with cpp_backend(False):
            original = blas_threads()
            with mock.patch.dict(os.environ, env, clear=True):
                with cpp_backend(True):
                    self.assertEqual(blas_threads(), {1})
                self.assertEqual(blas_threads(), original)
                with mock.patch.dict(os.environ, {'OPENBLAS_NUM_THREADS': '4'}), cpp_backend(True):
                    self.assertEqual(blas_threads(), original)

    def test_interp2d(self):
        rng = np.random.default_rng(1)
        for dtype, rtol in ((np.float32, 1e-6), (np.float64, 1e-12)):
            data = rng.standard_normal((200, 200)).astype(dtype)

            def f():
                interp = Interp2D((200, 200), (160, 170), rotInDeg=7.0, rowShiftInPixels=3.3,
                                  colShiftInPixels=-2.1, magnification=1.1, dtype=dtype, xp=np)
                amp = np.ones((160, 170), dtype=dtype) * 2
                phase = np.ones((160, 170), dtype=dtype)
                interp.interpolate_accumulate(np.abs(data), data, amp, phase)
                return interp.interpolate(data), amp, phase

            for r, n in zip(*numpy_and_cpp(f)):
                self.assert_close(n, r, rtol)

    def test_toccd(self):
        rng = np.random.default_rng(2)
        for dtype, rtol in ((np.float32, 1e-6), (np.float64, 1e-12)):
            for inshape, outshape in (((480, 480), (90, 90)), ((37, 37), (11, 11)), ((64, 48), (24, 20))):
                a = rng.random(inshape).astype(dtype)
                for set_total in (None, 0, 1000.0):
                    ref, new = numpy_and_cpp(lambda: toccd(a, outshape, set_total=set_total, xp=np))
                    if set_total == 0:
                        # Not normalized: compare the shape only
                        ref, new = ref / ref.sum(), new / new.sum()
                    self.assert_close(new, ref, rtol)

    def test_ef_at_lambda(self):
        def f():
            ef = pupil_field(64, 0.1, 300, seed=3)
            out = np.zeros((16, 64), dtype=np.complex64)
            ef.ef_at_lambda(750, slicey=np.s_[8:24], slicex=np.s_[:], out=out)
            return ef.ef_at_lambda(750), out, ef.ef_at_lambda(500, slicex=np.s_[::2])
        for r, n in zip(*numpy_and_cpp(f)):
            self.assert_close(n, r, 1e-6)

    def test_affine_transform(self):
        rng = np.random.default_rng(4)
        data = rng.standard_normal((100, 120)).astype(np.float32)
        a = np.radians(30)
        # Rotation about the center, partially outside the data
        matrix = np.array([[np.cos(a), np.sin(a), 10.0], [-np.sin(a), np.cos(a), 40.0]])

        def f():
            out = np.empty((90, 110), dtype=np.float32)
            affine_transform(data, matrix, out, xp=np)
            return out
        ref, new = numpy_and_cpp(f)
        self.assert_close(new, ref, 1e-6)
        np.testing.assert_array_equal(new == 0, ref == 0)

    def test_ef_interpolator(self):
        def f():
            ef = pupil_field(160, 0.05, 200, seed=5)
            interp = EFInterpolator(ef, (320, 320), rotAnglePhInDeg=3.0, xShiftPhInPixel=0.3,
                                    yShiftPhInPixel=-0.2, target_device_idx=-1, precision=1)
            interp.interpolate()
            out = interp.interpolated_ef()
            return out.A.copy(), out.phaseInNm.copy()
        for r, n in zip(*numpy_and_cpp(f)):
            self.assert_close(n, r, 1e-6)

    def test_calc_psf(self):
        ef = pupil_field(160, 0.05, 100, seed=6)
        phase = ef.phi_at_lambda(1650)
        for kwargs in ({'imwidth': 640, 'normalize': True, 'return_total': True},
                       {'imwidth': 640, 'nocenter': True},
                       {}):
            ref, new = numpy_and_cpp(lambda: calc_psf(phase, ef.A, xp=np, complex_dtype=np.complex64,
                                                       **kwargs))
            if kwargs.get('return_total'):
                self.assertAlmostEqual(float(new[1]) / float(ref[1]), 1.0, places=5)
                ref, new = ref[0], new[0]
            self.assert_close(new, ref, 1e-5)

    def test_psf_object(self):
        def f():
            ef = pupil_field(160, 0.05, 100, seed=7)
            psf = PSF(SimulParams(160, 0.05), wavelengthInNm=1650, nd=4, target_device_idx=-1, precision=1)
            psf.inputs['in_ef'].set(ef)
            psf.setup()
            # A PSF moved by a random tilt at each step, otherwise std_psf is just rounding noise
            rng = np.random.default_rng(11)
            yy, xx = np.mgrid[0:160, 0:160]
            for t in range(1_000_000, 4_000_000, 1_000_000):
                tilt = rng.uniform(-2, 2, 2) * 1650 / 160
                ef.phaseInNm = (tilt[0] * xx + tilt[1] * yy) * ef.A
                ef.generation_time = t
                psf.check_ready(t)
                psf.trigger()
                psf.post_trigger()
            psf.finalize()
            return psf.psf.value, psf.int_psf.value, np.float32(psf.sr.value), psf.std_psf.value
        ref, new = numpy_and_cpp(f)
        for r, n in zip(ref[:3], new[:3]):
            self.assert_close(n, r, 1e-5)
        # The variance is computed as E[x^2] - E[x]^2, which amplifies rounding differences
        self.assert_close(new[3], ref[3], 1e-3)

    def run_pyramid(self, n_steps=2, **kwargs):
        def f():
            ef = pupil_field(160, 0.05, 50, seed=8)
            pyr = ModulatedPyramid(SimulParams(160, 0.05), wavelengthInNm=750, fov=2.0, pup_diam=30,
                                   output_resolution=80, mod_amp=3.0, pup_dist=36,
                                   target_device_idx=-1, precision=1, **kwargs)
            pyr.inputs['in_ef'].set(ef)
            run_steps(pyr, [ef], n_steps)
            self.assertEqual(pyr._cpp_nc is not None, specula.cpp is not None)
            return (pyr.outputs['out_i'].i, pyr.outputs['out_psf_tot'].value,
                    pyr.outputs['out_psf_bfm'].value, pyr.outputs['out_transmission'].value)
        for r, n in zip(*numpy_and_cpp(f)):
            self.assert_close(n, r, 1e-5)

    def test_pyramid(self):
        self.run_pyramid()

    def test_pyramid_chunks(self):
        # Several chunks of modulation steps, the last one partial
        with mock.patch.object(modulated_pyramid, '_CPP_CHUNK_BYTES', 5 * 480 * 480 * 8):
            self.run_pyramid()

    def test_pyramid_alternating(self):
        self.run_pyramid(n_steps=3, mod_type='alternating')

    def run_sh(self, laser_launch_tel=None, subap_rows_slice=None, pupil=(160, 0.05), **kwargs):
        args = dict(wavelengthInNm=750, subap_wanted_fov=1.2, sensor_pxscale=0.15,
                    subap_on_diameter=20, subap_npx=6)
        args.update(kwargs)

        def f():
            ef = pupil_field(*pupil, 50, seed=9)
            with tempfile.TemporaryDirectory() as data_dir:
                shobj = SH(laser_launch_tel=laser_launch_tel, subap_rows_slice=subap_rows_slice,
                           data_dir=data_dir, target_device_idx=-1, precision=1, **args)
                shobj.inputs['in_ef'].set(ef)
                run_steps(shobj, [ef], 2)
            return shobj.outputs['out_i'].i.copy(), shobj._cpp_rows
        (ref, ref_rows), (new, new_rows) = numpy_and_cpp(f)
        # The C++ path is used only with the backend
        self.assertIsNone(ref_rows)
        self.assertIsNotNone(new_rows)
        self.assert_close(new, ref, 1e-5)
        return new_rows

    def test_sh(self):
        self.run_sh()

    def test_sh_rows_slice(self):
        self.run_sh(subap_rows_slice=slice(3, 17))

    def test_sh_lgs_kernel(self):
        self.run_sh(laser_launch_tel=LaserLaunchTelescope(spot_size=1.0))

    def test_sh_lgs_kernel_rows(self):
        self.run_sh(laser_launch_tel=LaserLaunchTelescope(spot_size=1.0), subap_rows_slice=slice(2, 15))

    def test_sh_lgs_kernel_refined(self):
        # A narrow kernel refines the FFT grid (kernel_sampling_factor()), so N is much
        # larger than 2n (192 vs 48 here)
        self.run_sh(laser_launch_tel=LaserLaunchTelescope(spot_size=0.5), pupil=(120, 0.008333),
                    subap_on_diameter=10, subap_npx=23, sensor_pxscale=0.928, subap_wanted_fov=9.28)

    def test_sh_chunks(self):
        # Several chunks of subaperture rows, the last one partial
        with mock.patch.object(sh, '_CPP_CHUNK_BYTES', 1):
            self.assertEqual(self.run_sh(subap_rows_slice=slice(3, 17)), 1)
        # About 604 kB per row of 20 subapertures (n = 16): 3 rows per chunk, 20 rows
        with mock.patch.object(sh, '_CPP_CHUNK_BYTES', 2_000_000):
            self.assertEqual(self.run_sh(laser_launch_tel=LaserLaunchTelescope(spot_size=1.0)), 3)

    def test_atmo_propagation(self):
        def f():
            simul_params = SimulParams(160, 0.05)
            rng = np.random.default_rng(10)
            layer = Layer(200, 200, 0.05, height=5000, target_device_idx=-1, precision=1)
            layer.phaseInNm = rng.standard_normal((200, 200)) * 100
            pupilstop = Pupilstop(simul_params, obs_diam=0.1, target_device_idx=-1, precision=1)
            sources = {'off': Source(polar_coordinates=[30.0, 45.0], magnitude=8, wavelengthInNm=750),
                       'on': Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)}
            prop = AtmoPropagation(simul_params, source_dict=sources, target_device_idx=-1, precision=1)
            prop.inputs['atmo_layer_list'].set([layer])
            prop.inputs['common_layer_list'].set([pupilstop])
            run_steps(prop, [layer, pupilstop], 1)
            return [prop.outputs[f'out_{name}_ef'].field.copy() for name in sources]
        for r, n in zip(*numpy_and_cpp(f)):
            self.assert_close(n, r, 1e-6)

    # CCD noise: same distributions as numpy, not the same random numbers

    def test_poisson_noise(self):
        import specula_cpp as cpp
        n = 100_000
        for lam in (0.3, 4.0, 10.0, 37.5, 1e4):  # both algorithms (multiplication, PTRS)
            a = np.full(n, lam)
            cpp.poisson_noise(a, 11, 0)
            np.testing.assert_array_equal(a, np.round(a))
            # Mean and variance within 5 standard errors
            self.assertLess(abs(a.mean() - lam), 5 * np.sqrt(lam / n))
            self.assertLess(abs(a.var() / lam - 1), 5 * np.sqrt(2 / n) + 5 / np.sqrt(lam * n))
        zero = np.zeros(10, dtype=np.float32)
        cpp.poisson_noise(zero, 1, 0)
        np.testing.assert_array_equal(zero, 0)
        with self.assertRaises(ValueError):
            cpp.poisson_noise(np.array([1.0, -1.0]), 1, 0)

    def test_gaussian_noise(self):
        import specula_cpp as cpp
        n = 100_000
        g = np.full(n, 3.0)
        cpp.gaussian_noise(g, 2.0, 5, 0)
        self.assertLess(abs(g.mean() - 3.0), 5 * 2.0 / np.sqrt(n))
        self.assertLess(abs(g.std() / 2.0 - 1), 5 * np.sqrt(0.5 / n))
        # Fourth moment of a Gaussian: 3 sigma^4
        self.assertLess(abs(np.mean((g - 3.0) ** 4) / 16 - 3), 0.1)

    def test_noise_streams(self):
        import specula_cpp as cpp
        threads = cpp.get_num_threads()
        draws = []
        for n_threads, seed, stream in ((threads, 3, 4), (1, 3, 4), (threads, 3, 5), (threads, 4, 4)):
            cpp.set_num_threads(n_threads)
            a = np.full(5000, 50.0, dtype=np.float32)
            cpp.poisson_noise(a, seed, stream)
            draws.append(a)
        cpp.set_num_threads(threads)
        np.testing.assert_array_equal(draws[0], draws[1])  # any number of threads
        for other in draws[2:]:  # another stream or seed
            self.assertLess(abs(np.corrcoef(draws[0], other)[0, 1]), 0.1)

    def test_ccd_noise(self):
        def f():
            ccd = CCD(SimulParams(time_step=1), size=(64, 64), dt=1, bandw=1, photon_noise=True,
                      readout_noise=True, readout_level=3.0, target_device_idx=-1, precision=1)
            i = Intensity(64, 64, target_device_idx=-1, precision=1)
            i.i[:] = 200.0
            ccd.inputs['in_i'].set(i)
            frames = []
            for t in range(1, 4):
                i.generation_time = t * 1_000_000_000
                ccd.check_ready(i.generation_time)
                ccd.trigger()
                ccd.post_trigger()
                frames.append(ccd.outputs['out_pixels'].pixels.copy())
            self.assertEqual(ccd._noise_cpp(ccd._pixels.pixels), specula.cpp is not None)
            return np.array(frames)
        ref, new = numpy_and_cpp(f)
        # Photon (variance 200) and readout (3**2) noise: same statistics, other numbers
        for frames in (ref, new):
            self.assertLess(abs(frames.mean() - 200), 0.5)
            self.assertLess(abs(frames.var() / 209 - 1), 0.05)
        self.assertFalse(np.array_equal(ref, new))
        self.assertFalse(np.array_equal(new[0], new[1]))  # each frame draws new noise


if __name__ == '__main__':
    unittest.main()
