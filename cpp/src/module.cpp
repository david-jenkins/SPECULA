// Python bindings of the SPECULA C++ CPU backend (module specula_cpp).
//
// All array arguments must already have the right dtype and layout: they are
// never converted, so that outputs are always written in place. The GIL is
// released while the kernels run.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#include <process.h>
#define SPECULA_GETPID _getpid
#else
#include <unistd.h>
#define SPECULA_GETPID getpid
#endif
#ifdef __linux__
#include <sched.h>
#endif

#include "kernels.hpp"
#include "pyramid.hpp"

namespace nb = nanobind;
using namespace specula_cpp;

namespace specula_cpp {

static int default_threads() {
    if (const char *env = std::getenv("SPECULA_CPP_THREADS")) {
        int n = std::atoi(env);
        if (n > 0)
            return n;
    }
#ifdef __linux__
    // Respect CPU sets (e.g. batch schedulers), unlike hardware_concurrency()
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        return std::max(1, CPU_COUNT(&set));
#endif
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

static ThreadPool *g_pool = nullptr;
static long g_pool_pid = 0;
static int g_requested_threads = 0;

// The pool is never destroyed: joining threads at interpreter exit is fragile.
// After a fork() the child has no workers, so it gets a new pool.
ThreadPool &pool() {
    long pid = static_cast<long>(SPECULA_GETPID());
    if (g_pool == nullptr || pid != g_pool_pid) {
        g_pool = new ThreadPool(g_requested_threads > 0 ? g_requested_threads : default_threads());
        g_pool_pid = pid;
    }
    return *g_pool;
}

}  // namespace specula_cpp

namespace {

// Read-only inputs and in-place outputs, C-contiguous
template <class T>
using in_array = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <class T>
using out_array = nb::ndarray<T, nb::c_contig, nb::device::cpu>;

// Same, any strides
template <class T>
using in_strided = nb::ndarray<const T, nb::device::cpu>;
template <class T>
using out_strided = nb::ndarray<T, nb::device::cpu>;

template <class T>
using cplx = std::complex<T>;

void require(bool cond, const std::string &msg) {
    if (!cond)
        throw std::invalid_argument(msg);
}

template <class A>
idx dim(const A &a, size_t axis) {
    return static_cast<idx>(a.shape(axis));
}

template <class A>
idx count(const A &a) {
    return static_cast<idx>(a.size());
}

template <class A>
void require_shape(const A &a, std::initializer_list<idx> shape, const char *name) {
    bool ok = a.ndim() == shape.size();
    size_t i = 0;
    for (idx s : shape) {
        if (!ok)
            break;
        ok = dim(a, i++) == s;
    }
    if (!ok) {
        std::string msg = std::string(name) + " has shape (";
        for (size_t d = 0; d < a.ndim(); ++d)
            msg += std::to_string(a.shape(d)) + (d + 1 < a.ndim() ? ", " : "");
        msg += "), expected (";
        size_t j = 0;
        for (idx s : shape)
            msg += std::to_string(s) + (++j < shape.size() ? ", " : "");
        throw std::invalid_argument(msg + ")");
    }
}

// ---------------------------------------------------------------------------

template <class T>
void py_interp2d(in_array<T> in, in_array<T> xx, in_array<T> yy, out_array<T> out) {
    require(in.ndim() == 2 && out.ndim() == 2, "input and output must be 2D");
    idx n_out = count(out);
    require(count(xx) == n_out && count(yy) == n_out, "xx and yy must have one element per output pixel");
    const T *pin = in.data(), *px = xx.data(), *py = yy.data();
    T *pout = out.data();
    idx ny = dim(in, 0), nx = dim(in, 1);
    nb::gil_scoped_release release;
    interp2d(pin, ny, nx, px, py, pout, n_out);
}

template <class T>
void py_interp2d_accumulate(in_array<T> amp, in_array<T> phase, in_array<T> xx, in_array<T> yy,
                            out_array<T> amp_out, out_array<T> phase_out) {
    require(amp.ndim() == 2, "amp must be 2D");
    require_shape(phase, {dim(amp, 0), dim(amp, 1)}, "phase");
    require(phase_out.ndim() == 2, "phase_out must be 2D");
    require_shape(amp_out, {dim(phase_out, 0), dim(phase_out, 1)}, "amp_out");
    idx n_out = count(phase_out);
    require(count(xx) == n_out && count(yy) == n_out, "xx and yy must have one element per output pixel");
    const T *pa = amp.data(), *pp = phase.data(), *px = xx.data(), *py = yy.data();
    T *pao = amp_out.data(), *ppo = phase_out.data();
    idx ny = dim(amp, 0), nx = dim(amp, 1);
    nb::gil_scoped_release release;
    interp2d_accumulate(pa, pp, ny, nx, px, py, pao, ppo, n_out);
}

template <class T>
void py_affine_bilinear(in_array<T> data, in_array<double> matrix, out_array<T> out) {
    require(data.ndim() == 2 && out.ndim() == 2, "data and output must be 2D");
    require_shape(matrix, {2, 3}, "matrix");
    const T *pd = data.data();
    const double *pm = matrix.data();
    T *po = out.data();
    idx ny = dim(data, 0), nx = dim(data, 1), oy = dim(out, 0), ox = dim(out, 1);
    nb::gil_scoped_release release;
    affine_bilinear(pd, ny, nx, pm, po, oy, ox);
}

template <class T>
double py_toccd(in_array<T> in, out_array<T> out) {
    require(in.ndim() == 2 && out.ndim() == 2, "input and output must be 2D");
    require(count(in) > 0 && count(out) > 0, "empty array");
    const T *pi = in.data();
    T *po = out.data();
    idx iny = dim(in, 0), inx = dim(in, 1), outy = dim(out, 0), outx = dim(out, 1);
    nb::gil_scoped_release release;
    return toccd(pi, iny, inx, po, outy, outx);
}

template <class T>
void py_ef_at_lambda(in_strided<T> amp, in_strided<T> phase, T k, out_strided<cplx<T>> out) {
    require(amp.ndim() == 2, "amp must be 2D");
    require_shape(phase, {dim(amp, 0), dim(amp, 1)}, "phase");
    require_shape(out, {dim(amp, 0), dim(amp, 1)}, "out");
    const T *pa = amp.data(), *pp = phase.data();
    cplx<T> *po = out.data();
    idx as0 = amp.stride(0), as1 = amp.stride(1);
    idx ps0 = phase.stride(0), ps1 = phase.stride(1);
    idx os0 = out.stride(0), os1 = out.stride(1);
    idx ny = dim(amp, 0), nx = dim(amp, 1);
    nb::gil_scoped_release release;
    ef_at_lambda(pa, as0, as1, pp, ps0, ps1, ny, nx, k, po, os0, os1);
}

template <class T>
void py_masked_extrapolation(in_array<T> phase, in_array<T> amp, T threshold,
                             in_array<std::int32_t> edge_pixels, in_array<std::int32_t> ref_idx,
                             in_array<float> coeffs, in_array<std::int64_t> valid, out_array<T> out) {
    idx n = count(phase);
    require(count(amp) == n && count(out) == n, "phase, amp and out must have the same size");
    require(ref_idx.ndim() == 2 && dim(ref_idx, 1) == 8, "ref_idx must have shape (n, 8)");
    idx rows = dim(ref_idx, 0);
    require_shape(coeffs, {rows, 8}, "coeffs");
    require(count(edge_pixels) == rows, "edge_pixels must have one element per ref_idx row");
    const std::int64_t *pv = valid.data();
    const std::int32_t *pe = edge_pixels.data(), *pr = ref_idx.data();
    for (idx v = 0; v < count(valid); ++v) {
        require(pv[v] >= 0 && pv[v] < rows, "valid index out of range");
        require(pe[pv[v]] >= 0 && pe[pv[v]] < n, "edge pixel out of range");
        for (int j = 0; j < 8; ++j)
            require(pr[pv[v] * 8 + j] < n, "reference pixel out of range");
    }
    const T *pp = phase.data(), *pa = amp.data();
    const float *pc = coeffs.data();
    T *po = out.data();
    idx n_valid = count(valid);
    nb::gil_scoped_release release;
    masked_extrapolation(pp, pa, n, threshold, pe, pr, pc, pv, n_valid, po);
}

template <class T>
double py_abs2_shift(in_array<cplx<T>> in, bool shift, out_array<T> out) {
    require(in.ndim() == 2, "input must be 2D");
    require_shape(out, {dim(in, 0), dim(in, 1)}, "out");
    const cplx<T> *pi = in.data();
    T *po = out.data();
    idx ny = dim(in, 0), nx = dim(in, 1);
    nb::gil_scoped_release release;
    return abs2_shift(pi, ny, nx, shift, po);
}

template <class T>
void py_divide_inplace(out_array<T> a, T d) {
    T *pa = a.data();
    idx n = count(a);
    nb::gil_scoped_release release;
    divide_inplace(pa, n, d);
}

template <class T>
void py_accumulate_sum_sq(in_array<T> v, out_array<T> sum, out_array<T> sum_sq) {
    require(count(sum) == count(v) && count(sum_sq) == count(v), "arrays must have the same size");
    const T *pv = v.data();
    T *ps = sum.data(), *pq = sum_sq.data();
    idx n = count(v);
    nb::gil_scoped_release release;
    accumulate_sum_sq(pv, ps, pq, n);
}

template <class T>
void py_pyr_modulation(in_array<cplx<T>> ef, in_array<cplx<T>> tlt_f, in_array<cplx<T>> ttexp,
                       in_array<cplx<T>> mask, in_array<T> ffv, T norm, in_array<std::int64_t> stop_rows,
                       out_array<cplx<T>> chunk, out_array<T> fpsf, out_array<T> image) {
    require(ef.ndim() == 2 && dim(ef, 0) == dim(ef, 1), "ef must be square");
    idx s = dim(ef, 0);
    require_shape(tlt_f, {s, s}, "tlt_f");
    require(ttexp.ndim() == 3 && dim(ttexp, 1) == s && dim(ttexp, 2) == s, "ttexp must have shape (m, s, s)");
    idx m = dim(ttexp, 0);
    require(count(ffv) == m, "ffv must have one element per modulation step");
    require(mask.ndim() == 2 && dim(mask, 0) == dim(mask, 1) && dim(mask, 0) >= s,
            "mask must have shape (t, t), t >= s");
    idx t = dim(mask, 0);
    require(chunk.ndim() == 3 && dim(chunk, 0) > 0 && dim(chunk, 1) == t && dim(chunk, 2) == t,
            "chunk must have shape (nc, t, t)");
    require_shape(fpsf, {t, t}, "fpsf");
    require_shape(image, {t, t}, "image");
    require(stop_rows.ndim() == 2 && dim(stop_rows, 1) == 2, "stop_rows must have shape (n, 2)");
    const std::int64_t *pr = stop_rows.data();
    for (idx r = 0; r < dim(stop_rows, 0); ++r)
        require(pr[2 * r] >= 0 && pr[2 * r] < pr[2 * r + 1] && pr[2 * r + 1] <= t,
                "stop_rows ranges must be within [0, t)");
    const cplx<T> *pe = ef.data(), *pf = tlt_f.data(), *pt = ttexp.data(), *pm = mask.data();
    const T *pw = ffv.data();
    cplx<T> *pc = chunk.data();
    T *pp = fpsf.data(), *pi = image.data();
    idx n_ranges = dim(stop_rows, 0), nc = dim(chunk, 0);
    nb::gil_scoped_release release;
    pyr_modulation(pe, pf, pt, m, s, pm, pw, norm, pr, n_ranges, pc, nc, t, pp, pi);
}

template <class T>
void py_abs2_tiled_mask(in_array<cplx<T>> in, in_array<T> mask, out_array<T> out) {
    require(in.ndim() == 2 && mask.ndim() == 2, "input and mask must be 2D");
    require_shape(out, {dim(in, 0), dim(in, 1)}, "out");
    require(dim(mask, 0) > 0 && dim(mask, 1) > 0 && dim(in, 0) % dim(mask, 0) == 0 &&
                dim(in, 1) % dim(mask, 1) == 0,
            "the mask must tile the input");
    const cplx<T> *pi = in.data();
    const T *pm = mask.data();
    T *po = out.data();
    idx rows = dim(in, 0), cols = dim(in, 1), mr = dim(mask, 0), mcols = dim(mask, 1);
    nb::gil_scoped_release release;
    abs2_tiled_mask(pi, rows, cols, pm, mr, mcols, po);
}

template <class T>
void py_abs2(in_array<cplx<T>> in, out_array<T> out) {
    require(count(out) == count(in), "input and output must have the same size");
    const cplx<T> *pi = in.data();
    T *po = out.data();
    idx n = count(in);
    nb::gil_scoped_release release;
    abs2_array(pi, n, po);
}

template <class T>
void py_sh_subap_cube(in_array<cplx<T>> ef, idx dimx, in_array<cplx<T>> tltf, out_array<cplx<T>> out) {
    require(tltf.ndim() == 2 && dim(tltf, 0) == dim(tltf, 1), "tltf must be square");
    idx n = dim(tltf, 0);
    require(dimx > 0 && ef.ndim() == 2 && dim(ef, 1) == dimx * n && dim(ef, 0) % n == 0,
            "ef must have shape (rows * n, dimx * n)");
    idx rows = dim(ef, 0) / n;
    require(out.ndim() == 3 && dim(out, 0) == rows * dimx && dim(out, 1) == dim(out, 2) && dim(out, 1) >= n,
            "out must have shape (rows * dimx, fft, fft), fft >= n");
    const cplx<T> *pe = ef.data(), *pt = tltf.data();
    cplx<T> *po = out.data();
    idx fft = dim(out, 1);
    nb::gil_scoped_release release;
    sh_subap_cube(pe, rows, dimx, n, pt, fft, po);
}

template <class T>
void py_sh_otf_product(in_array<cplx<T>> A, in_array<cplx<T>> Kh, idx N, T scale, out_array<cplx<T>> Q) {
    require(A.ndim() == 3 && dim(A, 1) == dim(A, 2), "A must have shape (B, M, M)");
    idx B = dim(A, 0), M = dim(A, 1);
    require(N > 0, "N must be > 0");
    require_shape(Kh, {B, N, N / 2 + 1}, "Kh");
    require(Q.ndim() == 3 && dim(Q, 0) == B && dim(Q, 1) == dim(Q, 2) && dim(Q, 1) % 2 == 1,
            "Q must have shape (B, 2L+1, 2L+1)");
    idx L = (dim(Q, 1) - 1) / 2;
    require(2 * L + 1 <= M && L < N / 2, "the lags must fit in both FFT grids without aliasing");
    const cplx<T> *pa = A.data(), *pk = Kh.data();
    cplx<T> *pq = Q.data();
    nb::gil_scoped_release release;
    sh_otf_product(pa, B, M, pk, N, N / 2 + 1, L, scale, pq);
}

template <class T>
void py_sh_cut_into_image(in_array<T> psf, idx dimx, idx c0, in_array<T> mask, out_array<T> image) {
    require(dimx > 0 && psf.ndim() == 3 && dim(psf, 1) == dim(psf, 2) && dim(psf, 0) % dimx == 0,
            "psf must have shape (rows * dimx, fft, fft)");
    idx rows = dim(psf, 0) / dimx, fft = dim(psf, 1);
    require(rows > 0 && image.ndim() == 2 && dim(image, 0) % rows == 0, "image must have rows * cs rows");
    idx cs = dim(image, 0) / rows;
    require(dim(image, 1) == dimx * cs, "image must have shape (rows * cs, dimx * cs)");
    require(c0 >= 0 && c0 + cs <= fft, "cut outside the psf");
    require_shape(mask, {cs, cs}, "mask");
    const T *pp = psf.data(), *pm = mask.data();
    T *pi = image.data();
    nb::gil_scoped_release release;
    sh_cut_into_image(pp, rows, dimx, fft, c0, cs, pm, pi);
}

template <class T>
void bind_real(nb::module_ &m) {
    auto a = [](const char *name) { return nb::arg(name).noconvert(); };
    m.def("interp2d", &py_interp2d<T>, a("input"), a("xx"), a("yy"), a("out"));
    m.def("interp2d_accumulate", &py_interp2d_accumulate<T>, a("amp"), a("phase"), a("xx"), a("yy"),
          a("amp_out"), a("phase_out"));
    m.def("affine_bilinear", &py_affine_bilinear<T>, a("data"), a("matrix"), a("out"));
    m.def("toccd", &py_toccd<T>, a("input"), a("out"));
    m.def("ef_at_lambda", &py_ef_at_lambda<T>, a("amp"), a("phase"), nb::arg("k"), a("out"));
    m.def("masked_extrapolation", &py_masked_extrapolation<T>, a("phase"), a("amp"),
          nb::arg("threshold"), a("edge_pixels"), a("ref_idx"), a("coeffs"), a("valid"), a("out"));
    m.def("abs2_shift", &py_abs2_shift<T>, a("input"), nb::arg("shift"), a("out"));
    m.def("divide_inplace", &py_divide_inplace<T>, a("a"), nb::arg("d"));
    m.def("accumulate_sum_sq", &py_accumulate_sum_sq<T>, a("value"), a("total"), a("total_sq"));
    m.def("pyr_modulation", &py_pyr_modulation<T>, a("ef"), a("tlt_f"), a("ttexp"), a("mask"), a("ffv"),
          nb::arg("norm"), a("stop_rows"), a("chunk"), a("fpsf"), a("image"));
    m.def("abs2_tiled_mask", &py_abs2_tiled_mask<T>, a("input"), a("mask"), a("out"));
    m.def("abs2", &py_abs2<T>, a("input"), a("out"));
    m.def("sh_subap_cube", &py_sh_subap_cube<T>, a("ef"), nb::arg("dimx"), a("tltf"), a("out"));
    m.def("sh_otf_product", &py_sh_otf_product<T>, a("A"), a("Kh"), nb::arg("N"), nb::arg("scale"),
          a("Q"));
    m.def("sh_cut_into_image", &py_sh_cut_into_image<T>, a("psf"), nb::arg("dimx"), nb::arg("c0"),
          a("mask"), a("image"));
}

}  // namespace

NB_MODULE(specula_cpp, m) {
    m.doc() = "C++ CPU kernels for the SPECULA adaptive optics simulator";
    m.attr("__version__") = "0.1.0";

    m.def("get_num_threads", [] { return pool().size(); },
          "Number of threads used by the kernels and by the FFTs of the C++ backend");
    m.def(
        "set_num_threads",
        [](int n) {
            require(n > 0, "number of threads must be > 0");
            g_requested_threads = n;
            pool().resize(n);
        },
        nb::arg("n"), "Set the number of threads (default: SPECULA_CPP_THREADS or the available CPUs)");

    bind_real<float>(m);
    bind_real<double>(m);
}
