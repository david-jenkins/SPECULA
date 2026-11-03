#pragma once

// CPU kernels of the SPECULA C++ backend.
//
// Each kernel reproduces the numpy (or CUDA) code it replaces, keeping the
// same order of operations where possible, so that results agree to rounding
// level. Arrays are C-contiguous unless strides are passed explicitly
// (in elements, not bytes).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "thread_pool.hpp"

namespace specula_cpp {

using idx = std::ptrdiff_t;

// Elements per parallel task, enough to amortise the scheduling cost
constexpr idx kGrain = 16384;

inline idx rows_grain(idx row_length) {
    return std::max<idx>(1, kGrain / std::max<idx>(1, row_length));
}

// Plain complex product, as numpy (std::complex operator* may take a slow
// path to handle infinities)
template <class T>
inline std::complex<T> cmul(std::complex<T> a, std::complex<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

template <class T>
inline T abs2(std::complex<T> v) {
    return v.real() * v.real() + v.imag() * v.imag();
}

// ---------------------------------------------------------------------------
// lib/interp2d.py: bilinear interpolation at precomputed, clamped coordinates

template <class T>
struct Bilinear {
    idx i00, i01, i10, i11;
    T xd, yd;

    // Same indexing as the Interp2D CUDA kernel
    Bilinear(idx in_dx, idx in_dy, T x, T y) {
        idx xi = static_cast<idx>(std::floor(x));
        idx yi = static_cast<idx>(std::floor(y));
        idx xi2 = std::min(xi + 1, in_dx - 1);
        idx yi2 = std::min(yi + 1, in_dy - 1);
        xd = x - static_cast<T>(xi);
        yd = y - static_cast<T>(yi);
        i00 = yi * in_dx + xi;
        i01 = yi * in_dx + xi2;
        i10 = yi2 * in_dx + xi;
        i11 = yi2 * in_dx + xi2;
    }

    T operator()(const T *in) const {
        return in[i00] * (1 - xd) * (1 - yd) + in[i01] * xd * (1 - yd) +
               in[i10] * yd * (1 - xd) + in[i11] * xd * yd;
    }
};

template <class T>
void interp2d(const T *in, idx in_dy, idx in_dx, const T *xx, const T *yy, T *out, idx n_out) {
    parallel_for(n_out, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i)
            out[i] = Bilinear<T>(in_dx, in_dy, xx[i], yy[i])(in);
    });
}

// AtmoPropagation, geometric case: amp_out *= interp(amp), phase_out += interp(phase)
template <class T>
void interp2d_accumulate(const T *amp, const T *phase, idx in_dy, idx in_dx, const T *xx,
                         const T *yy, T *amp_out, T *phase_out, idx n_out) {
    parallel_for(n_out, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i) {
            Bilinear<T> w(in_dx, in_dy, xx[i], yy[i]);
            amp_out[i] *= w(amp);
            phase_out[i] += w(phase);
        }
    });
}

// ---------------------------------------------------------------------------
// lib/affine_transform.py: _bilinear(), zero outside [0, n-1] on either axis

template <class T>
void affine_bilinear(const T *data, idx ny, idx nx, const double *m, T *out, idx oy, idx ox) {
    // m = [[a00, a01, b0], [a10, a11, b1]]: data coordinates (y, x) of output (i0, i1)
    parallel_for(oy, rows_grain(ox), [&](idx r0, idx r1) {
        for (idx i0 = r0; i0 < r1; ++i0) {
            double ybase = m[0] * i0 + m[2];
            double xbase = m[3] * i0 + m[5];
            T *o = out + i0 * ox;
            for (idx i1 = 0; i1 < ox; ++i1) {
                double y = ybase + m[1] * i1;
                double x = xbase + m[4] * i1;
                if (!(y >= 0 && y <= ny - 1 && x >= 0 && x <= nx - 1)) {
                    o[i1] = 0;
                    continue;
                }
                double fy = std::floor(y), fx = std::floor(x);
                idx y0 = static_cast<idx>(fy), x0 = static_cast<idx>(fx);
                idx y1 = std::min(y0 + 1, ny - 1), x1 = std::min(x0 + 1, nx - 1);
                T wy = static_cast<T>(y - fy), wx = static_cast<T>(x - fx);
                const T *ra = data + y0 * nx;
                const T *rb = data + y1 * nx;
                T a = ra[x0] + (ra[x1] - ra[x0]) * wx;
                T b = rb[x0] + (rb[x1] - rb[x0]) * wx;
                o[i1] = a + (b - a) * wy;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// lib/toccd.py: rebin through the L.C.M. grid without allocating it

inline idx gcd(idx a, idx b) {
    while (b) {
        idx t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Overlap, in L.C.M. sub-pixels, of each output pixel with the input pixels
struct RebinAxis {
    std::vector<idx> first;   // first input pixel of each output pixel
    std::vector<idx> offset;  // its weights are w[offset[o]:offset[o+1]]
    std::vector<double> w;
    idx sub_out;              // sub-pixels per output pixel

    RebinAxis(idx n_in, idx n_out) : first(n_out), offset(n_out + 1, 0) {
        idx lcm = n_in / gcd(n_in, n_out) * n_out;
        idx sub_in = lcm / n_in;
        sub_out = lcm / n_out;
        for (idx o = 0; o < n_out; ++o) {
            idx a = o * sub_out, b = a + sub_out;
            first[o] = a / sub_in;
            for (idx i = a / sub_in; i <= (b - 1) / sub_in; ++i)
                w.push_back(static_cast<double>(std::min(b, (i + 1) * sub_in) - std::max(a, i * sub_in)));
            offset[o + 1] = static_cast<idx>(w.size());
        }
    }
};

// Same values as the toccd() CUDA kernels (sum over the L.C.M. grid divided by
// the sub-pixels per output pixel), not normalised. Returns the output total.
template <class T>
double toccd(const T *in, idx iny, idx inx, T *out, idx outy, idx outx) {
    RebinAxis ax(inx, outx), ay(iny, outy);
    std::vector<double> tmp(static_cast<size_t>(iny) * outx);
    parallel_for(iny, rows_grain(inx), [&](idx r0, idx r1) {
        for (idx y = r0; y < r1; ++y) {
            const T *row = in + y * inx;
            double *t = tmp.data() + y * outx;
            for (idx o = 0; o < outx; ++o) {
                const double *w = ax.w.data() + ax.offset[o];
                const T *src = row + ax.first[o];
                double s = 0;
                for (idx k = 0; k < ax.offset[o + 1] - ax.offset[o]; ++k)
                    s += w[k] * src[k];
                t[o] = s;
            }
        }
    });
    double f = 1.0 / (static_cast<double>(ax.sub_out) * static_cast<double>(ay.sub_out));
    std::vector<double> row_total(outy);
    parallel_for(outy, rows_grain(outx * 4), [&](idx r0, idx r1) {
        std::vector<double> acc(outx);
        for (idx o = r0; o < r1; ++o) {
            std::fill(acc.begin(), acc.end(), 0.0);
            const double *w = ay.w.data() + ay.offset[o];
            for (idx k = 0; k < ay.offset[o + 1] - ay.offset[o]; ++k) {
                const double *t = tmp.data() + (ay.first[o] + k) * outx;
                for (idx x = 0; x < outx; ++x)
                    acc[x] += w[k] * t[x];
            }
            T *dst = out + o * outx;
            double s = 0;
            for (idx x = 0; x < outx; ++x) {
                dst[x] = static_cast<T>(acc[x] * f);
                s += dst[x];
            }
            row_total[o] = s;
        }
    });
    return std::accumulate(row_total.begin(), row_total.end(), 0.0);
}

// ---------------------------------------------------------------------------
// data_objects/electric_field.py: ef_at_lambda(), A * exp(i * k * phase)

template <class T>
void ef_at_lambda(const T *amp, idx as0, idx as1, const T *phase, idx ps0, idx ps1, idx ny,
                  idx nx, T k, std::complex<T> *out, idx os0, idx os1) {
    parallel_for(ny, rows_grain(nx), [&](idx r0, idx r1) {
        for (idx y = r0; y < r1; ++y) {
            const T *a = amp + y * as0;
            const T *p = phase + y * ps0;
            std::complex<T> *o = out + y * os0;
            for (idx x = 0; x < nx; ++x) {
                T ph = p[x * ps1] * k;
                T ax = a[x * as1];
                o[x * os1] = {std::cos(ph) * ax, std::sin(ph) * ax};
            }
        }
    });
}

// lib/extrapolation_2d.py: EFInterpolator phase masking and edge extrapolation
template <class T>
void masked_extrapolation(const T *phase, const T *amp, idx n, T threshold,
                          const std::int32_t *edge_pixels, const std::int32_t *ref_idx,
                          const float *coeffs, const std::int64_t *valid, idx n_valid, T *out) {
    parallel_for(n, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i)
            out[i] = amp[i] >= threshold ? phase[i] : T(0);
    });
    for (idx v = 0; v < n_valid; ++v) {
        idx row = valid[v];
        const std::int32_t *r = ref_idx + row * 8;
        const float *c = coeffs + row * 8;
        T s = 0;
        for (int j = 0; j < 8; ++j)
            if (r[j] >= 0)
                s += static_cast<T>(c[j]) * phase[r[j]];
        out[edge_pixels[row]] = s;
    }
}

// ---------------------------------------------------------------------------
// PSF: |FFT|^2 with optional fftshift; returns the total

template <class T>
double abs2_shift(const std::complex<T> *in, idx ny, idx nx, bool shift, T *out) {
    std::vector<double> row_total(ny);
    parallel_for(ny, rows_grain(nx), [&](idx r0, idx r1) {
        for (idx y = r0; y < r1; ++y) {
            const std::complex<T> *src = in + y * nx;
            idx ys = shift ? (y + ny / 2) % ny : y;
            T *dst = out + ys * nx;
            idx xs = shift ? nx / 2 : 0;
            double s = 0;
            for (idx x = 0; x < nx - xs; ++x) {
                T v = abs2(src[x]);
                dst[x + xs] = v;
                s += v;
            }
            for (idx x = nx - xs; x < nx; ++x) {
                T v = abs2(src[x]);
                dst[x + xs - nx] = v;
                s += v;
            }
            row_total[y] = s;
        }
    });
    return std::accumulate(row_total.begin(), row_total.end(), 0.0);
}

template <class T>
void divide_inplace(T *a, idx n, T d) {
    parallel_for(n, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i)
            a[i] /= d;
    });
}

// PSF.post_trigger(): sum += v, sum_sq += v * v
template <class T>
void accumulate_sum_sq(const T *v, T *sum, T *sum_sq, idx n) {
    parallel_for(n, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i) {
            sum[i] += v[i];
            sum_sq[i] += v[i] * v[i];
        }
    });
}

// ---------------------------------------------------------------------------
// ModulatedPyramid, for a chunk of mc modulation steps

// u[c, :s, :s] = ef * tlt_f * ttexp[c], zero elsewhere; u is (mc, t, t)
template <class T>
void pyr_modulate(const std::complex<T> *ef, const std::complex<T> *tlt_f,
                  const std::complex<T> *ttexp, idx mc, idx s, idx t, std::complex<T> *u) {
    parallel_for(mc * t, rows_grain(t), [&](idx r0, idx r1) {
        for (idx r = r0; r < r1; ++r) {
            idx c = r / t, y = r % t;
            std::complex<T> *dst = u + r * t;
            idx filled = 0;
            if (y < s) {
                const std::complex<T> *e = ef + y * s;
                const std::complex<T> *f = tlt_f + y * s;
                const std::complex<T> *m = ttexp + (c * s + y) * s;
                for (idx x = 0; x < s; ++x)
                    dst[x] = cmul(cmul(e[x], f[x]), m[x]);
                filled = s;
            }
            std::fill(dst + filled, dst + t, std::complex<T>(0));
        }
    });
}

// Focal plane of each step c: fpsf += ffv[c] * |F[c]|^2, then F[c] *= mask
template <class T>
void pyr_focal(std::complex<T> *F, idx mc, idx n, const std::complex<T> *mask, const T *ffv,
               T *fpsf) {
    constexpr idx block = 2048;
    parallel_for((n + block - 1) / block, 1, [&](idx b0, idx b1) {
        for (idx b = b0; b < b1; ++b) {
            idx p0 = b * block, p1 = std::min(n, p0 + block);
            for (idx c = 0; c < mc; ++c) {
                std::complex<T> *f = F + c * n;
                for (idx p = p0; p < p1; ++p) {
                    fpsf[p] += abs2(f[p]) * ffv[c];
                    f[p] = cmul(f[p], mask[p]);
                }
            }
        }
    });
}

// Pupil plane of each step c: image += ffv[c] * |norm * P[c]|^2
template <class T>
void pyr_pupil(const std::complex<T> *P, idx mc, idx n, const T *ffv, T norm, T *image) {
    constexpr idx block = 2048;
    parallel_for((n + block - 1) / block, 1, [&](idx b0, idx b1) {
        for (idx b = b0; b < b1; ++b) {
            idx p0 = b * block, p1 = std::min(n, p0 + block);
            for (idx c = 0; c < mc; ++c) {
                const std::complex<T> *f = P + c * n;
                for (idx p = p0; p < p1; ++p) {
                    std::complex<T> v{f[p].real() * norm, f[p].imag() * norm};
                    image[p] += abs2(v) * ffv[c];
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// SH

// out[r, c] = |in[r, c]|^2 * mask[r % mr, c % mc]
template <class T>
void abs2_tiled_mask(const std::complex<T> *in, idx rows, idx cols, const T *mask, idx mr,
                     idx mcols, T *out) {
    parallel_for(rows, rows_grain(cols), [&](idx r0, idx r1) {
        for (idx r = r0; r < r1; ++r) {
            const std::complex<T> *src = in + r * cols;
            const T *mrow = mask + (r % mr) * mcols;
            T *dst = out + r * cols;
            for (idx c = 0; c < cols; ++c)
                dst[c] = abs2(src[c]) * mrow[c % mcols];
        }
    });
}

template <class T>
void abs2_array(const std::complex<T> *in, idx n, T *out) {
    parallel_for(n, kGrain, [&](idx b, idx e) {
        for (idx i = b; i < e; ++i)
            out[i] = abs2(in[i]);
    });
}

// Zero-padded subaperture cube from `rows` rows of subapertures of the field:
// out[r*dimx + sx][y][x] = ef[r*n + y][sx*n + x] * tltf[y][x] for y, x < n, else 0
template <class T>
void sh_subap_cube(const std::complex<T> *ef, idx rows, idx dimx, idx n,
                   const std::complex<T> *tltf, idx fft, std::complex<T> *out) {
    idx width = dimx * n;
    parallel_for(rows * dimx * fft, rows_grain(fft), [&](idx l0, idx l1) {
        for (idx l = l0; l < l1; ++l) {
            idx subap = l / fft, y = l % fft;
            idx r = subap / dimx, sx = subap % dimx;
            std::complex<T> *dst = out + l * fft;
            idx filled = 0;
            if (y < n) {
                const std::complex<T> *src = ef + (r * n + y) * width + sx * n;
                const std::complex<T> *tl = tltf + y * n;
                for (idx x = 0; x < n; ++x)
                    dst[x] = cmul(src[x], tl[x]);
                filled = n;
            }
            std::fill(dst + filled, dst + fft, std::complex<T>(0));
        }
    });
}

// SH with a convolution kernel, on the support of the OTF only: product of the
// autocorrelation of each subaperture field and of its kernel, at the lags |l| <= L.
// A[b] is the inverse FFT (size M >= 2L+1) of |FFT(w)|^2, so A[b][l] = sum_y w[y+l] conj(w[y]);
// Kh[b] is the half spectrum (rfft2 layout) of the real kernel on the N x N FFT grid.
// Q[b][L+ly][L+lx] = scale * A[b][-ly][-lx] * K(ly, lx), with K(ly, lx) = Kh[ly][lx] for lx >= 0
// and conj(Kh[-ly][-lx]) for lx < 0 (indices modulo M for A, modulo N for Kh).
template <class T>
void sh_otf_product(const std::complex<T> *A, idx B, idx M, const std::complex<T> *Kh, idx N,
                    idx NH, idx L, T scale, std::complex<T> *Q) {
    idx S = 2 * L + 1;
    auto mod = [](idx a, idx m) { return ((a % m) + m) % m; };
    parallel_for(B * S, rows_grain(S), [&](idx r0, idx r1) {
        for (idx r = r0; r < r1; ++r) {
            idx b = r / S, ly = r % S - L;
            const std::complex<T> *a = A + (b * M + mod(-ly, M)) * M;
            const std::complex<T> *kp = Kh + (b * N + mod(ly, N)) * NH;   // lx >= 0
            const std::complex<T> *kn = Kh + (b * N + mod(-ly, N)) * NH;  // lx < 0
            std::complex<T> *q = Q + r * S;
            for (idx j = 0; j < S; ++j) {
                idx lx = j - L;
                std::complex<T> k = lx >= 0 ? kp[lx] : std::conj(kn[-lx]);
                std::complex<T> v = cmul(a[mod(-lx, M)], k);
                q[j] = {v.real() * scale, v.imag() * scale};
            }
        }
    });
}

// image[r*cs + y][sx*cs + x] = psf[r*dimx + sx][c0 + y][c0 + x] * mask[y][x]
template <class T>
void sh_cut_into_image(const T *psf, idx rows, idx dimx, idx fft, idx c0, idx cs, const T *mask,
                       T *image) {
    idx width = dimx * cs;
    parallel_for(rows * cs, rows_grain(width), [&](idx l0, idx l1) {
        for (idx l = l0; l < l1; ++l) {
            idx r = l / cs, y = l % cs;
            T *dst = image + l * width;
            const T *mrow = mask + y * cs;
            for (idx sx = 0; sx < dimx; ++sx) {
                const T *src = psf + ((r * dimx + sx) * fft + c0 + y) * fft + c0;
                T *d = dst + sx * cs;
                for (idx x = 0; x < cs; ++x)
                    d[x] = src[x] * mrow[x];
            }
        }
    });
}

}  // namespace specula_cpp
