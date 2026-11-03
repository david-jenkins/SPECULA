#pragma once

// Modulated pyramid WFS: all the modulation steps of a frame, as in the loop of
// ModulatedPyramid.trigger_code(). Each 2D FFT is done one axis at a time with
// pocketfft, on vl rows or columns at once (SIMD). The intermediate results stay
// in the layout of the pocketfft buffers, so that most FFTs run in place, and the
// operations between the FFTs are fused with them.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fft.hpp"

namespace specula_cpp {

// m modulation steps of size t x t; the field ef, tlt_f and ttexp[k] are s x s.
// stop_rows: n_ranges [first, last) ranges of the rows of mask that are not all zero.
// nc: number of steps computed together. fpsf and image are accumulated (the caller
// zeroes them):
//   fpsf  += ffv[k] |FFT(u_k)|^2,  image += ffv[k] |norm IFFT(mask FFT(u_k))|^2
// with u_k = ef * tlt_f * ttexp[k] zero-padded to t x t, and each pixel accumulated
// over k in order, as in trigger_code().
//
// For each chunk of nc steps:
// A. FFT of the s columns of u_k, vl columns at a time (lanes = columns), in place;
// B. for each group of vl rows (lanes = rows): transpose from A, FFT of the rows,
//    fpsf, and, for the rows inside the stop, mask and inverse FFT of the rows,
//    transposed into C (zero for the rows outside the stop);
// C. inverse FFT of the columns, in place in C (lanes = columns), and image.
template <class T>
void pyr_modulation(const std::complex<T> *ef, const std::complex<T> *tlt_f,
                    const std::complex<T> *ttexp, idx m, idx s, const std::complex<T> *mask,
                    const T *ffv, T norm, const std::int64_t *stop_rows, idx n_ranges, idx nc,
                    idx t, T *fpsf, T *image) {
    using E = lane_cplx<T>;
    constexpr idx vl = n_lanes<T>;
    const pfft::pocketfft_c<T> plan(static_cast<size_t>(t));
    const idx gs = (s + vl - 1) / vl;  // groups of columns of the field
    const idx gt = (t + vl - 1) / vl;  // groups of rows or columns of the t x t arrays
    nc = std::min(nc, m);

    // A[c][gx][y], C[c][gx][y]: column y of the group of columns gx of step c.
    // M[gy][x]: the mask on the group of rows gy.
    E *A = thread_scratch<E, 0>(nc * gs * t);
    E *C = thread_scratch<E, 1>(nc * gt * t);
    E *M = thread_scratch<E, 2>(gt * t);
    // fpsf and image accumulated in the lanes layout, added to them at the end:
    // F[gy][x] (lanes = rows), I[gx][y] (lanes = columns)
    using V = lane_t<T>;
    V *F = thread_scratch<V, 0>(gt * t);
    V *I = thread_scratch<V, 1>(gt * t);
    std::fill(F, F + gt * t, V{});
    std::fill(I, I + gt * t, V{});

    std::vector<char> group_in_stop(gt, 0);
    for (idx r = 0; r < n_ranges; ++r)
        for (idx y = stop_rows[2 * r]; y < stop_rows[2 * r + 1]; ++y)
            group_in_stop[y / vl] = 1;

    pool().run(static_cast<int>(gt), [&](int gy) {
        if (!group_in_stop[gy])
            return;
        idx y0 = gy * vl, ny = std::min(vl, t - y0);
        T re[vl] = {}, im[vl] = {};
        for (idx x = 0; x < t; ++x) {
            for (idx j = 0; j < ny; ++j) {
                std::complex<T> v = mask[(y0 + j) * t + x];
                re[j] = v.real();
                im[j] = v.imag();
            }
            store_lanes<T>(M[gy * t + x], re, im);
        }
    });

    for (idx k0 = 0; k0 < m; k0 += nc) {
        idx kc = std::min(nc, m - k0);

        // A. Modulated fields, and FFT of their columns
        pool().run(static_cast<int>(kc * gs), [&](int task) {
            idx c = task / gs, gx = task % gs;
            idx x0 = gx * vl, nx = std::min(vl, s - x0);
            const std::complex<T> *tt = ttexp + (k0 + c) * s * s;
            E *col = A + (c * gs + gx) * t;
            T re[vl] = {}, im[vl] = {};
            for (idx y = 0; y < s; ++y) {
                idx i0 = y * s + x0;
                for (idx j = 0; j < nx; ++j) {
                    std::complex<T> v = cmul(cmul(ef[i0 + j], tlt_f[i0 + j]), tt[i0 + j]);
                    re[j] = v.real();
                    im[j] = v.imag();
                }
                store_lanes<T>(col[y], re, im);
            }
            std::fill(re, re + vl, T(0));
            std::fill(im, im + vl, T(0));
            for (idx y = s; y < t; ++y)
                store_lanes<T>(col[y], re, im);
            plan.exec(col, T(1), true);
        });

        // B. FFT of the rows, focal plane PSF, pyramid mask and inverse FFT of the rows
        pool().run(static_cast<int>(gt), [&](int gy) {
            idx y0 = gy * vl, ny = std::min(vl, t - y0);
            lane_buffer<T> buf(t);
            T re[vl], im[vl];
            for (idx c = 0; c < kc; ++c) {
                for (idx gx = 0; gx < gs; ++gx)
                    transpose_lanes<T>(A + (c * gs + gx) * t + y0, 1, ny, &buf[gx * vl], 1,
                                       std::min(vl, t - gx * vl));
                std::fill(re, re + vl, T(0));
                std::fill(im, im + vl, T(0));
                for (idx x = gs * vl; x < t; ++x)
                    store_lanes(buf, x, re, im);
                plan.exec(buf.data(), T(1), true);

                T w = ffv[k0 + c];
                V *f = F + gy * t;
                for (idx x = 0; x < t; ++x)
                    f[x] += (buf[x].r * buf[x].r + buf[x].i * buf[x].i) * w;

                E *out = C + c * gt * t + y0;
                if (!group_in_stop[gy]) {
                    // Zero after the mask
                    std::fill(re, re + vl, T(0));
                    std::fill(im, im + vl, T(0));
                    for (idx gx = 0; gx < gt; ++gx)
                        for (idx j = 0; j < ny; ++j)
                            store_lanes<T>(out[gx * t + j], re, im);
                    continue;
                }
                const E *mrow = M + gy * t;
                for (idx x = 0; x < t; ++x)
                    buf[x] = buf[x] * mrow[x];
                plan.exec(buf.data(), T(1), false);
                for (idx gx = 0; gx < gt; ++gx)
                    transpose_lanes<T>(&buf[gx * vl], 1, std::min(vl, t - gx * vl), out + gx * t, 1, ny);
            }
        });

        // C. Inverse FFT of the columns, and accumulation of the pupil plane image
        pool().run(static_cast<int>(gt), [&](int gx) {
            V *acc = I + gx * t;
            for (idx c = 0; c < kc; ++c) {
                T w = ffv[k0 + c];
                E *col = C + (c * gt + gx) * t;
                plan.exec(col, T(1), false);
                for (idx y = 0; y < t; ++y) {
                    V a = col[y].r * norm, b = col[y].i * norm;
                    acc[y] += (a * a + b * b) * w;
                }
            }
        });
    }

    // Accumulated values into fpsf and image, in blocks of 16 columns of image
    // (whole cache lines of each row), so that threads never share a line
    constexpr idx block_groups = std::max<idx>(1, 16 / vl);
    pool().run(static_cast<int>(gt), [&](int gy) {
        idx y0 = gy * vl, ny = std::min(vl, t - y0);
        T v[vl];
        for (idx x = 0; x < t; ++x) {
            std::memcpy(v, &F[gy * t + x], sizeof(V));
            for (idx j = 0; j < ny; ++j)
                fpsf[(y0 + j) * t + x] += v[j];
        }
    });
    pool().run(static_cast<int>((gt + block_groups - 1) / block_groups), [&](int b) {
        idx g1 = std::min(gt, (b + 1) * block_groups);
        T v[vl];
        for (idx y = 0; y < t; ++y)
            for (idx gx = b * block_groups; gx < g1; ++gx) {
                idx x0 = gx * vl, nx = std::min(vl, t - x0);
                std::memcpy(v, &I[gx * t + y], sizeof(V));
                T *img = image + y * t + x0;
                for (idx j = 0; j < nx; ++j)
                    img[j] += v[j];
            }
    });
}

}  // namespace specula_cpp
