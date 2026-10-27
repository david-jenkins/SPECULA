#pragma once

// Modulated pyramid WFS: all the modulation steps of a frame, as in the loop of
// ModulatedPyramid.trigger_code(). Each 2D FFT is done one axis at a time with
// pocketfft, on vl rows or columns at once (SIMD), fused with the operations
// between the FFTs, and skipping the rows known to be zero.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "kernels.hpp"
#include "pocketfft_hdronly.h"

namespace specula_cpp {

namespace pfft = pocketfft::detail;

// Element of the pocketfft buffers: the same element of vl 1D FFTs at once (SIMD)
#ifndef POCKETFFT_NO_VECTORS
template <class T>
using lane_t = pfft::vtype_t<T>;
template <class T>
constexpr idx n_lanes = static_cast<idx>(pfft::VLEN<T>::val);
#else
template <class T>
using lane_t = T;
template <class T>
constexpr idx n_lanes = 1;
#endif

template <class T>
using lane_buffer = pfft::arr<pfft::cmplx<lane_t<T>>>;

// The lanes of element i of a buffer, to or from separate real and imaginary parts
template <class T>
inline void store_lanes(lane_buffer<T> &buf, idx i, const T *re, const T *im) {
    std::memcpy(&buf[i].r, re, sizeof(lane_t<T>));
    std::memcpy(&buf[i].i, im, sizeof(lane_t<T>));
}

template <class T>
inline void load_lanes(const lane_buffer<T> &buf, idx i, T *re, T *im) {
    std::memcpy(re, &buf[i].r, sizeof(lane_t<T>));
    std::memcpy(im, &buf[i].i, sizeof(lane_t<T>));
}

// m modulation steps of size t x t; the field ef, tlt_f and ttexp[k] are s x s.
// stop_rows: n_ranges [first, last) ranges of the rows of mask that are not all zero.
// chunk: buffer of nc steps. fpsf and image are accumulated (the caller zeroes them):
//   fpsf  += ffv[k] |FFT(u_k)|^2,  image += ffv[k] |norm IFFT(mask FFT(u_k))|^2
// with u_k = ef * tlt_f * ttexp[k] zero-padded to t x t, and each pixel accumulated
// over k in order, as in trigger_code().
template <class T>
void pyr_modulation(const std::complex<T> *ef, const std::complex<T> *tlt_f,
                    const std::complex<T> *ttexp, idx m, idx s, const std::complex<T> *mask,
                    const T *ffv, T norm, const std::int64_t *stop_rows, idx n_ranges,
                    std::complex<T> *chunk, idx nc, idx t, T *fpsf, T *image) {
    constexpr idx vl = n_lanes<T>;
    const pfft::pocketfft_c<T> plan(static_cast<size_t>(t));

    std::vector<char> in_field(t, 0), in_stop(t, 0);
    std::fill(in_field.begin(), in_field.begin() + s, 1);
    std::vector<std::pair<idx, idx>> stop_groups;  // up to vl consecutive rows inside the stop
    for (idx r = 0; r < n_ranges; ++r) {
        idx first = stop_rows[2 * r], last = stop_rows[2 * r + 1];
        std::fill(in_stop.begin() + first, in_stop.begin() + last, 1);
        for (idx y = first; y < last; y += vl)
            stop_groups.emplace_back(y, std::min(vl, last - y));
    }
    const idx field_groups = (s + vl - 1) / vl;
    const idx n_stop_groups = static_cast<idx>(stop_groups.size());

    // The column passes work on blocks of 16 columns (whole cache lines of each row),
    // in groups of vl columns: blocks done by different threads never share a line
    constexpr idx block_cols = std::max<idx>(16, vl);
    const idx col_blocks = (t + block_cols - 1) / block_cols;

    // Lanes of row y: the columns [x0, x0 + nx) of u, zero beyond nx or if !rows[y]
    auto load_columns = [&](const std::complex<T> *u, idx x0, idx nx, const char *rows,
                            std::vector<lane_buffer<T>> &bufs) {
        T re[vl], im[vl];
        const T zero[vl] = {};
        for (idx y = 0; y < t; ++y) {
            const std::complex<T> *row = u + y * t + x0;
            for (idx g = 0; g * vl < nx; ++g) {
                if (!rows[y]) {
                    store_lanes(bufs[g], y, zero, zero);
                    continue;
                }
                idx nj = std::min(vl, nx - g * vl);
                for (idx j = 0; j < vl; ++j) {
                    std::complex<T> v = j < nj ? row[g * vl + j] : std::complex<T>(0);
                    re[j] = v.real();
                    im[j] = v.imag();
                }
                store_lanes(bufs[g], y, re, im);
            }
        }
    };
    auto column_buffers = [&](idx nx) {
        std::vector<lane_buffer<T>> bufs;
        for (idx g = 0; g * vl < nx; ++g)
            bufs.emplace_back(t);
        return bufs;
    };

    for (idx k0 = 0; k0 < m; k0 += nc) {
        idx kc = std::min(nc, m - k0);

        // Modulated fields, and FFT of the s rows of the field
        pool().run(static_cast<int>(kc * field_groups), [&](int task) {
            idx c = task / field_groups, y0 = (task % field_groups) * vl;
            idx ny = std::min(vl, s - y0);
            const std::complex<T> *tt = ttexp + (k0 + c) * s * s;
            std::complex<T> *u = chunk + c * t * t;
            lane_buffer<T> buf(t);
            T re[vl] = {}, im[vl] = {};
            for (idx x = 0; x < s; ++x) {
                for (idx j = 0; j < ny; ++j) {
                    idx i = (y0 + j) * s + x;
                    std::complex<T> v = cmul(cmul(ef[i], tlt_f[i]), tt[i]);
                    re[j] = v.real();
                    im[j] = v.imag();
                }
                store_lanes(buf, x, re, im);
            }
            std::fill(re, re + vl, T(0));
            std::fill(im, im + vl, T(0));
            for (idx x = s; x < t; ++x)
                store_lanes(buf, x, re, im);
            plan.exec(buf.data(), T(1), true);
            for (idx x = 0; x < t; ++x) {
                load_lanes(buf, x, re, im);
                for (idx j = 0; j < ny; ++j)
                    u[(y0 + j) * t + x] = {re[j], im[j]};
            }
        });

        // FFT of the columns, accumulation of the focal plane PSF, and pyramid mask
        // (rows outside the stop are zero after the mask, and are not stored)
        pool().run(static_cast<int>(col_blocks), [&](int b) {
            idx x0 = b * block_cols, nx = std::min(block_cols, t - x0);
            auto bufs = column_buffers(nx);
            T re[vl], im[vl];
            for (idx c = 0; c < kc; ++c) {
                std::complex<T> *u = chunk + c * t * t;
                load_columns(u, x0, nx, in_field.data(), bufs);
                for (auto &buf : bufs)
                    plan.exec(buf.data(), T(1), true);
                T w = ffv[k0 + c];
                for (idx y = 0; y < t; ++y) {
                    T *fp = fpsf + y * t + x0;
                    std::complex<T> *row = u + y * t + x0;
                    const std::complex<T> *mrow = mask + y * t + x0;
                    for (idx g = 0; g * vl < nx; ++g) {
                        load_lanes(bufs[g], y, re, im);
                        idx nj = std::min(vl, nx - g * vl);
                        T *fpg = fp + g * vl;
                        for (idx j = 0; j < nj; ++j)
                            fpg[j] += (re[j] * re[j] + im[j] * im[j]) * w;
                        if (in_stop[y]) {
                            std::complex<T> *rowg = row + g * vl;
                            const std::complex<T> *mrowg = mrow + g * vl;
                            for (idx j = 0; j < nj; ++j)
                                rowg[j] = cmul(std::complex<T>(re[j], im[j]), mrowg[j]);
                        }
                    }
                }
            }
        });

        // Inverse FFT of the rows inside the stop
        pool().run(static_cast<int>(kc * n_stop_groups), [&](int task) {
            idx c = task / n_stop_groups;
            auto [y0, ny] = stop_groups[task % n_stop_groups];
            std::complex<T> *u = chunk + c * t * t;
            lane_buffer<T> buf(t);
            T re[vl] = {}, im[vl] = {};
            for (idx x = 0; x < t; ++x) {
                for (idx j = 0; j < ny; ++j) {
                    std::complex<T> v = u[(y0 + j) * t + x];
                    re[j] = v.real();
                    im[j] = v.imag();
                }
                store_lanes(buf, x, re, im);
            }
            plan.exec(buf.data(), T(1), false);
            for (idx x = 0; x < t; ++x) {
                load_lanes(buf, x, re, im);
                for (idx j = 0; j < ny; ++j)
                    u[(y0 + j) * t + x] = {re[j], im[j]};
            }
        });

        // Inverse FFT of the columns, and accumulation of the pupil plane image
        pool().run(static_cast<int>(col_blocks), [&](int b) {
            idx x0 = b * block_cols, nx = std::min(block_cols, t - x0);
            auto bufs = column_buffers(nx);
            T re[vl], im[vl];
            for (idx c = 0; c < kc; ++c) {
                load_columns(chunk + c * t * t, x0, nx, in_stop.data(), bufs);
                for (auto &buf : bufs)
                    plan.exec(buf.data(), T(1), false);
                T w = ffv[k0 + c];
                for (idx y = 0; y < t; ++y) {
                    T *im_row = image + y * t + x0;
                    for (idx g = 0; g * vl < nx; ++g) {
                        load_lanes(bufs[g], y, re, im);
                        idx nj = std::min(vl, nx - g * vl);
                        T *img = im_row + g * vl;
                        for (idx j = 0; j < nj; ++j) {
                            T a = re[j] * norm, b2 = im[j] * norm;
                            img[j] += (a * a + b2 * b2) * w;
                        }
                    }
                }
            }
        });
    }
}

}  // namespace specula_cpp
