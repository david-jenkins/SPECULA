#pragma once

// lib/calc_psf.py: |FFT2|^2 of the field amp * exp(i k phase), zero-padded, in
// a single pass over the output. The rows of the field are transformed first,
// into a buffer of ny x W; the column FFTs (on vl columns at once) are fused
// with |.|^2, the fftshift and the total, so the zero-padded complex field is
// never stored.

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "fft.hpp"

namespace specula_cpp {

// Field ny x nx (strides in elements), output H x W with H >= ny, W >= nx.
// out[ys][xs] = |FFT2(u)[y][x]|^2, with (ys, xs) = (y, x) shifted by (H / 2, W / 2)
// if shift (as fftshift). Returns the total of out.
template <class T>
double psf_fft(const T *amp, idx as0, idx as1, const T *phase, idx ps0, idx ps1, idx ny, idx nx,
               T k, idx H, idx W, bool shift, T *out) {
    constexpr idx vl = n_lanes<T>;
    const pfft::pocketfft_c<T> row_plan(static_cast<size_t>(W)), col_plan(static_cast<size_t>(H));

    // Kept between calls, so that each call does not page fault a new buffer
    thread_local std::vector<std::complex<T>> rows_buffer;
    if (static_cast<idx>(rows_buffer.size()) < ny * W)
        rows_buffer.resize(ny * W);
    std::complex<T> *u = rows_buffer.data();

    // Field and FFT of its rows, vl rows at a time
    pool().run(static_cast<int>((ny + vl - 1) / vl), [&](int g) {
        idx y0 = g * vl, n = std::min(vl, ny - y0);
        lane_buffer<T> buf(W);
        T re[vl] = {}, im[vl] = {};
        for (idx x = 0; x < nx; ++x) {
            for (idx j = 0; j < n; ++j) {
                T ph = phase[(y0 + j) * ps0 + x * ps1] * k;
                T ax = amp[(y0 + j) * as0 + x * as1];
                re[j] = std::cos(ph) * ax;
                im[j] = std::sin(ph) * ax;
            }
            store_lanes(buf, x, re, im);
        }
        std::fill(re, re + vl, T(0));
        std::fill(im, im + vl, T(0));
        for (idx x = nx; x < W; ++x)
            store_lanes(buf, x, re, im);
        row_plan.exec(buf.data(), T(1), true);
        for (idx x = 0; x < W; ++x) {
            load_lanes(buf, x, re, im);
            for (idx j = 0; j < n; ++j)
                u[(y0 + j) * W + x] = {re[j], im[j]};
        }
    });

    // FFT of the columns, on blocks of 16 columns (whole cache lines of each output
    // row), with |.|^2 and the shift. Block totals are summed in order at the end,
    // so the total does not depend on the number of threads.
    constexpr idx block_cols = std::max<idx>(16, vl);
    const idx n_blocks = (W + block_cols - 1) / block_cols;
    const idx ysh = shift ? H / 2 : 0, xsh = shift ? W / 2 : 0;
    std::vector<double> block_total(n_blocks);
    pool().run(static_cast<int>(n_blocks), [&](int b) {
        idx x0 = b * block_cols, nb = std::min(block_cols, W - x0);
        std::vector<lane_buffer<T>> bufs;
        for (idx g = 0; g * vl < nb; ++g)
            bufs.emplace_back(H);
        T re[vl], im[vl];
        const T zero[vl] = {};
        for (idx g = 0; g * vl < nb; ++g) {
            idx nj = std::min(vl, nb - g * vl);
            for (idx y = 0; y < ny; ++y) {
                const std::complex<T> *row = u + y * W + x0 + g * vl;
                for (idx j = 0; j < vl; ++j) {
                    std::complex<T> v = j < nj ? row[j] : std::complex<T>(0);
                    re[j] = v.real();
                    im[j] = v.imag();
                }
                store_lanes(bufs[g], y, re, im);
            }
            for (idx y = ny; y < H; ++y)
                store_lanes(bufs[g], y, zero, zero);
            col_plan.exec(bufs[g].data(), T(1), true);
        }
        idx xs[block_cols];
        for (idx c = 0; c < nb; ++c)
            xs[c] = (x0 + c + xsh) % W;
        double total = 0;
        for (idx y = 0; y < H; ++y) {
            T *dst = out + ((y + ysh) % H) * W;
            double row_total = 0;
            for (idx g = 0; g * vl < nb; ++g) {
                load_lanes(bufs[g], y, re, im);
                idx nj = std::min(vl, nb - g * vl);
                for (idx j = 0; j < nj; ++j) {
                    T v = re[j] * re[j] + im[j] * im[j];
                    dst[xs[g * vl + j]] = v;
                    row_total += v;
                }
            }
            total += row_total;
        }
        block_total[b] = total;
    });
    double total = 0;
    for (double t : block_total)
        total += t;
    return total;
}

}  // namespace specula_cpp
