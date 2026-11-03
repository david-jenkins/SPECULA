#pragma once

// pocketfft 1D FFTs on vl rows or columns at once (SIMD): each element of the
// buffers holds the same element of vl transforms.

#include <complex>
#include <cstring>

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
using lane_cplx = pfft::cmplx<lane_t<T>>;
template <class T>
using lane_buffer = pfft::arr<lane_cplx<T>>;

// The lanes of an element, to or from separate real and imaginary parts
template <class T>
inline void store_lanes(lane_cplx<T> &e, const T *re, const T *im) {
    std::memcpy(&e.r, re, sizeof(lane_t<T>));
    std::memcpy(&e.i, im, sizeof(lane_t<T>));
}

template <class T>
inline void load_lanes(const lane_cplx<T> &e, T *re, T *im) {
    std::memcpy(re, &e.r, sizeof(lane_t<T>));
    std::memcpy(im, &e.i, sizeof(lane_t<T>));
}

template <class T>
inline void store_lanes(lane_buffer<T> &buf, idx i, const T *re, const T *im) {
    store_lanes<T>(buf[i], re, im);
}

template <class T>
inline void load_lanes(const lane_buffer<T> &buf, idx i, T *re, T *im) {
    load_lanes<T>(buf[i], re, im);
}

// Transpose of a block of vl x vl values: element in[j * in_step] holds the values
// (j, 0..vl-1), element out[i * out_step] receives the values (0..vl-1, i).
// Only the first n_in input elements are read (the others are zero), and only
// the first n_out output elements are written.
template <class T>
inline void transpose_lanes(const lane_cplx<T> *in, idx in_step, idx n_in, lane_cplx<T> *out,
                            idx out_step, idx n_out) {
    constexpr idx vl = n_lanes<T>;
    T re[vl][vl], im[vl][vl];
    if (n_in == vl && n_out == vl) {
        // Whole block: fixed loop bounds, which the compiler turns into shuffles
        for (idx j = 0; j < vl; ++j)
            load_lanes<T>(in[j * in_step], re[j], im[j]);
        for (idx i = 0; i < vl; ++i) {
            T r[vl], m[vl];
            for (idx j = 0; j < vl; ++j) {
                r[j] = re[j][i];
                m[j] = im[j][i];
            }
            store_lanes<T>(out[i * out_step], r, m);
        }
        return;
    }
    for (idx j = 0; j < vl; ++j)
        for (idx i = 0; i < vl; ++i)
            re[j][i] = im[j][i] = T(0);
    for (idx j = 0; j < n_in; ++j)
        load_lanes<T>(in[j * in_step], re[j], im[j]);
    for (idx i = 0; i < n_out; ++i) {
        T r[vl], m[vl];
        for (idx j = 0; j < vl; ++j) {
            r[j] = re[j][i];
            m[j] = im[j][i];
        }
        store_lanes<T>(out[i * out_step], r, m);
    }
}

// Aligned buffer of at least n elements, kept between calls by the calling thread
// (so that each call does not page fault new memory); Id tells buffers apart.
template <class E, int Id>
E *thread_scratch(size_t n) {
    thread_local pfft::arr<E> buf;
    if (buf.size() < n)
        buf.resize(n);
    return buf.data();
}

}  // namespace specula_cpp
