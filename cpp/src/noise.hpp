#pragma once

// CCD noise: Poisson (photon) noise and Gaussian (readout) noise.
//
// The pixels are split into blocks of kNoiseBlock pixels, and each block draws
// from its own xoshiro256++ generator, seeded from (seed, stream, block index):
// the result depends on the seed and the stream (e.g. the frame number), not on
// the number of threads. The distributions are the same as numpy's, but not the
// random numbers: the noise differs from the noise of a numpy Generator with the
// same seed.

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "kernels.hpp"

namespace specula_cpp {

constexpr idx kNoiseBlock = 256;

inline std::uint64_t splitmix64(std::uint64_t &x) {
    std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// xoshiro256++ (Blackman and Vigna), with doubles in [0, 1) of 53 random bits
class Xoshiro256 {
public:
    Xoshiro256(std::uint64_t seed, std::uint64_t stream, std::uint64_t block) {
        std::uint64_t x = seed;
        x = splitmix64(x) ^ stream;
        x = splitmix64(x) ^ block;
        for (auto &w : s_)
            w = splitmix64(x);
    }

    std::uint64_t next() {
        std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    double next_double() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

// log(Gamma(x)) for x >= 1, as random_loggam() in numpy
inline double loggam(double x) {
    static const double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                                 7.936507936507937e-04, -5.952380952380952e-04,
                                 8.417508417508418e-04, -1.917526917526918e-03,
                                 6.410256410256410e-03, -2.955065359477124e-02,
                                 1.796443723688307e-01, -1.39243221690590e+00};
    if (x == 1.0 || x == 2.0)
        return 0.0;
    std::int64_t n = x < 7.0 ? static_cast<std::int64_t>(7 - x) : 0;
    double x0 = x + n;
    double x2 = (1.0 / x0) * (1.0 / x0);
    double gl0 = a[9];
    for (int k = 8; k >= 0; --k) {
        gl0 *= x2;
        gl0 += a[k];
    }
    double gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * std::log(x0) - x0;
    for (std::int64_t k = 1; k <= n; ++k) {
        gl -= std::log(x0 - 1.0);
        x0 -= 1.0;
    }
    return gl;
}

// Poisson sample, as random_poisson() in numpy: multiplication method for lam < 10,
// PTRS transformed rejection (Hoermann 1993) otherwise
inline double poisson_sample(Xoshiro256 &rng, double lam) {
    if (lam == 0)
        return 0;
    if (lam < 10) {
        double enlam = std::exp(-lam), prod = 1.0;
        std::int64_t x = 0;
        for (;;) {
            prod *= rng.next_double();
            if (prod <= enlam)
                return static_cast<double>(x);
            ++x;
        }
    }
    double slam = std::sqrt(lam), loglam = std::log(lam);
    double b = 0.931 + 2.53 * slam;
    double a = -0.059 + 0.02483 * b;
    double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
        double u = rng.next_double() - 0.5;
        double v = rng.next_double();
        double us = 0.5 - std::fabs(u);
        double k = std::floor((2 * a / us + b) * u + lam + 0.43);
        if (us >= 0.07 && v <= vr)
            return k;
        if (k < 0 || (us < 0.013 && v > us))
            continue;
        if (std::log(v) + std::log(invalpha) - std::log(a / (us * us) + b) <=
            -lam + k * loglam - loggam(k + 1))
            return k;
    }
}

// Call f(rng, begin, end) on each block of pixels, in parallel
template <class F>
void for_noise_blocks(idx n, std::uint64_t seed, std::uint64_t stream, F &&f) {
    idx n_blocks = (n + kNoiseBlock - 1) / kNoiseBlock;
    pool().run(static_cast<int>(n_blocks), [&](int b) {
        Xoshiro256 rng(seed, stream, static_cast<std::uint64_t>(b));
        idx begin = b * kNoiseBlock;
        f(rng, begin, std::min(n, begin + kNoiseBlock));
    });
}

// pixels = Poisson(pixels). Throws if a value is negative or NaN, as numpy.
template <class T>
void poisson_noise(T *pixels, idx n, std::uint64_t seed, std::uint64_t stream) {
    for (idx i = 0; i < n; ++i)
        if (!(pixels[i] >= 0))
            throw std::invalid_argument("poisson_noise: values must be >= 0 and not NaN");
    for_noise_blocks(n, seed, stream, [&](Xoshiro256 &rng, idx b, idx e) {
        for (idx i = b; i < e; ++i)
            pixels[i] = static_cast<T>(poisson_sample(rng, static_cast<double>(pixels[i])));
    });
}

// pixels += sigma * N(0, 1), with the polar method (Marsaglia), two samples at a time
template <class T>
void gaussian_noise(T *pixels, idx n, double sigma, std::uint64_t seed, std::uint64_t stream) {
    for_noise_blocks(n, seed, stream, [&](Xoshiro256 &rng, idx b, idx e) {
        for (idx i = b; i < e; i += 2) {
            double u, v, s;
            do {
                u = 2 * rng.next_double() - 1;
                v = 2 * rng.next_double() - 1;
                s = u * u + v * v;
            } while (s >= 1 || s == 0);
            double f = std::sqrt(-2 * std::log(s) / s);
            pixels[i] = static_cast<T>(pixels[i] + sigma * (u * f));
            if (i + 1 < e)
                pixels[i + 1] = static_cast<T>(pixels[i + 1] + sigma * (v * f));
        }
    });
}

}  // namespace specula_cpp
