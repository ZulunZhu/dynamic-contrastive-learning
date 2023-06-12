//------------------------------------------------------------------------------
//  Copyright (c) 2018-2020 Michele Morrone
//  All rights reserved.
//
//  https://michelemorrone.eu - https://BrutPitt.com
//
//  twitter: https://twitter.com/BrutPitt - github: https://github.com/BrutPitt/fastPRNG
//
//  mailto:brutpitt@gmail.com - mailto:me@michelemorrone.eu
//
//  This software is distributed under the terms of the BSD 2-Clause license
//------------------------------------------------------------------------------
#pragma once

#include <stdint.h>
#include <chrono>
#include <type_traits>
#include <cfloat>

namespace fastPRNG {
#define UNI_32BIT_INV 2.3283064365386962890625e-10
#define VNI_32BIT_INV 4.6566128730773925781250e-10   // UNI_32BIT_INV * 2

#define UNI_64BIT_INV 5.42101086242752217003726400434970e-20
#define VNI_64BIT_INV 1.08420217248550443400745280086994e-19 // UNI_64BIT_INV * 2

#define FPRNG_SEED_INIT64 std::chrono::system_clock::now().time_since_epoch().count()
#define FPRNG_SEED_INIT32 FPRNG_SEED_INIT64

inline static uint32_t splitMix32(const uint32_t val) {
    uint32_t z = val + 0x9e3779b9;
    z ^= z >> 15; // 16 for murmur3
    z *= 0x85ebca6b;
    z ^= z >> 13;
    z *= 0xc2b2ae35;
    return z ^ (z >> 16);
}

inline static uint64_t splitMix64(const uint64_t val) {
    uint64_t z = val    + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// 32/64 bit rotation func
template <typename T> inline static T rotl(const T x, const int k) { return (x << k) | (x >> (sizeof(T)*8 - k)); } // sizeof*8 is resolved to compile-time

/*--------------------------------------------------------------------------
 64bit PRNG Algorithms: xoshiro / xoroshiro

 xoshiro256+ / xoshiro256++ / xoshiro256**
 xoroshiro128+ / xoroshiro128++ / xoroshiro128**

 Algorithms by David Blackman and Sebastiano Vigna
 http://prng.di.unimi.it/

 To the extent possible under law, the author has dedicated all copyright
 and related and neighboring rights to this software to the public domain
 worldwide. This software is distributed without any warranty.

 See <http://creativecommons.org/publicdomain/zero/1.0/>.
-------------------------------------------------------------------------- */
#define XOSHIRO256\
    const uint64_t t = s1 << 17;\
    s2 ^= s0;\
    s3 ^= s1;\
    s1 ^= s2;\
    s0 ^= s3;\
    s2 ^= t;\
    s3 = rotl<uint64_t>(s3, 45);\
    return result;

#define XOROSHIRO128(A,B,C)\
    s1 ^= s0;\
    s0 = rotl<uint64_t>(s0, A) ^ s1 ^ (s1 << B);\
    s1 = rotl<uint64_t>(s1, C);\
    return result;

#define XORSHIFT64\
    s0 ^= s0 << 13;\
    s0 ^= s0 >> 7;\
    s0 ^= s0 << 17;\
    return s0;

#define XOSHIRO256_STATIC(FUNC)\
    static const uint64_t seed = uint64_t(FPRNG_SEED_INIT64);\
    static uint64_t s0 = splitMix64(seed), s1 = splitMix64(s0), s2 = splitMix64(s1), s3 = splitMix64(s2);\
    FUNC; XOSHIRO256

#define XOROSHIRO128_STATIC(FUNC, A, B, C)\
    static const uint64_t seed = uint64_t(FPRNG_SEED_INIT64);\
    static uint64_t s0 = splitMix64(seed), s1 = splitMix64(s0);\
    FUNC; XOROSHIRO128(A,B,C)

#define XORSHIFT64_STATIC\
    static uint64_t s0 = uint64_t(FPRNG_SEED_INIT64);\
    XORSHIFT64

//
// 64bit pseudo-random generator
// All integer values are returned in interval [0, UINT64_MAX]
// to get values between [INT64_MIN, INT64_MAX] just cast result to int64_t
///////////////////////////////////////////////////////////////////////////////
class fastXS64
{
public:
    fastXS64(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) { seed(seedVal); }

    inline uint64_t xoshiro256p()  { return xoshiro256(s0 + s3); }
    inline uint64_t xoshiro256pp() { return xoshiro256(rotl<uint64_t>(s0 + s3, 23) + s0); }
    inline uint64_t xoshiro256xx() { return xoshiro256(rotl<uint64_t>(s1 * 5, 7) * 9); }

    template <typename T> inline T xoshiro256p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline T xoshiro256p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline T xoshiro256p_Range(T min, T max)                                         // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xoshiro256p_UNI<T>(); }

    inline uint64_t xoroshiro128p()  { return xoroshiro128(     s0 + s1); }
    inline uint64_t xoroshiro128pp() { return xoroshiro128(rotl<uint64_t>(s0 + s1, 17) + s0, 49, 21, 28); }
    inline uint64_t xoroshiro128xx() { return xoroshiro128(rotl<uint64_t>(s0 * 5, 7) * 9); }

    template <typename T> inline T xoroshiro128p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline T xoroshiro128p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline T xoroshiro128p_Range(T min, T max)                                         // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xoroshiro128p_UNI<T>(); }

    inline uint64_t xorShift() { XORSHIFT64 } // Marsaglia xorShift: period 2^64-1

    template <typename T> inline T xorShift_UNI() { return         xorShift()  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline T xorShift_VNI() { return int64_t(xorShift()) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline T xorShift_Range(T min, T max)                                   // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xorShift_UNI<T>(); }

    void seed(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) {
        s0 = splitMix64(seedVal);
        s1 = splitMix64(s0);
        s2 = splitMix64(s1);
        s3 = splitMix64(s2);
    }
private:
    inline uint64_t xoshiro256(const uint64_t result)   { XOSHIRO256 }
    inline uint64_t xoroshiro128(const uint64_t result, const int A = 24, const int B = 16, const int C = 37) { XOROSHIRO128(A,B,C) }

    uint64_t s0, s1, s2, s3;
};

// fastXS64s - static members
//      you can call directly w/o declaration, but..
//      N.B. all members/functions share same seed, and subsequents xor & shift
//           operations on it, if you need different seeds declare more
//           fastXS32 (non static) objects
//
// 64bit pseudo-random generator
// All integer values are returned in interval [0, UINT64_MAX]
// to get values between [INT64_MIN, INT64_MAX] just cast result to int64_t
///////////////////////////////////////////////////////////////////////////////
class fastXS64s
{
public:
    fastXS64s() = default;

    inline static uint64_t xoshiro256p()  { XOSHIRO256_STATIC(const uint64_t result = s0 + s3) }
    inline static uint64_t xoshiro256pp() { XOSHIRO256_STATIC(const uint64_t result = rotl<uint64_t>(s0 + s3, 23) + s0) }
    inline static uint64_t xoshiro256xx() { XOSHIRO256_STATIC(const uint64_t result = rotl<uint64_t>(s1 * 5, 7) * 9) }

    template <typename T> inline static T xoshiro256p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline static T xoshiro256p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline static T xoshiro256p_Range(T min, T max)                                         // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xoshiro256p_UNI<T>(); }

    inline static uint64_t xoroshiro128p()  { XOROSHIRO128_STATIC(const uint64_t result =      s0 + s1,           24, 13, 27) }
    inline static uint64_t xoroshiro128pp() { XOROSHIRO128_STATIC(const uint64_t result = rotl<uint64_t>(s0 + s1, 17) + s0, 49, 21, 28) }
    inline static uint64_t xoroshiro128xx() { XOROSHIRO128_STATIC(const uint64_t result = rotl<uint64_t>(s0 * 5, 7) * 9,    24, 13, 27) }

    template <typename T> inline static T xoroshiro128p_UNI() { return T(        xoshiro256p())  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline static T xoroshiro128p_VNI() { return T(int64_t(xoshiro256p())) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline static T xoroshiro128p_Range(T min, T max)                                         // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xoroshiro128p_UNI<T>(); }

    inline static uint64_t xorShift() { XORSHIFT64_STATIC } // Marsaglia xorShift: period 2^64-1

    template <typename T> inline static T xorShift_UNI() { return         xorShift()  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline static T xorShift_VNI() { return int64_t(xorShift()) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline static T xorShift_Range(T min, T max)                                   // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * xorShift_UNI<T>(); }
};

#undef XOSHIRO256
#undef XOROSHIRO128
#undef XORSHIFT64
#undef XOSHIRO256_STATIC
#undef XOROSHIRO128_STATIC
#undef XORSHIFT64_STATIC

/*--------------------------------------------------------------------------
 64bit PRNG Algorithms:

    znew / wnew / MWC / CNG / FIB / XSH / KISS

 Originally written from George Marsaglia
-------------------------------------------------------------------------- */


// fastRandom64Class
//
// 64bit pseudo-random generator
// All values are returned in interval [0, UINT64_MAX]
// to get values between [INT64_MIN, INT64_MAX] just cast result to int64_t
///////////////////////////////////////////////////////////////////////////////
class fastRandom64Class
{
public:
    // no vaule, seed from system clock, or same seed for same sequence of numbers
    fastRandom64Class(const uint64_t seedVal = uint64_t(FPRNG_SEED_INIT64)) { reset(); seed(seedVal);  }

    // re-seed the current state/values with a new random values
    void seed(const uint64_t seed = uint64_t(FPRNG_SEED_INIT64)) {
        uint64_t s[6];
        s[0] = splitMix64(seed);
        for(int i=1; i<6; i++) s[i] = splitMix64(s[i-1]);
        initialize(s);
    }
    // reset to initial state
    void reset() {
        x=uint64_t(1234567890987654321ULL); c=uint64_t(123456123456123456ULL);
        y=uint64_t(362436362436362436ULL ); z=uint64_t(1066149217761810ULL  );
        a=uint64_t(224466889);              b=uint64_t(7584631);
    }

    inline uint64_t MWC() { uint64_t t; return t=(x<<58)+c, c=(x>>6), x+=t, c+=(x<t), x; }
    inline uint64_t CNG() { return z=6906969069LL*z+1234567;            }
    inline uint64_t XSH() { return y^=(y<<13), y^=(y>>17), y^=(y<<43);  }
    inline uint64_t FIB() { return (b=a+b),(a=b-a);                     }

    inline uint64_t KISS () { return MWC()+XSH()+CNG(); } //period 2^250

    template <typename T> inline T KISS_UNI() { return         KISS()  * UNI_64BIT_INV; } // _UNI<T>   returns value in [ 0, 1] with T ==> float/double
    template <typename T> inline T KISS_VNI() { return int64_t(KISS()) * VNI_64BIT_INV; } // _VNI<T>   returns value in [-1, 1] with T ==> float/double
    template <typename T> inline T KISS_Range(T min, T max)                               // _Range<T> returns value in [min, max] with T ==> float/double
            { return min + (max-min) * KISS_UNI<T>(); }

private:
    void initialize(const uint64_t *i){ x+=i[0]; y+=i[1]; z+=i[2]; c+=i[3]; a=+i[4]; b=+i[5]; }

    uint64_t x, c, y, z;
    uint64_t a, b;
};

} // end of namespace FstRnd

#undef UNI_32BIT_INV
#undef VNI_32BIT_INV
#undef UNI_64BIT_INV
#undef VNI_64BIT_INV
#undef FPRNG_SEED_INIT32
#undef FPRNG_SEED_INIT64
