/*
Copyright 2024 Emmanouil Krasanakis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef tensorless_BIT256_H
#define tensorless_BIT256_H

#include <immintrin.h>  // For AVX intrinsics
#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "backend.h"

namespace CPU {

using Bit128VECTOR = __int128;
using Bit256VECTOR = __m256i;

template<>
class Bit<256> {
public:
    typedef Bit256VECTOR VECTOR;

    static INLINE int get(const VECTOR &x, int i) {
        int lane = i<128?0:1;
        int bit_position = i<128?i:(i-128);
        
        // Extract the 128-bit half and check the bit
        Bit128VECTOR half = extract128(x, lane);
        return (half & (static_cast<Bit128VECTOR>(1) << bit_position)) != 0;
    }

    static INLINE int get(const VECTOR &x, int i, int value) {
        return get(x, i) ? value : 0;
    }

    static INLINE int bitcount(const VECTOR &x) {
        Bit128VECTOR lower = extract128(x, 0);
        Bit128VECTOR upper = extract128(x, 1);
        return __builtin_popcountll(lower) + __builtin_popcountll(upper);
    }
    
    static INLINE VECTOR rand() {
        return _mm256_set_epi64x(distribution(generator),
                                 distribution(generator), 
                                 distribution(generator), 
                                 distribution(generator));
    }


    static INLINE bool any(const VECTOR &x) {
        Bit128VECTOR lower = extract128(x, 0);
        Bit128VECTOR upper = extract128(x, 1);
        return lower != 0 || upper != 0;
    }

    static INLINE VECTOR onehot(int i) {
        if(i<128) {
            Bit128VECTOR one_hot = static_cast<Bit128VECTOR>(1) << i;
            return combine128(one_hot, 0);
        }
        Bit128VECTOR one_hot = static_cast<Bit128VECTOR>(1) << (i%128);
        return combine128(0, one_hot);
    }

    static INLINE void set(VECTOR &x, int i, const bool &value) {
        if (value) one(x, i); else zero(x, i);
    }

    static const VECTOR ZERO;
    static const VECTOR ONES;
    static const int size = 256;

private:
    static std::mt19937_64 generator;
    static std::uniform_int_distribution<uint64_t> distribution;

    // Set the bit at index i to 1
    static INLINE void one(VECTOR &x, int i) {
        x = _mm256_or_si256(x, onehot(i));
    }

    // Set the bit at index i to 0
    static INLINE void zero(VECTOR &x, int i) {
        x = _mm256_andnot_si256(onehot(i), x);
    }

    // Extract the 128-bit half (0 = lower, 1 = upper)
    static INLINE Bit128VECTOR extract128(const VECTOR &x, int index) {
        return index == 0 ? _mm256_extract_epi64(x, 0) | 
                            (static_cast<Bit128VECTOR>(_mm256_extract_epi64(x, 1)) << 64)
                          : _mm256_extract_epi64(x, 2) | 
                            (static_cast<Bit128VECTOR>(_mm256_extract_epi64(x, 3)) << 64);
    }

    // Combine two 128-bit halves into a 256-bit vector
    static INLINE VECTOR combine128(const Bit128VECTOR &lower, const Bit128VECTOR &upper) {
        return _mm256_set_epi64x(static_cast<int64_t>(upper >> 64), 
                                 static_cast<int64_t>(upper),
                                 static_cast<int64_t>(lower >> 64),
                                 static_cast<int64_t>(lower));
    }
};

// Static member initialization
const Bit<256>::VECTOR Bit<256>::ZERO = _mm256_setzero_si256();
const Bit<256>::VECTOR Bit<256>::ONES = _mm256_set1_epi64x(-1);
std::mt19937_64 Bit<256>::generator = std::mt19937_64(rd());
std::uniform_int_distribution<uint64_t> Bit<256>::distribution(0, UINT64_MAX);

}  // namespace CPU

#endif  // tensorless_BIT256_H
