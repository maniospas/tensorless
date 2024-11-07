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

#ifndef tensorless_BIT128_H
#define tensorless_BIT128_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "backend.h"

namespace CPU {

using Bit64VECTOR = int_fast64_t;//long long;
using Bit128VECTOR = __int128;

template<>
class Bit<128> {
public:
    typedef Bit128VECTOR VECTOR;
    static INLINE auto get(const VECTOR &x, int i) {return (x >> i) & 1;}
    static INLINE int get(const VECTOR &x, int i, int value) {if((x >> i) & 1) return value; return 0;}
    static INLINE int bitcount(const VECTOR &x) {return __builtin_popcountll(static_cast<uint64_t>(x))+__builtin_popcountll(static_cast<uint64_t>((x) >> 64));}
    static INLINE const VECTOR& any(const VECTOR &x) {return x;}
    static INLINE VECTOR rand() {return ((VECTOR)distribution(generator))<<64 | (VECTOR)distribution(generator);}
    static INLINE VECTOR onehot(int i) {return ((VECTOR)1) << i;}
    static INLINE void set(VECTOR &x, int i, const bool &value) {if(value) one(x, i); else zero(x, i);}
    static const VECTOR ZERO = 0;
    static const VECTOR ONES = ~(Bit128VECTOR)0;
    static const int size = 128;
private:
    static std::mt19937_64 generator;
    static std::uniform_int_distribution<Bit64VECTOR> distribution;
    static INLINE void one(VECTOR &x, int i) {x |= onehot(i);}
    static INLINE void zero(VECTOR &x, int i) {x &= ~onehot(i);}
};

std::mt19937_64 Bit<128>::generator = std::mt19937_64(rd());
std::uniform_int_distribution<Bit64VECTOR> Bit<128>::distribution(0, LONG_LONG_MAX);

}  // namespace CPU

#endif  // tensorless_BIT128_H
