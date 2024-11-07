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

#ifndef TENSORLESS_TYPES_ALL_H
#define TENSORLESS_TYPES_ALL_H

#include "fixed.h"
#include "scaled.h"
#include "unit4.h"
#include "vec.h"

namespace CPU {
    template<int size> using Fast = CPU::Scaled<CPU::Unit4<CPU::Bit<size>>, CPU::PowerScaleStrategy<-12>>;
    typedef Fast<32> Fast32;
    typedef Fast<64> Fast64;
    typedef Fast<128> Fast128;
    typedef Fast<256> Fast256;
}

#endif  // TENSORLESS_TYPES_ALL_H
