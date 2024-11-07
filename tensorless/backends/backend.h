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

#ifndef tensorless_BACKEND_H
#define tensorless_BACKEND_H

#define INLINE inline __attribute__((always_inline)) 

namespace CPU {
    std::random_device rd;
    
    template<int precision>
    class Bit;

}  // namespace CPU

#endif  // tensorless_BACKEND_H