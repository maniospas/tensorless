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


#ifndef TENSORLESS_CONV2D_H
#define TENSORLESS_CONV2D_H

#include <iostream>
#include <vector>
#include "neural.h"
#include <cmath>
#include <random>

namespace CPU {

template <typename TensorIn, typename TensorOut, int ins, int outs, int receptive>
class Conv2d {
private:
    int shifts[receptive];
    double weights[ins][outs][receptive][receptive];
    double bias;

public:
    Conv2d() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for(int i=0;i<receptive;++i) {
            shifts[i] = i-receptive/2-1;
            for(int j=0;j<receptive;++j) 
                for(int in=0;in<ins;++in)
                    for(int out=0;out<outs;++out) {
                        weights[in][out][i][j] = dis(gen);
                        //std::cout << in << ", "<<out<<", "<<i<<", "<<j<<"\n";
                    }
        }
    }

    const TensorOut forward(const TensorIn& input) {
        if(input.size()!=ins)
            throw std::logic_error("Mismatching sizes in Conv: input size "+std::to_string(input.size())+" but expected "+std::to_string(ins));
        TensorOut out = TensorOut::broadcast(bias);
        for(int cout=0;cout<outs;cout++) {
            auto ou = out[cout];
            for(int cin=0;cin<ins;cin++) {
                auto inp = input[cin];
                for (int i=0;i<receptive;++i) {
                    int shifti = shifts[i];
                    auto in = shifti>0?(inp.shallowShiftLeft(shifti)):(shifti?inp:inp.shallowShiftRight(-shifti));
                    for(int j=0;j<receptive;j++) {
                        int shiftj = shifts[j];
                        double weight = weights[cin][cout][i][j];
                        ou += shiftj>0?((in<<shiftj)*weight):(shiftj?((in>>-shiftj)*weight):in*weight);
                    }
                }
            }
        }
        return std::move(out);
    }
};

}
#endif  // TENSORLESS_CONV2D_H