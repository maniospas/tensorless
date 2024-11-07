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


#ifndef TENSORLESS_CONV_H
#define TENSORLESS_CONV_H

#include <iostream>
#include <vector>
#include "neural.h"
#include <cmath>
#include <random>

namespace CPU {

template <typename Tensor, int ins, int receptive>
class Conv: public Neural<Tensor> {
private:
    int shifts[receptive];
    double weights[receptive];
    double bias;

public:
    Conv() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for(int i=0;i<receptive;++i) {
            shifts[i] = i-receptive/2-1;
            weights[i] = dis(gen);
        }
    }

    virtual std::string describe() const {
        std::string description;
        description += "Conv";
        description += "\n  Inputs   " + std::to_string(ins);
        description += "\n  Window   " + std::to_string(receptive);
        description += "\n  Params   " + std::to_string(receptive+1);
        description += "\n";
        return description;
    }

    virtual const Tensor forward(const Tensor& input) {
        if(input.size()!=ins)
            throw std::logic_error("Mismatching sizes in Conv: input size "+std::to_string(input.size())+" but expected "+std::to_string(ins));
        Tensor out = Tensor::broadcast(bias);
        for (int i=0;i<receptive;++i) {
            int shift = shifts[i];
            double weight = weights[i];
            out += shift>0?((input<<shift)*weight):(shift==0?(input*weight):((input>>-shift)*weight));
        }
        out.selfRelu();
        return std::move(out);
    }

    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) {
        return Tensor::random();// std::error;
    }
};

}
#endif  // TENSORLESS_CONV_H