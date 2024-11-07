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


#ifndef TENSORLESS_DENSE_H
#define TENSORLESS_DENSE_H

#include <iostream>
#include <vector>
#include "neural.h"
#include <cmath>

namespace CPU {

template <typename Tensor, int ins, int outs>
class Dense: public Neural<Tensor> {
private:
    Tensor weights[outs];
    double biases[outs];
    bool activations[outs];

public:
    Dense() {
        for (int i=0; i<outs;++i) 
            weights[i] = Tensor::random();
    }

    virtual std::string describe() const {
        std::string description;
        description += "Dense";
        description += "\n  Inputs   " + std::to_string(ins);
        description += "\n  Outputs  " + std::to_string(outs);
        description += "\n";
        return description;
    }

    virtual Tensor forward(const Tensor& input) {
        Tensor out = Tensor();
        Tensor weightedIns;
        double sum;
        bool activation;
        for (int i=0;i<outs;++i) {
            weightedIns = input*weights[i]; 
            sum = weightedIns.sum() + biases[i];
            activation = sum>0;
            activations[i] = activation;
            if(activation) 
                out.set(i, sum);
        }
        return out;
    }

    virtual Tensor backward(const Tensor &error, Optimizer<Tensor> &optimizer) {
        return Tensor::random();
    }
};

}
#endif  // TENSORLESS_DENSE_H