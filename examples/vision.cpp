
#include <memory>
#include <chrono>
#include "../tensorless/vision.h"

// g++ examples/vision.cpp -o test -O3 -std=c++20 -msse4.2 -mavx2
// 
// Linux instructions for increased stack size (high stacks crash for larger than 64 IMSIZE)
// ulimit -s           >>> 8192 (this is the stack size beforehand)
// ulimit -s 4194304   >>> increases stack size for complicated parameters

#define NOW std::chrono::high_resolution_clock::now()
using namespace CPU::float5;

int main() {
    const int IMSIZE = 256;
    const int kernel_size = 3;
    auto in = Image<IMSIZE, IMSIZE, 3>::random();
    
    std::cout << "in dims ("<<in.size()<<", "<<in[0].size()<<", "<<in[0][0].size()<<")\n";
    
    auto layer1 = ImConv2d<IMSIZE, IMSIZE, 3, 16, kernel_size>();
    auto layer2 = ImConv2d<IMSIZE, IMSIZE, 16, 32, kernel_size>();
    //auto layer3 = ImConv2d<IMSIZE, IMSIZE, 32, 64, kernel_size>();
    //auto layer4 = ImConv2d<IMSIZE, IMSIZE, 64, 128, kernel_size>();
    
    auto start = NOW;
    auto out1 = layer1.forward(in);
    out1.selfRelu();
    auto out2 = layer2.forward(out1);
    out2.selfRelu();
    //auto out3 = layer3.forward(out2);
    //out3.selfRelu();
    //auto out4 = layer4.forward(out3);
    //out4.selfRelu();

    std::cout << ((std::chrono::duration<double>)(NOW - start)).count() << " sec" << std::endl;

    std::cout << "out1 dims ("<<out1.size()<<", "<<out1[0].size()<<", "<<out1[0][0].size()<<")\n";
    std::cout << "out2 dims ("<<out2.size()<<", "<<out2[0].size()<<", "<<out2[0][0].size()<<")\n";
    //std::cout << "out3 dims ("<<out3.size()<<", "<<out3[0].size()<<", "<<out3[0][0].size()<<")\n";
    //std::cout << "out4 dims ("<<out4.size()<<", "<<out4[0].size()<<", "<<out4[0][0].size()<<")\n";

}
