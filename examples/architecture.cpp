#include "../tensorless/backends/all.h"
#include "../tensorless/types/unit4.h"
#include "../tensorless/types/scaled.h"
#include "../tensorless/types/fixed.h"
#include "../tensorless/types/vec.h"
#include "../tensorless/layers/all.h"
#include <memory>
#include <chrono>

using namespace CPU;

#define NOW std::chrono::high_resolution_clock::now()


//template<int size> using F = Scaled<Unit4<Bit<size>>, PowerScaleStrategy<-12>>;
template<int size> using F = Scaled<Unit4<Bit<size>>, PowerScaleStrategy<-12>>;
//template<int size> using F = Fixed<double, size>;

typedef F<32> F32;
typedef F<64> F64;
typedef F<128> F128;
typedef F<256> F256;

int main() {
    auto layer1 = Conv<F256, 256, 7>();
    auto layer2 = Conv<F256, 256, 7>();
    auto in = F256::random();

    auto start = NOW;
    double s = 0;
    for(long epoch=0;epoch<1000000;++epoch) {
        auto out1 = layer1.forward(in);
        auto out2 = layer2.forward(out1);
        s += out2.sum();
    }

    std::cout << s << " "<<((std::chrono::duration<double>)(NOW - start)).count() << " sec" << std::endl;

}
