#include "../tensorless/backends/all.h"
#include "../tensorless/types/all.h"

#define print(msg) std::cout<<msg<<"\n";

int main() {
    auto vec = CPU::Fast256();
    vec.set(0, 1);
    vec.set(1, 0.9);
    vec.set(2, 0.8);
    vec.set(3, 0.7);
    vec.set(4, 0.6);
    vec.set(5, 0.5);
    vec.set(6, 0.4);
    vec.set(7, -0.75);
    vec.set(8, -0.5);
    print(vec)
    return 0;
}