#include <iostream>
#include <chrono>

// tensorless includes (assuming you have the tensorless library properly set up)
#include "../tensorless/backends/all.h"
#include "../tensorless/types/unit4.h"
#include "../tensorless/types/scaled.h"

using namespace CPU;

#define SIZE 128
typedef Scaled<Unit4<Bit<SIZE>>, PowerScaleStrategy<-12>> F;

void benchmark_tensorless(int N) {
    F data1 = F(); 
    F data2 = F(); 
    data1.set(0, 1);
    data1.set(1, 0.5);
    data1.set(2, 0.25);
    data1.set(3, 1.25);
    data2.set(0, 0.5);
    data2.set(1, 0.5);
    data2.set(2, 0.5);
    data2.set(3, 0.5);
    double sum = 0;

    std::cout << data1 << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    for(int repeat=0;repeat<N;++repeat) {
        sum += (data1*0.5+(data2<<1)*2).sum();
        sum += (data1*0.5+(data2<<1)*2).sum();
        sum += (data1*0.5+(data2<<1)*2).sum();
        sum += (data1*0.5+(data2<<1)*2).sum();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Size " << data1.size() << "\n";
    std::cout << "Dot  " << sum/N << "\n";
    std::cout << "Time " << duration.count() << " sec\n";
}

void benchmark_array(int N) {
    const int size = SIZE;
    double data1[size];
    double data2[size];
    double result[size];
    for (int i=0;i<size;++i) {
        data1[i] = 0;
        data2[i] = 0;
    }
    data1[0] = 1;
    data1[1] = -0.5;
    data1[2] = -0.25;
    data1[3] = -1.5;
    data2[0] = 0.5;
    data2[1] = 0.5;
    data2[2] = 0.5;
    double sum = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(int repeat=0;repeat<N;++repeat) {
        for (int i=1;i<size;++i) {
            result[i] = data1[i]*0.5+data2[i-1]*2;
            result[i] = data1[i]*0.5+data2[i-1]*2;
            result[i] = data1[i]*0.5+data2[i-1]*2;
            result[i] = data1[i]*0.5+data2[i-1]*2;
            sum += result[i];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Size " << size << "\n";
    std::cout << "Dot  " << sum/N << "\n";
    std::cout << "Time " << duration.count() << " sec\n";
}

int main() {
    std::cout << "===== Benchmarking tensorless\n";
    benchmark_tensorless(10000000);
    
    std::cout << "\n===== Benchmarking Array\n";
    benchmark_array(10000000);
    
    return 0;
}
