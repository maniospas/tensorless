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

#ifndef FIXED_H
#define FIXED_H


#include <iostream>
#include <stdexcept>
#include <random>
#include <array>

template <typename T, int N>
class Fixed {
public:
    static Fixed<T, N> broadcast(const T& value) {
        Fixed<T,N> ret;
        for (int i = 0; i < N; ++i) 
            ret.data[i] = value;
        return ret;
    }
    Fixed() noexcept {
        //for (int i = 0; i < N; ++i) 
        //    data[i] = T(0);
    }
    
    Fixed(const Fixed<T, N> &other) {
        for (int i = 0; i < N; ++i) 
            data[i] = other[i]; 
    }

    Fixed(Fixed<T, N> &&other) noexcept: data(std::move(other.data)) {
    }

    Fixed<T, N>& set(int index, T value) {
        data[index] = value;
        return *this;
    }

    T& operator[](int index) {
        return data[index];
    }

    const T& operator[](int index) const {
        return data[index];
    }

    Fixed<T, N> operator+(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] + other[i];
        return std::move(result);
    }

    Fixed<T, N> operator+(const T& other) const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] + other;
        return std::move(result);
    }

    Fixed<T, N> operator-(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] - other[i];
        return std::move(result);
    }

    Fixed<T, N> operator*(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] * other[i];
        return std::move(result);
    }

    Fixed<T, N> operator*(const T &other) const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] * other;
        return std::move(result);
    }
    
    Fixed<T, N> operator>>(int offset) const {
        Fixed<T, N> result;
        int Noff = N-offset;
        for (int i = 0; i < Noff; ++i) 
            result.data[i] = data[i+offset];
        return std::move(result);
    }

    Fixed<T, N> operator<<(int offset) const {
        Fixed<T, N> result;
        for (int i = offset; i < N; ++i) 
            result.data[i] = data[i-offset];
        return std::move(result);
    }

    Fixed<T, N>& operator+=(const Fixed<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] += other[i];
        return *this;
    }

    Fixed<T, N>& operator+=(const T &other) {
        for (int i = 0; i < N; ++i) 
            data[i] += other;
        return *this;
    }

    Fixed<T, N>& operator-=(const Fixed<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] -= other[i];
        return *this;
    }

    Fixed<T, N>& operator*=(const Fixed<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] *= other[i];
        return *this;
    }

    Fixed<T, N>& operator*=(const T & other) {
        for (int i = 0; i < N; ++i) 
            data[i] *= other;
        return *this;
    }

    Fixed<T, N> relu() const {
        Fixed<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i]>0?data[i]:0;
        return std::move(result);
    }
    
    void selfRelu() {
        for (int i = 0; i < N; ++i) 
            if(data[i]<0)
                data[i] = 0;
    }

    T sum() const {
        T ret(0);
        for (int i = 0; i < N; ++i) 
            ret += data[i];
        return ret;
    }
    
    static Fixed<T, N> random() {
        Fixed<T, N> result;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            result.data[i] = dis(gen);
        }

        return result;
    }
    
    static const int num_params() {
        return N;
    }

    static const size_t num_bits() {
        return N*sizeof(T)*8;
    }

    static const int size() {
        return N;
    }
    
    INLINE Fixed<T, N> operator=(const Fixed<T, N> &other) {
        data = other.data;
        return *this;
    }
    
    INLINE Fixed<T, N> operator=(Fixed<T, N> &&other) {
        data = std::move(other.data);
        return *this;
    }

private:
    std::array<T, N> data;
};

#endif  // FIXED_H
