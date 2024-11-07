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

#ifndef tensorless_VECT_H
#define tensorless_VECT_H


#include <iostream>
#include <stdexcept>
#include <random>
#include <array>

template <typename T, int N>
class Vec {
public:
    static Vec<T, N> broadcast(const double& value) {
        Vec<T,N> ret;
        for (int i = 0; i < N; ++i) 
            ret.data[i] = std::move(T::broadcast(value));
        return std::move(ret);
    }

    Vec() noexcept {
        //for (int i = 0; i < N; ++i) 
        //    data[i] = T();
    }

    Vec(const Vec<T, N> &other) noexcept {
        for (int i = 0; i < N; ++i) 
            data[i] = other[i]; 
    }

    Vec(const Vec<T, N> &&other) noexcept: data(std::move(other.data)) {
    }
    
    INLINE Vec<T, N> operator=(const Vec<T, N> &other) {
        data = other.data;
        return *this;
    }
    
    INLINE Vec<T, N> operator=(Vec<T, N> &&other) {
        data = std::move(other.data);
        return *this;
    }

    Vec<T, N>& set(int index, T value) {
        data[index] = value;
        return *this;
    }

    T& operator[](int index) {
        return data[index];
    }

    const T& operator[](int index) const {
        return data[index];
    }

    Vec<T, N> operator+(const Vec<T, N>& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] + other[i];
        return std::move(result);
    }

    Vec<T, N> operator+(const T& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] + other;
        return std::move(result);
    }

    Vec<T, N> operator-(const Vec<T, N>& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] - other[i];
        return std::move(result);
    }

    Vec<T, N> operator*(const Vec<T, N>& other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] * other[i];
        return std::move(result);
    }

    Vec<T, N> operator*(const T &other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] * other;
        return std::move(result);
    }

    Vec<T, N> operator*(const double &other) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] * other;
        return std::move(result);
    }
    
    Vec<T, N> operator>>(int offset) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] >> offset;
        return std::move(result);
    }

    Vec<T, N> operator<<(int offset) const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i] << offset;
        return std::move(result);
    }
    
    Vec<T, N> shallowShiftLeft(int offset) const {
        Vec<T, N> result;
        int Noff = N-offset;
        int i = 0;
        for (;i < Noff; ++i) 
            result.data[i] = data[i+offset];
        for (;i < N; ++i) 
            result.data[i] = T::broadcast(0);
        return std::move(result);
    }

    Vec<T, N> shallowShiftRight(int offset) const {
        Vec<T, N> result;
        int i = 0;
        for(; i < offset; ++i)
            result.data[i] = T::broadcast(0);
        for(; i < N; ++i) 
            result.data[i] = data[i-offset];
        return std::move(result);
    }

    Vec<T, N>& operator+=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] += other[i];
        return *this;
    }

    Vec<T, N>& operator+=(const T &other) {
        for (int i = 0; i < N; ++i) 
            data[i] += other;
        return *this;
    }

    Vec<T, N>& operator-=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] -= other[i];
        return *this;
    }

    Vec<T, N>& operator*=(const Vec<T, N>& other) {
        for (int i = 0; i < N; ++i) 
            data[i] *= other[i];
        return *this;
    }

    Vec<T, N>& operator*=(const T & other) {
        for (int i = 0; i < N; ++i) 
            data[i] *= other;
        return *this;
    }

    Vec<T, N> relu() const {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = data[i].relu();
        return std::move(result);
    }
    
    void selfRelu() {
        for (int i = 0; i < N; ++i) 
            data[i].selfRelu();
    }

    T sum() const {
        T ret(0);
        for (int i = 0; i < N; ++i) 
            ret += data[i];
        return ret;
    }
    
    static Vec<T, N> random() {
        Vec<T, N> result;
        for (int i = 0; i < N; ++i) 
            result.data[i] = std::move(T::random());
        return std::move(result);
    }

    static const int size() {
        return N;
    }


private:
    std::array<T, N> data;
};

#endif  // tensorless_VECT_H
