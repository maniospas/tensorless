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

    #ifndef tensorless_UNIT4_H
    #define tensorless_UNIT4_H
    #include <cstdlib>
    #include <iostream>

    namespace CPU {

    template <typename Backend>
    class Unit4 {
    public:
        INLINE static Unit4 zero() {return Unit4();}
        
        INLINE Unit4(Unit4&& other) noexcept
            : v0(std::move(other.v0)), v1(std::move(other.v1)), v2(std::move(other.v2)), sgn(std::move(other.sgn)) {
            //other.selfZero();
        }

        INLINE Unit4(const Unit4& other) noexcept
            : v0((other.v0)), v1((other.v1)), v2((other.v2)), sgn((other.sgn)) {
        }

        INLINE Unit4<Backend> operator=(const Unit4<Backend> &other) {
            v0 = other.v0;
            v1 = other.v1;
            v2 = other.v2;
            sgn = other.sgn;
            return *this;
        }
        
        INLINE Unit4<Backend> operator=(Unit4<Backend> &&other) {
            v0 = std::move(other.v0);
            v1 = std::move(other.v1);
            v2 = std::move(other.v2);
            sgn = std::move(other.sgn);
            return *this;
        }
        
        INLINE Unit4<Backend> operator*(const Unit4<Backend> &other) const {
            typename Backend::VECTOR ret0 = v2&other.v0;
            typename Backend::VECTOR ret1 = v2&other.v1;
            typename Backend::VECTOR ret2 = v2&other.v2;
            
            {
                const typename Backend::VECTOR adder0 = v1&other.v1;
                const typename Backend::VECTOR adder1 = v1&other.v2;
                const typename Backend::VECTOR carry0 = ret0&adder0;
                const typename Backend::VECTOR carry1 = (ret1 & adder1) | (carry0 & (ret1 ^ adder1));
                ret0 = ret0^adder0;
                ret1 = ret1^adder1^carry0;
                ret2 = ret2^carry1;
            }

            {
                const typename Backend::VECTOR adder0 = v1&other.v2;
                const typename Backend::VECTOR carry0 = ret0&adder0;
                const typename Backend::VECTOR carry1 = carry0&ret1;
                ret0 = ret0^adder0;
                ret1 = ret1^carry0;
                ret2 = ret2^carry1;
            }

            return Unit4<Backend>(ret0, ret1, ret2, sgn^other.sgn);
        }

        INLINE Unit4<Backend> operator+(const Unit4<Backend> &other) const {
            const typename Backend::VECTOR carry0 = other.v0 & v0;
            const typename Backend::VECTOR carry1 = (v1 & other.v1) | (carry0 & (v1 ^ other.v1));
            const typename Backend::VECTOR carry2 = (v2 & other.v2) | (carry1 & (v2 ^ other.v2));
            const typename Backend::VECTOR finalSign = sgn ^ other.sgn ^ carry2;

            return Unit4<Backend>(other.v0^v0, other.v1^v1^carry0, other.v2^v2^carry1, finalSign);
        }
        
        INLINE const Unit4<Backend>& operator+=(const Unit4<Backend> &other) {
            const typename Backend::VECTOR carry0 = other.v0 & v0;
            const typename Backend::VECTOR carry1 = (v1 & other.v1) | (carry0 & (v1 ^ other.v1));
            const typename Backend::VECTOR carry2 = (v2 & other.v2) | (carry1 & (v2 ^ other.v2));

            v0 = v0^other.v0;
            v1 = v0^other.v1^carry0;
            v2 = v0^other.v2^carry1;
            sgn = sgn ^ other.sgn ^ carry2;
            return *this;
        }

        INLINE Unit4<Backend> operator-(const Unit4<Backend> &other) const {
            return *this+other.complement();
        }

        INLINE Unit4<Backend> complement() const {
            const typename Backend::VECTOR notv0 = ~v0;
            const typename Backend::VECTOR notv1 = ~v1;
            const typename Backend::VECTOR notv2 = ~v2;
            const typename Backend::VECTOR notsgn = ~sgn;

            const typename Backend::VECTOR carry0 = Backend::ONES & notv0;
            const typename Backend::VECTOR carry1 = carry0 & notv1;
            const typename Backend::VECTOR carry2 = carry1 & notv2;
            const typename Backend::VECTOR finalSign = notsgn ^ carry2;

            return Unit4<Backend>(notv0^Backend::ONES, notv1^carry0, notv2^carry1, finalSign);
        }
        
        INLINE Unit4<Backend> operator<<(int other) const {
            return Unit4<Backend>(v0<<other, v1<<other, v2<<other, sgn<<other);
        }

        INLINE Unit4<Backend> operator>>(int other) const {
            return Unit4<Backend>(v0>>other, v1>>other, v2>>other, sgn>>other);
        }

        INLINE Unit4<Backend> relu() const {
            const typename Backend::VECTOR notsgn = ~sgn;
            return Unit4<Backend>(v0&notsgn, v1&notsgn, v2&notsgn, Backend::ZERO);
        }
        
        INLINE void selfRelu() {
            const typename Backend::VECTOR notsgn = ~sgn;
            v0 = v0&notsgn;
            v1 = v1&notsgn;
            v2 = v2&notsgn;
        }

        INLINE double sum() const {
            return _sum(~sgn)+_sum(sgn)-2*Backend::bitcount(sgn);
        }

        INLINE double get(int i) const {
            int ret = Backend::get(v0, i, 1);
            ret += Backend::get(v1, i, 2);
            ret += Backend::get(v2, i, 4);
            if(Backend::get(sgn, i)) 
                return (ret-8)*0.25;
            return ret*0.25;
        }

        INLINE static Unit4<Backend> random() {
            return Unit4<Backend>(Backend::rand(), Backend::rand(), Backend::rand(), Backend::rand());
        }

        INLINE static Unit4<Backend> broadcast(const double &value) {
            double val;
            if(value<0) {
                val = -value-0.25;
                const typename Backend::VECTOR v2 = value<1?Backend::ONES:Backend::ZERO;
                if(val>=1) 
                    val -= 1;
                const typename Backend::VECTOR v1 = val<0.5?Backend::ONES:Backend::ZERO;
                if(val>=0.5) 
                    val -= 0.5;
                const typename Backend::VECTOR v0 = val<0.125?Backend::ONES:Backend::ZERO;
                return Unit4<Backend>(v0, v1, v2, Backend::ONES);
            }
            else {
                val = value;
                const typename Backend::VECTOR v2 = val>=1?Backend::ONES:Backend::ZERO;
                if(val>=1) 
                    val -= 1;
                const typename Backend::VECTOR v1 = val>=0.5?Backend::ONES:Backend::ZERO;
                if(val>=0.5) 
                    val -= 0.5;
                const typename Backend::VECTOR v0 = val>=0.125?Backend::ONES:Backend::ZERO;
                return Unit4<Backend>(v0, v1, v2, Backend::ZERO);
            }
        }
        
        INLINE static Unit4<Backend> broadcast_positive(double val) {
            const typename Backend::VECTOR v2 = val>=1?Backend::ONES:Backend::ZERO;
            if(val>=1) 
                val -= 1;
            const typename Backend::VECTOR v1 = val>=0.5?Backend::ONES:Backend::ZERO;
            if(val>=0.5) 
                val -= 0.5;
            const typename Backend::VECTOR v0 = val>=0.125?Backend::ONES:Backend::ZERO;
            return Unit4<Backend>(v0, v1, v2, Backend::ZERO);
        }

        INLINE void set(int i, const double &value) {
            Backend::set(sgn, i, value<0);
            double val;
            if(value<0) {
                val = -value-0.25;
                Backend::set(v2, i, value<1);
                if(val>=1) 
                    val -= 1;
                Backend::set(v1, i, val<0.5);
                if(val>=0.5) 
                    val -= 0.5;
                Backend::set(v0, i, val<0.125);
            }
            else {
                val = value;
                Backend::set(v2, i, value>=1);
                if(val>=1) 
                    val -= 1;
                Backend::set(v1, i, val>=0.5);
                if(val>=0.5) 
                    val -= 0.5;
                Backend::set(v0, i, val>=0.125);
            }
        }

        Unit4(): v0(Backend::ZERO), v1(Backend::ZERO), v2(Backend::ZERO), sgn(Backend::ZERO) {}

        friend std::ostream& operator<<(std::ostream &os, const Unit4 &si) {
            os << "[" << si.get(0);
            for(int i=1;i<si.size();++i)
                os << "," << si.get(i);
            os << "]";
            return os;
        }
        
        INLINE double size() const {
            return Backend::size;
        }

        INLINE void selfHalf() {
            v0 = v1;
            v1 = v2;
            v2 = sgn;
        }
        INLINE void selfZero() {
            v0 = Backend::ZERO;
            v1 = Backend::ZERO;
            v2 = Backend::ZERO;
            sgn = Backend::ZERO;
        }
        INLINE void selfDouble() {
            v2 = v1;
            v1 = v0;
            v0 = Backend::ZERO;
        }
        INLINE const double absmax() const {
            return std::max(_absmax(~sgn), complement()._absmax(sgn));
        }
        static const int params;
    private:
        typename Backend::VECTOR v0;
        typename Backend::VECTOR v1;
        typename Backend::VECTOR v2;
        typename Backend::VECTOR sgn;

        Unit4(const typename Backend::VECTOR &v0, 
            const typename Backend::VECTOR &v1, 
            const typename Backend::VECTOR &v2, 
            const typename Backend::VECTOR &sgn) : v0(v0), v1(v1), v2(v2), sgn(sgn) {}

        INLINE const double _sum(const typename Backend::VECTOR& mask) const {
            int ret = Backend::bitcount(v0 & mask);
            ret += Backend::bitcount(v1 & mask)*2;
            ret += Backend::bitcount(v2 & mask)*4;
            return ret*0.25;
        }
        
        INLINE const double _absmax(const typename Backend::VECTOR& mask) const {
            int ret = 0;
            const typename Backend::VECTOR ret2 = v2 & mask;
            typename Backend::VECTOR ret1 = v1 & mask;
            typename Backend::VECTOR ret0 = v0 & mask;
            if(Backend::any(ret2)) {
                ret += 4;
                ret1 &= ret2;
                ret0 &= ret2;
            }
            if(Backend::any(ret1)) {
                ret += 2;
                ret0 &= ret1;
            }
            if(Backend::any(ret0))
                ret += 1;
            return ret*0.25;
        }
    };

    template <typename Backend>
    const int Unit4<Backend>::params = 4;


    }  // namespace CPU
    #endif  // tensorless_UNIT4_H
