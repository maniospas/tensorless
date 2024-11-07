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

#ifndef tensorless_SCALED_H
#define tensorless_SCALED_H
#include <cstdlib>
#include <iostream>

#define INLINE inline __attribute__((always_inline)) 


namespace CPU {

template <int default_power>
class PowerScaleStrategy {
public:
    INLINE static double init() {
        if constexpr (default_power<0)
            return 1./(1<<-default_power);
        else 
            return (1<<default_power);
    }
    
    template <typename Unit>
    INLINE static void standardize(Unit &unit, double &scale) {
        double val = unit.absmax();
        if(val>1) {
            while(val>1) {
                unit.selfHalf();
                scale *= 2;
                val /= 2;
            }
        }
        else if(val){
            while(val<0.5) {
                val *= 0.5;
                unit.selfDouble();
                scale /= 2;
            }
        }
    }
};

template <typename Unit, typename ScaleStrategy>
class Scaled {
public:
    INLINE static Scaled<Unit, ScaleStrategy> random() {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(Unit::random()), 1));
    }
    INLINE static Scaled<Unit, ScaleStrategy> broadcast(double value) {
        double scale = ScaleStrategy::init();
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(Unit::broadcast(value/scale)), scale));
    }
    
    template <typename OtherScaleStrategy>
    INLINE Scaled<Unit, ScaleStrategy> operator+(const Scaled<Unit, OtherScaleStrategy> &other) const {
        double otherscale = other.getScale();
        if(scale>otherscale) 
            return std::move(Scaled<Unit, ScaleStrategy>(std::move(other.getUnit()*Unit::broadcast_positive(otherscale/scale)+unit), scale).standardize());
        else if(scale<otherscale)
            return std::move(Scaled<Unit, ScaleStrategy>(std::move((unit*Unit::broadcast_positive(scale/otherscale))+other.getUnit()), otherscale).standardize());
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit+other.getUnit()), scale).standardize());
    }
    
    template <typename OtherScaleStrategy>
    INLINE const Scaled<Unit, ScaleStrategy>& operator+=(const Scaled<Unit, OtherScaleStrategy> &other) {
        double otherscale = other.getScale();
        if(scale>otherscale) {
            unit += other.getUnit()*Unit::broadcast_positive(otherscale/scale);
            standardize();
        }
        else if(scale<otherscale) {
            unit = unit*Unit::broadcast_positive(scale/otherscale);
            unit += other.getUnit();
            scale = otherscale;
            standardize();
        }
        else {
            unit += other.getUnit();
            standardize();
        }
        return *this;
    }

    template <typename OtherScaleStrategy>
    INLINE Scaled<Unit, ScaleStrategy> operator-(const Scaled<Unit, OtherScaleStrategy> &other) const {
        double otherscale = other.getScale();
        if(scale>otherscale) 
            return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit-other.getUnit()*Unit::broadcast(otherscale/scale)), scale).standardize());
        else if(scale<otherscale)
            return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit*Unit::broadcast(scale/otherscale)-other.getUnit()), otherscale).standardize());
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit-other.getUnit()), scale).standardize());
    }
    
    template <typename OtherScaleStrategy>
    INLINE Scaled<Unit, ScaleStrategy> operator*(const Scaled<Unit, OtherScaleStrategy> &other) const {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit*other.getUnit()), scale*other.getScale()).standardize());
    }
    
    INLINE Scaled<Unit, ScaleStrategy> operator*(const double &other) const {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit), scale*other));
    }

    INLINE Scaled<Unit, ScaleStrategy> operator<<(int other) const {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit<<other), scale));
    }

    INLINE Scaled<Unit, ScaleStrategy> operator>>(int other) const {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit>>other), scale));
    }

    INLINE Scaled<Unit, ScaleStrategy> relu() const {
        return std::move(Scaled<Unit, ScaleStrategy>(std::move(unit.relu()), scale));
    }

    INLINE void selfRelu() {
        unit.selfRelu();
    }
    
    template <typename OtherScaleStrategy>
    INLINE Scaled<Unit, ScaleStrategy>& operator=(const Scaled<Unit, OtherScaleStrategy> &other) {
        if (this != &other) {
            unit = other.unit;
            scale = other.scale;
        }
        return *this;
    }

    Scaled(): unit(Unit()), scale(ScaleStrategy::init()) {}
    friend std::ostream& operator<<(std::ostream &os, const Scaled<Unit, ScaleStrategy> &si) {
        os << "[" << si.get(0);
        for(int i=1;i<si.size();++i)
            os << "," << si.get(i);
        os << "]";
        return os;
    }

    INLINE void set(int i, const double &value) {
        //if(value>scale || value<-scale || scale==0) 
        //    throw std::logic_error("Number out of bounds: "+std::to_string(value)); 
        double ratio = value/scale;
        double absratio = ratio;
        if(ratio<0)
            absratio = -ratio;
        if(absratio>(1<<(Unit::params-1))) {
            unit.selfZero();
            ratio = 1;
            scale = value;
        }
        else {
            while(absratio>1) {
                unit.selfHalf();
                absratio *= 0.5;
                ratio *= 0.5;
                scale *= 2;
            }
        }
        unit.set(i, ratio);
    }

    INLINE double get(int i) const {
        return unit.get(i)*scale;
    }

    INLINE double sum() const {
        return unit.sum()*scale;
    }

    INLINE int size() const {return unit.size();}

    INLINE const Unit& getUnit() const {
        return unit;
    }

    INLINE const double getScale() const {
        return scale;
    }

    INLINE double absmax() const {
        return unit.absmax()*scale;
    }

    INLINE Scaled(const Scaled& other) noexcept
        : unit(other.unit), scale((other.scale)) {
        // Optionally, reset other to a default state
        //other.scale = ScaleStrategy::init();
    }

    INLINE Scaled(Scaled&& other) noexcept
        : unit(std::move(other.unit)), scale(std::move(other.scale)) {
        // Optionally, reset other to a default state
        //other.scale = ScaleStrategy::init();
    }
    
    INLINE Scaled operator=(const Scaled &other) {
        scale = other.scale;
        unit = other.unit;
        return *this;
    }
    
    INLINE Scaled operator=(Scaled &&other) {
        scale = other.scale;
        unit = std::move(other.unit);
        return *this;
    }

private:
    INLINE const Scaled<Unit, ScaleStrategy>& standardize() {
        ScaleStrategy::standardize(unit, scale);
        return *this;
    }
    double scale;
    Unit unit;
    Scaled(const Unit& unit, const double &scale): unit(unit), scale(scale) {}
    Scaled(Unit&& unit, const double &scale): unit(std::move(unit)), scale(scale) {}
};

}  // namespace CPU

#endif  // tensorless_SCALED_H
