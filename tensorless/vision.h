#ifndef tensorless_VISION_H
#define tensorless_VISION_H

#include "backends/all.h"
#include "types/unit4.h"
#include "types/scaled.h"
#include "types/fixed.h"
#include "types/vec.h"
#include "layers/all.h"

namespace CPU {
namespace float5 {
    template<int size> using F = Scaled<Unit4<Bit<size>>, PowerScaleStrategy<-12>>;

    template<int width, int height, int channels> 
    using Image = Vec<Vec<F<height>, width>, channels>;

    template<int width, int height, int channelsin, int channelsout, int kernels> 
    using ImConv2d = Conv2d<Image<width,height,channelsin>, 
                            Image<width,height,channelsout>, 
                            channelsin, channelsout, kernels>;

}
namespace float64 {  
    template<int size> using F = Fixed<double, size>;

    template<int width, int height, int channels> 
    using Image = Vec<Vec<F<height>, width>, channels>;

    template<int width, int height, int channelsin, int channelsout, int kernels> 
    using ImConv2d = Conv2d<Image<width,height,channelsin>, 
                            Image<width,height,channelsout>, 
                            channelsin, channelsout, kernels>;
}
}

#endif  // tensorless_VISION_H
