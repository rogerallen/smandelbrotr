#include "appCUDA.h"

namespace AppCUDA {
    // set CUDA device to the first one (FIXME should be more flexible)
    // return true on error
    bool setDevice() {
        cudaDeviceProp prop;
        int dev;
        memset(&prop, 0, sizeof(cudaDeviceProp));
        prop.major = 6;
        prop.minor = 0;
        if (cudaChooseDevice(&dev, &prop) != cudaSuccess) {
            std::cerr << "failed to choose device" << std::endl;
            return true;
        }
        std::cout << "cuda chose device" << dev << std::endl;
        if (cudaSetDevice(dev) != cudaSuccess) {
            std::cerr << "failed to set gl device" << std::endl;
            return true;
        }
        return false;
    }
}
