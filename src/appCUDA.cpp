#include "appCUDA.h"

namespace AppCUDA {
// set CUDA device return true on error
bool setDevice(const int cudaDevice)
{
#ifndef NDEBUG
    std::cout << "choosing CUDA device " << cudaDevice << std::endl;
#endif
    if (cudaSetDevice(cudaDevice) != cudaSuccess) {
        std::cerr << "failed to set CUDA device" << std::endl;
        return true;
    }
    return false;
}
} // namespace AppCUDA
