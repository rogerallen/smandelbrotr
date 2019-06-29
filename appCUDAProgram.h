#ifndef SMANDELBROTR_APP_CUDA_PROGRAM_H
#define SMANDELBROTR_APP_CUDA_PROGRAM_H

#include <nvrtc.h>
#include <cuda.h>

#include <string>

class AppCUDAProgram {
    CUmodule *mModule;
public:
    AppCUDAProgram();
    void init(std::string fileName);  // FIXME reference to string
    CUfunction function(std::string kernelName);
};

#endif