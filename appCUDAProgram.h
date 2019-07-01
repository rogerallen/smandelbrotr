#ifndef SMANDELBROTR_APP_CUDA_PROGRAM_H
#define SMANDELBROTR_APP_CUDA_PROGRAM_H

#include <cuda.h>
#include <nvrtc.h>

#include <string>

class AppCUDAProgram {
    CUmodule *mModule;

  public:
    AppCUDAProgram();
    void init(const std::string &fileName);
    CUfunction function(const std::string &kernelName);
};

#endif