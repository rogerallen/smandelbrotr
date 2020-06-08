#include "mandelbrot.h"
#include "mandelbrotKernels.h"
#include "appCUDAProgram.h"
#include "cudaErrorCheck.h"

#include <iostream>

// CONFIGURATION SWITCH -- realtime compile or standard CUDA compiled path?
const static bool USE_REAL_TIME_COMPILE = true;

static AppCUDAProgram *gAppCUDAProgram  = nullptr;
static CUfunction gMandelFloatFn        = nullptr;
static CUfunction gMandelDoubleFn       = nullptr;

void mandelbrot(void *devPtr,
                unsigned winWidth, unsigned winHeight,
                unsigned texWidth, unsigned texHeight,
                double centerX, double centerY,
                double zoom, int iterMult, bool doublePrecision,
                const std::string &kernelFilePath) 
{
    const int blockSize = 16; // 256 threads per block
    unsigned mandelWidth = min(winWidth, texWidth);
    unsigned mandelHeight = min(winHeight, texHeight);
    dim3 gridDim(texWidth / blockSize, texHeight / blockSize);
    dim3 blockDim(blockSize, blockSize);
    if(USE_REAL_TIME_COMPILE) {
        if(gAppCUDAProgram == nullptr) {
#ifndef NDEBUG
            std::cout << "Using realtime compiled kernel: " << kernelFilePath << std::endl;
#endif
            gAppCUDAProgram = new AppCUDAProgram();
            gAppCUDAProgram->init(kernelFilePath);
            gMandelFloatFn = gAppCUDAProgram->function("mandel_float");
            gMandelDoubleFn = gAppCUDAProgram->function("mandel_double");
        }
        if(doublePrecision) {
            void *args[] = {
                reinterpret_cast<void *>(&devPtr),
                reinterpret_cast<void *>(&winWidth), 
                reinterpret_cast<void *>(&winHeight),
                reinterpret_cast<void *>(&mandelWidth),
                reinterpret_cast<void *>(&mandelHeight),
                reinterpret_cast<void *>(&centerX),
                reinterpret_cast<void *>(&centerY),
                reinterpret_cast<void *>(&zoom),
                reinterpret_cast<void *>(&iterMult)
            };
            cuErrChk(cuLaunchKernel(gMandelDoubleFn, 
                    gridDim.x, gridDim.y, gridDim.z,    // grid dim 
                    blockDim.x, blockDim.y, blockDim.z, // block dim
                    0, 0,                               // shared mem, stream
                    &args[0],                           // arguments 
                    0));
        }
        else {
            float fCenterX = (float)centerX;
            float fCenterY = (float)centerY;
            float fZoom = (float)zoom;
            void *args[] = {
                reinterpret_cast<void *>(&devPtr),
                reinterpret_cast<void *>(&winWidth), 
                reinterpret_cast<void *>(&winHeight),
                reinterpret_cast<void *>(&mandelWidth),
                reinterpret_cast<void *>(&mandelHeight),
                reinterpret_cast<void *>(&fCenterX),
                reinterpret_cast<void *>(&fCenterY),
                reinterpret_cast<void *>(&fZoom),
                reinterpret_cast<void *>(&iterMult)
            };
            cuErrChk(cuLaunchKernel(gMandelFloatFn, 
                    gridDim.x, gridDim.y, gridDim.z,    // grid dim 
                    blockDim.x, blockDim.y, blockDim.z, // block dim
                    0, 0,                               // shared mem, stream
                    &args[0],                           // arguments 
                    0));
        }

    } else {
        if(doublePrecision) {
            mandel_double<<<gridDim,blockDim>>>((uchar4 *) devPtr,
                                            winWidth, winHeight,
                                            mandelWidth, mandelHeight,
                                            centerX, centerY,
                                            zoom, iterMult);
        }
        else {
            mandel_float<<<gridDim,blockDim>>>((uchar4 *) devPtr,
                                            winWidth, winHeight,
                                            mandelWidth, mandelHeight,
                                            (float)centerX, (float)centerY,
                                            (float)zoom, iterMult);
        }
    }
    cudaErrChk(cudaGetLastError());
    cuErrChk(cuCtxSynchronize());
}
