#include "mandelbrot.h"
#include "mandelbrot_kernels.h"
#include "cudaErrorCheck.h"

void mandelbrot(void *devPtr,
                unsigned winWidth, unsigned winHeight,
                unsigned texWidth, unsigned texHeight,
                double centerX, double centerY,
                double zoom, int iterMult, bool doublePrecision) {
    const int blockSize = 16; // 256 threads per block
    unsigned mandelWidth = min(winWidth, texWidth);
    unsigned mandelHeight = min(winHeight, texHeight);
    if(doublePrecision) {
        mandel_double<<<dim3(texWidth / blockSize, texHeight / blockSize),
          dim3(blockSize, blockSize)>>>((uchar4 *) devPtr,
                                        winWidth, winHeight,
                                        mandelWidth, mandelHeight,
                                        centerX, centerY,
                                        zoom, iterMult);
    }
    else {
        mandel_float<<<dim3(texWidth / blockSize, texHeight / blockSize),
          dim3(blockSize, blockSize)>>>((uchar4 *) devPtr,
                                        winWidth, winHeight,
                                        mandelWidth, mandelHeight,
                                        (float)centerX, (float)centerY,
                                        (float)zoom, iterMult);
    }
    cudaErrChk(cudaGetLastError());
    cuErrChk(cuCtxSynchronize());
}
