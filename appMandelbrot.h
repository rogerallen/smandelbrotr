#ifndef SMANDELBROTR_APP_MANDELBROT_H
#define SMANDELBROTR_APP_MANDELBROT_H

#include "cudaErrorCheck.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

class AppMandelbrot {
    const static bool USE_REAL_TIME_COMPILE = false;
    AppWindow *mWindow;
    //AppPbo    *mSharedPbo;
    AppGL     *mAppGL;
    // controls for the renderer
    double    mCenterX, mCenterY, mZoom;
    int       mIterMult;
    bool      mDoublePrecision;
    void mandelbrot(void *devPtr, unsigned winWidth, unsigned winHeight, unsigned texWidth, unsigned texHeight) {
        //const int blockSize = 16; // 256 threads per block
        if(mDoublePrecision) {
            //FIXMEmandel_double <<< dim3(texWidth / blockSize, texHeight / blockSize), dim3(blockSize, blockSize)>>>((uchar4 *) devPtr, winWidth, winHeight, texWidth, texHeight, mCenterX, mCenterY, mZoom, mIterMult);
        }
        else {
            //FIXMEmandel_float<<<dim3(texWidth / blockSize, texHeight / blockSize), dim3(blockSize, blockSize)>>>((uchar4 *) devPtr, winWidth, winHeight, texWidth, texHeight, (float)mCenterX, (float)mCenterY, (float)mZoom, mIterMult);
        }
        cudaErrChk(cudaGetLastError());
        cuErrChk(cuCtxSynchronize());
    }
public:
    AppMandelbrot(AppWindow *window, /*AppPbo *sharedPbo,*/ AppGL *appGL) : mWindow(window),
                                                                        //mSharedPbo(sharedPbo),
                                                                        mAppGL(appGL) {
        mCenterX = -0.5;
        mCenterY = 0.0;
        mZoom = 0.5;
        mIterMult = 1;
        mDoublePrecision = false;
    }
    void init() {
        mAppGL->sharedPbo()->registerBuffer();
        // FIXME real-time compile stuff here
    }
    void render() {
        // Do some CUDA that writes to the pbo
        void *devPtr = mAppGL->sharedPbo()->mapGraphicsResource();
        mandelbrot(devPtr, mWindow->width(), mWindow->height(), mAppGL->textureWidth(), mAppGL->textureHeight());
        mAppGL->sharedPbo()->unmapGraphicsResource();
    }
    void doublePrecision(bool b) { mDoublePrecision = b; }
    void iterMult(int i) { mIterMult = i; }
    void centerX(double d) { mCenterX = d; }
    double centerX() { return mCenterX; }
    void centerY(double d) { mCenterY = d; }
    double centerY() { return mCenterY; }
    double zoom() { return mZoom; }
    void zoomMul(double d) { mZoom *= d; }
    void zoomDiv(double d) { mZoom /= d; }
};

#endif
