#ifndef SMANDELBROTR_APP_MANDELBROT_H
#define SMANDELBROTR_APP_MANDELBROT_H

#include "appCUDAProgram.h"
#include "mandelbrot.h"
#include <string>

class AppMandelbrot {
    AppWindow *mWindow;
    AppGL *mAppGL;
    // controls for the renderer
    double mCenterX, mCenterY, mZoom;
    int mIterMult;
    bool mDoublePrecision;
    std::string kernelFilePath;

  public:
    AppMandelbrot(AppWindow *window, AppGL *appGL, const std::string &shaderPath) : mWindow(window), mAppGL(appGL)
    {
        mCenterX = -0.5;
        mCenterY = 0.0;
        mZoom = 0.5;
        mIterMult = 1;
        mDoublePrecision = false;
#ifdef WIN32
        const std::string pathSep = "\\";
#else
        const std::string pathSep = "/";
#endif
        kernelFilePath = shaderPath + pathSep + "mandelbrotKernels.cu";
    }

    void init()
    {
        mAppGL->sharedPbo()->registerBuffer();
    }

    void render()
    {
        // Run CUDA kernel to write to the PBO
        void *devPtr = mAppGL->sharedPbo()->mapGraphicsResource();
        mandelbrot(devPtr,
                   mWindow->width(), mWindow->height(),
                   mAppGL->textureWidth(), mAppGL->textureHeight(),
                   mCenterX, mCenterY,
                   mZoom, mIterMult, mDoublePrecision,
                   kernelFilePath);
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
    void zoomDiv(double d) { 
        mZoom /= d; 
        if (mZoom < 0.25) {
            mZoom = 0.25;
        }
    }
};

#endif
