#ifndef SMANDELBROTR_APP_MANDELBROT_H
#define SMANDELBROTR_APP_MANDELBROT_H

#include "mandelbrot.h"

class AppMandelbrot {
    const static bool USE_REAL_TIME_COMPILE = false;
    AppWindow *mWindow;
    //AppPbo    *mSharedPbo;
    AppGL     *mAppGL;
    // controls for the renderer
    double    mCenterX, mCenterY, mZoom;
    int       mIterMult;
    bool      mDoublePrecision;
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
        // Run CUDA kernel to write to the PBO
        void *devPtr = mAppGL->sharedPbo()->mapGraphicsResource();
        mandelbrot(devPtr,
                   mWindow->width(), mWindow->height(),
                   mAppGL->textureWidth(), mAppGL->textureHeight(),
                   mCenterX, mCenterY,
                   mZoom, mIterMult, mDoublePrecision);
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
