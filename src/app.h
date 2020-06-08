#ifndef SMANDELBROTR_APP_H
#define SMANDELBROTR_APP_H

#include "appCUDA.h"
#include "appGL.h"
#include "appMandelbrot.h"
#include "appWindow.h"
#include <GL/glew.h>
#include <SDL.h>
#include <algorithm>
#include <iostream>
#include <string>

class App {

    bool init(const int cudaDevice, const std::string &shaderPath);
    bool initWindow();
    void loop();
    void update();
    void cleanup();
    void resize(unsigned width, unsigned height);

    AppWindow *mAppWindow;
    AppGL *mAppGL;
    AppMandelbrot *mAppMandelbrot;
    SDL_Window *mSDLWindow;
    SDL_GLContext mSDLGLContext;
    bool mSwitchFullscreen;
    bool mIsFullscreen;
    int mMonitorWidth, mMonitorHeight;
    int mPrevWindowWidth, mPrevWindowHeight;
    int mPrevWindowX, mPrevWindowY;
    bool mZoomOutMode;
    bool mSaveImage;
    bool mMouseDown;
    double mMouseStartX, mMouseStartY;
    double mMouseX, mMouseY;
    double mCenterStartX, mCenterStartY;
    bool mReverseZoomMode;

  public:
    App();
    void run(const int cudaDevice, const std::string &shaderPath);
};
#endif
