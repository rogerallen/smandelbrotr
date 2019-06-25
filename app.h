#ifndef SMANDELBROTR_APP_H
#define SMANDELBROTR_APP_H

#include "appCUDA.h"
#include "appGL.h"
#include "appMandelbrot.h"
#include "appWindow.h"
#include <GL/glew.h>
#include <SDL.h>
#include <iostream>

class App {

    bool init();
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
    bool mZoomOutMode;
    bool mSaveImage;
    bool mMouseDown;
    double mMouseStartX, mMouseStartY;
    double mMouseX, mMouseY;
    double mCenterStartX, mCenterStartY;

  public:
    App();
    void run();
};
#endif
