#ifndef SMANDELBROTR_APP_H
#define SMANDELBROTR_APP_H

#include <GL/glew.h>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>
#include "appWindow.h"
#include "appGL.h"
#include "appCUDA.h"
#include "appMandelbrot.h"

class App {

    bool init();
    void initWindow();
    void loop();
    void update();
    void resize(unsigned width, unsigned height);

    const unsigned WINDOW_START_WIDTH = 800, WINDOW_START_HEIGHT = 800;

    AppWindow        *mAppWindow;
    AppGL            *mAppGL;
    AppMandelbrot    *mAppMandelbrot;
    sf::RenderWindow *mRenderWindow;
    bool mSwitchFullscreen;
    bool mIsFullscreen;
    int mMonitorWidth, mMonitorHeight;
    int mPrevWindowWidth, mPrevWindowHeight;
    bool mZoomOutMode;
    bool mSaveImage;
    bool mMouseDown;
    double mMouseStartX, mMouseStartY, mCenterStartX, mCenterStartY;

public:
    App();
    void run();

};
#endif
