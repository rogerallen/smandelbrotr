#ifndef SMANDELBROTR_APP_H
#define SMANDELBROTR_APP_H

#include <SFML/Graphics.hpp>
#include <iostream>
#include "appWindow.h"
#include "appGL.h"

class App {

    bool init();
    void initWindow();
    void loop();
    void update();
    void resize(unsigned width, unsigned height);

    const unsigned WINDOW_START_WIDTH = 800, WINDOW_START_HEIGHT = 800;

    AppWindow *mAppWindow;
    AppGL     *mAppGL;
    sf::RenderWindow *mRenderWindow;
    // TODO Mandelbrot mandelbrot;
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