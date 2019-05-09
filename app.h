#ifndef SMANDELBROTR_APP_H
#define SMANDELBROTR_APP_H

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
#include "window.h"

class App {

    bool init();
    void initWindow();
    void loop();
    void update();
    void resize(unsigned int width, unsigned int height);

    const unsigned int WINDOW_START_WIDTH = 800, WINDOW_START_HEIGHT = 800;

    Window *mWindow;
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
