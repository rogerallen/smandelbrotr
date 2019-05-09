#ifndef SMANDELBROTR_APP_WINDOW_H
#define SMANDELBROTR_APP_WINDOW_H

#include <iostream>

class AppWindow {
    unsigned int mWidth, mHeight;
    bool mResized;

public:
    AppWindow(unsigned int width, unsigned int height) :
    mWidth{width}, mHeight{height} {
        mResized = true;
    }
    void width(unsigned int width) { mWidth = width; mResized = true;}
    unsigned int width() { return mWidth; }
    void height(unsigned int height) { mHeight = height; mResized = true;}
    unsigned int height() { return mHeight; }
    bool resized() { return mResized; }
    void resizeHandled() { mResized = false; }
};
#endif
