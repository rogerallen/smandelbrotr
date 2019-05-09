#ifndef SMANDELBROTR_WINDOW_H
#define SMANDELBROTR_WINDOW_H

#include <iostream>

class Window {
    unsigned int mWidth, mHeight;
    bool mResized;

public:
    Window(unsigned int width, unsigned int height) :
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
