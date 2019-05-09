#ifndef SMANDELBROTR_APP_GL_H
#define SMANDELBROTR_APP_GL_H

#include <SFML/OpenGL.hpp>

class AppGL {
    AppWindow *mWindow;
    //AppVerts verts;
    //AppProgram basicProg;
    //AppTexture sharedTex;
    //AppPbo sharedPbo;
    //Matrix4f cameraToView = new Matrix4f();
public:
    AppGL(AppWindow *appWindow, unsigned maxWidth, unsigned maxHeight) {
        mWindow = appWindow;
        glClearColor(1.0,1.0,0.0,0.0); // FIXME
        // FIXME rest of state
    }
    void handleResize() {
        glViewport(0, 0, mWindow->width(), mWindow->height());
        if(mWindow->resized()) {
            mWindow->resizeHandled();
            // FIXME  cameratoview fixup
        }
    }
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        // FIXME everything else
    }
};

#endif
