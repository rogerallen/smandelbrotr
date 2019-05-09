#ifndef SMANDELBROTR_APP_GL_H
#define SMANDELBROTR_APP_GL_H

#include <SFML/OpenGL.hpp>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "appPbo.h"

class AppGL {
    AppWindow *mWindow;
    //AppVerts verts;
    //AppProgram basicProg;
    //AppTexture sharedTex;
    AppPbo *sharedPbo;
    glm::mat4 mCameraToView;
public:
    AppGL(AppWindow *appWindow, unsigned maxWidth, unsigned maxHeight) {
        mWindow = appWindow;
        glClearColor(1.0,1.0,0.0,0.0); // FIXME
        // Shared CUDA/GL pixel buffer
        sharedPbo = new AppPbo(maxWidth, maxHeight);
        // FIXME rest of state
    }
    void handleResize() {
        glViewport(0, 0, mWindow->width(), mWindow->height());
        if(mWindow->resized()) {
            mWindow->resizeHandled();
            // anchor viewport to upper left corner (0, 0) to match the anchor on
            // the sharedTexture surface. See picture above.
            float xpos = 1.0f, ypos = 1.0f;
            if (mWindow->width() >= mWindow->height()) {
                ypos = (float) mWindow->height() / (float) mWindow->width();
            } else {
                xpos = (float) mWindow->width() / (float) mWindow->height();
            }
            mCameraToView = glm::ortho(0.0f, xpos, ypos, 0.0f);
            // FIXME more to do
        }
    }
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        // FIXME everything else
    }
};

#endif
