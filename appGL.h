#ifndef SMANDELBROTR_APP_GL_H
#define SMANDELBROTR_APP_GL_H

#include "appGLProgram.h"
#include "appPbo.h"
#include "appTexture.h"
#include "appVerts.h"

#include <SFML/OpenGL.hpp>
#include "glm/glm.hpp"
#include "glm/ext.hpp"

class AppGL {
    AppWindow    *mWindow;
    AppVerts     *mVerts;
    AppGLProgram *mBasicProg;
    AppTexture   *mSharedTex;
    AppPbo       *mSharedPbo;
    glm::mat4    mCameraToView;
public:
    // OpenGL-related code for JMandelbrot.  This is a static class to
    // reflect a single GL context.
    //
    // - Creates a fullscreen quad (2 triangles)
    // - The quad has x,y and s,t coordinates & the upper-left corner
    //   is always 0,0 for both.
    // - When resized,
    //   - the x,y for the larger axis ranges from 0-1 and the shorter
    //     axis 0-ratio where ratio is < 1.0
    //   - the s,t is a ratio of the window size to the shared CUDA/GL
    //     texture size.
    //   - the shared CUDA/GL texture size should be set to the maximum
    //     size you expect. (Monitor width/height)
    // - These values are updated inside the vertex buffer.
    //
    // t y
    // 0 0 C--*--D triangle_strip ABCD
    //     |\....|
    //     |.\...|
    //     *..*..*
    //     |...\.|
    //     |....\|
    // 1 1 A--*--B
    //     0     1 x position coords
    //     0     1 s texture coords
    //
    AppGL(AppWindow *appWindow, unsigned maxWidth, unsigned maxHeight) {
        mWindow = appWindow;
        glClearColor(1.0,1.0,0.5,0.0);
        // Shared CUDA/GL pixel buffer
        mSharedPbo = new AppPbo(maxWidth, maxHeight);
        mSharedTex = new AppTexture(maxWidth, maxHeight);
        mBasicProg = new AppGLProgram("basic_vert.glsl", "basic_frag.glsl");
        float coords[] = {0.0f, 1.0f, // 8 attrs, 4 verts
                           1.0f, 1.0f,
                           0.0f, 0.0f,
                           1.0f, 0.0f};
        mVerts = new AppVerts(8, 4,
                              mBasicProg->attrPosition(), coords,
                              mBasicProg->attrTexCoords(), coords);
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
