#ifndef SMANDELBROTR_APP_GL_H
#define SMANDELBROTR_APP_GL_H

#include "appGLProgram.h"
#include "appPbo.h"
#include "appTexture.h"
#include "appVerts.h"

#include <GL/glew.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"

#include <string>

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam );

class AppGL {
    AppWindow     *mWindow;
    AppVerts      *mVerts;
    AppGLProgram  *mBasicProg;
    AppTexture    *mSharedTex;
    AppPbo        *mSharedPbo;
    glm::mat4      mCameraToView;
    unsigned char *mPixels;  // storage for data from framebuffer
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
#ifndef NDEBUG
        std::cout << "maxWidth,height = " << maxWidth << "," << maxHeight << std::endl;
#endif
        mWindow = appWindow;
        glClearColor(1.0,1.0,0.5,0.0);
        // Shared CUDA/GL pixel buffer
        mSharedPbo = new AppPbo(maxWidth, maxHeight);
        mSharedTex = new AppTexture(maxWidth, maxHeight);
        mPixels = nullptr;
		// FIXME -- the path to the shaders is fixed at compiletime & used at runtime.  This is too fragile.
        // find absolute paths.
        std::string file_path = __FILE__;
#ifdef WIN32
        const std::string path_sep = "\\";
#else
        const std::string path_sep = "/";
#endif
        std::string file_dir = file_path.substr(0, file_path.rfind(path_sep));
#ifndef NDEBUG
        std::cout << "source directory = " << file_dir << std::endl;
#endif
        mBasicProg = new AppGLProgram(
            file_dir + path_sep + "basic_vert.glsl",
            file_dir + path_sep + "basic_frag.glsl");
        float coords[] = {0.0f, 1.0f, // 8 attrs, 4 verts
                           1.0f, 1.0f,
                           0.0f, 0.0f,
                           1.0f, 0.0f};
        mVerts = new AppVerts(8, 4,
                              mBasicProg->attrPosition(), coords,
                              mBasicProg->attrTexCoords(), coords);
        // During init, enable debug output
        glEnable( GL_DEBUG_OUTPUT );
        glDebugMessageCallback( MessageCallback, 0 );

    }
    AppPbo* sharedPbo() {
        return mSharedPbo;
    }
    unsigned textureWidth() { return mSharedTex->width(); }
    unsigned textureHeight() { return mSharedTex->height(); }
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

            // update on-screen triangles to reflect the aspect ratio change.
            float newPos[] = { 0.0f, ypos, xpos, ypos, 0.0f, 0.0f, xpos, 0.0f };
            mVerts->updatePosition(newPos);

            float wratio = (float) mWindow->width() / mSharedTex->width();
            float hratio = (float) mWindow->height() / mSharedTex->height();
            float newCoords[] = { 0.0f, hratio, wratio, hratio, 0.0f, 0.0f, wratio, 0.0f };
            mVerts->updateTexCoords(newCoords);
        }
        // resize rgb array for saving pixels (FIXME? add to destructor)
        if(mPixels != nullptr) {
            delete[](mPixels);
        }
        mPixels = new unsigned char[mWindow->width() * mWindow->height() * 4];
    }
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        // copy the CUDA-updated pixel buffer to the texture.
        mSharedPbo->bind();
        mSharedTex->bind();
        mSharedTex->copyFromPbo();

        mBasicProg->bind();
        mBasicProg->updateCameraToView(mCameraToView);
        mVerts->bind();
        mVerts->draw();

        mVerts->unbind();
        mSharedPbo->unbind();
        mSharedTex->unbind();
        mBasicProg->unbind();
    }
    unsigned char *readPixels() {
        glReadPixels(0, 0, mWindow->width(), mWindow->height(), GL_RGBA, GL_UNSIGNED_BYTE, mPixels);
        return mPixels;
    }
};

#endif
