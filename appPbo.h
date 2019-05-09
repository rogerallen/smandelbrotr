#ifndef SMANDELBROTR_APP_PBO_H
#define SMANDELBROTR_APP_PBO_H

//#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

class AppPbo {
    GLuint mId;
    //cudaGraphicsResource cudaPBOHandle;
public:
    AppPbo(unsigned width, unsigned height) {
        //mCudaPboHandle = nullptr;
        glGenBuffers(1, &mId);
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer
        // Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mId);
        // Allocate data for the buffer. DYNAMIC (modified repeatedly)
        // DRAW (not reading from GL)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
    };
    // FIXME boolean registerBuffer();
    // FIXME CUdeviceptr mapGraphicsResource();
    // FIXME void unmapGraphicsResource();

    // bind the PBO for OpenGL's use.
    void bind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mId);
    }

    // unbind the PBO so OpenGL does not use it.
    void unbind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
};

#endif
