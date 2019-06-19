#ifndef SMANDELBROTR_APP_TEXTURE_H
#define SMANDELBROTR_APP_TEXTURE_H

#include <GL/glew.h>

class AppTexture {
    GLuint mId;
    unsigned mWidth, mHeight;
public:
    AppTexture(unsigned width, unsigned height) : mWidth(width), mHeight(height) {
        glGenTextures(1, &mId);
        glBindTexture(GL_TEXTURE_2D, mId);
        // Allocate the texture memory. This will be filled in by the
        // PBO during rendering
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        // Set filter mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // copy from the pixel buffer object to this texture. Since the
    // TexSubImage pixels parameter (final one) is 0, Data is coming
    // from a PBO, not host memory
    void copyFromPbo() {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    }

    unsigned width() { return mWidth; }
    unsigned height() { return mHeight; }

    // bind this texture so OpenGL will use it
    void bind() {
        glBindTexture(GL_TEXTURE_2D, mId);
    }

    // unbind this texture so OpenGL will stop using it
    void unbind() {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

};

#endif
