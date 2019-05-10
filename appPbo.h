#ifndef SMANDELBROTR_APP_PBO_H
#define SMANDELBROTR_APP_PBO_H

#include "cudaErrorCheck.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

class AppPbo {
    GLuint mId;
    cudaGraphicsResource *mCudaPboHandle; // handle? or pointer?  FIXME
public:
    AppPbo(unsigned width, unsigned height) {
        mCudaPboHandle = nullptr;
        glGenBuffers(1, &mId);
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer
        // Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mId);
        // Allocate data for the buffer. DYNAMIC (modified repeatedly)
        // DRAW (not reading from GL)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);

    };

    void registerBuffer() {
        cudaErrChk(cudaGraphicsGLRegisterBuffer(&mCudaPboHandle, mId, cudaGraphicsRegisterFlagsNone));
    }

    void *mapGraphicsResource()
    {
        cudaErrChk(cudaGraphicsMapResources(1, &mCudaPboHandle));
        void *devPtr = nullptr;
        size_t size;
        cudaErrChk(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, mCudaPboHandle));
        return devPtr;
    }

    void unmapGraphicsResource()
    {
        cudaErrChk(cudaGraphicsUnmapResources(1, &mCudaPboHandle));
    }

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
