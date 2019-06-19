#ifndef SMANDELBROTR_APP_PBO_H
#define SMANDELBROTR_APP_PBO_H

#include "cudaErrorCheck.h"
#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <GL/glew.h>

class AppPbo {
    GLuint mId;
    cudaGraphicsResource *mCudaPbo;
public:
    AppPbo(unsigned width, unsigned height) {
        mCudaPbo = nullptr;
        glGenBuffers(1, &mId);
        // Make mId the current UNPACK buffer aka PBO (Pixel Buffer
        // Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mId);
        // Allocate data for the buffer. DYNAMIC (modified repeatedly)
        // DRAW (not reading from GL)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);

    };

    // registers & set mCudaPbo as the same as mId PBO.
    void registerBuffer() {
        // WriteDiscard flag = CUDA will not read this buffer--only
        // write the entire contents
        cudaErrChk(cudaGraphicsGLRegisterBuffer(&mCudaPbo,
                                                mId,
                                                cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // get a device pointer for cuda to access mCudaPbo
    void *mapGraphicsResource()
    {
        cudaErrChk(cudaGraphicsMapResources(1, &mCudaPbo));
        void *devPtr = nullptr;
        size_t size;
        cudaErrChk(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, mCudaPbo));
        return devPtr;
    }

    // stop access to mCudaPbo
    void unmapGraphicsResource()
    {
        cudaErrChk(cudaGraphicsUnmapResources(1, &mCudaPbo));
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
