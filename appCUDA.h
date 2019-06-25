#ifndef SMANDELBROTR_APP_CUDA_H
#define SMANDELBROTR_APP_CUDA_H

#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstring>

namespace AppCUDA {
    bool setDevice();
};

#endif
