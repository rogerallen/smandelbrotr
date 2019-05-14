#ifndef CUDA_ERROR_CHECK
#define CUDA_ERROR_CHECK
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// s/gpu/cuda/

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <stdio.h>

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA assert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define cuErrChk(ans) { cuAssert((ans), __FILE__, __LINE__); }
inline void cuAssert(CUresult code, const char *file, int line, bool abort=true)
{
    if (code != CUDA_SUCCESS)
    {
        const char *errStr;
        cuGetErrorString(code, &errStr);
        fprintf(stderr,"CU assert: %s %s:%d\n", errStr, file, line);
        if (abort) exit(code);
    }
}
#endif
