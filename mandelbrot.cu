// mandelbrot cuda kernels
#include "mandelbrot.h"
#include "cudaErrorCheck.h"
//#include <cuda_runtime_api.h>
//#include <cuda_gl_interop.h>
#include <cuda.h>

// viridis color map buffer
__device__ static unsigned char viridis[256][3] = {
    { 84,  1, 68}, { 85,  2, 68}, { 87,  3, 69}, { 88,  5, 69}, { 90,  6, 69}, { 91,  8, 70},
    { 93,  9, 70}, { 94, 11, 70}, { 96, 12, 70}, { 97, 14, 71}, { 98, 15, 71}, {100, 17, 71},
    {101, 18, 71}, {102, 20, 71}, {104, 21, 72}, {105, 22, 72}, {106, 24, 72}, {108, 25, 72},
    {109, 26, 72}, {110, 28, 72}, {111, 29, 72}, {112, 30, 72}, {113, 32, 72}, {115, 33, 72},
    {116, 34, 72}, {117, 36, 72}, {118, 37, 72}, {119, 38, 72}, {120, 39, 72}, {121, 41, 71},
    {121, 42, 71}, {122, 43, 71}, {123, 44, 71}, {124, 46, 71}, {125, 47, 70}, {126, 48, 70},
    {126, 49, 70}, {127, 51, 70}, {128, 52, 69}, {129, 53, 69}, {129, 54, 69}, {130, 56, 68},
    {131, 57, 68}, {131, 58, 68}, {132, 59, 67}, {132, 60, 67}, {133, 62, 67}, {133, 63, 66},
    {134, 64, 66}, {134, 65, 65}, {135, 66, 65}, {135, 67, 65}, {136, 69, 64}, {136, 70, 64},
    {136, 71, 63}, {137, 72, 63}, {137, 73, 62}, {137, 74, 62}, {138, 75, 61}, {138, 77, 61},
    {138, 78, 60}, {138, 79, 60}, {139, 80, 59}, {139, 81, 59}, {139, 82, 58}, {139, 83, 58},
    {140, 84, 57}, {140, 85, 57}, {140, 86, 56}, {140, 87, 56}, {140, 88, 55}, {140, 89, 55},
    {141, 91, 54}, {141, 92, 54}, {141, 93, 53}, {141, 94, 53}, {141, 95, 52}, {141, 96, 52},
    {141, 97, 51}, {141, 98, 51}, {141, 99, 51}, {142,100, 50}, {142,101, 50}, {142,102, 49},
    {142,103, 49}, {142,104, 48}, {142,105, 48}, {142,106, 47}, {142,107, 47}, {142,108, 47},
    {142,109, 46}, {142,110, 46}, {142,111, 45}, {142,112, 45}, {142,112, 45}, {142,113, 44},
    {142,114, 44}, {142,115, 43}, {142,116, 43}, {142,117, 43}, {142,118, 42}, {142,119, 42},
    {142,120, 41}, {142,121, 41}, {142,122, 41}, {142,123, 40}, {142,124, 40}, {142,125, 40},
    {142,126, 39}, {142,127, 39}, {142,128, 38}, {142,129, 38}, {142,130, 38}, {142,130, 37},
    {142,131, 37}, {142,132, 37}, {142,133, 36}, {142,134, 36}, {142,135, 35}, {142,136, 35},
    {142,137, 35}, {141,138, 34}, {141,139, 34}, {141,140, 34}, {141,141, 33}, {141,142, 33},
    {141,143, 33}, {141,144, 32}, {140,145, 32}, {140,146, 32}, {140,147, 32}, {140,147, 31},
    {140,148, 31}, {139,149, 31}, {139,150, 31}, {139,151, 31}, {139,152, 30}, {138,153, 30},
    {138,154, 30}, {138,155, 30}, {137,156, 30}, {137,157, 30}, {137,158, 30}, {136,159, 30},
    {136,160, 30}, {136,161, 31}, {135,162, 31}, {135,163, 31}, {134,163, 31}, {134,164, 32},
    {134,165, 32}, {133,166, 33}, {133,167, 33}, {132,168, 34}, {131,169, 35}, {131,170, 35},
    {130,171, 36}, {130,172, 37}, {129,173, 38}, {129,174, 39}, {128,175, 40}, {127,175, 41},
    {127,176, 42}, {126,177, 43}, {125,178, 44}, {124,179, 46}, {124,180, 47}, {123,181, 48},
    {122,182, 50}, {121,183, 51}, {121,183, 53}, {120,184, 54}, {119,185, 56}, {118,186, 57},
    {117,187, 59}, {116,188, 61}, {115,189, 62}, {114,189, 64}, {113,190, 66}, {112,191, 68},
    {111,192, 70}, {110,193, 72}, {109,194, 73}, {108,194, 75}, {107,195, 77}, {106,196, 79},
    {105,197, 81}, {104,198, 83}, {102,198, 85}, {101,199, 88}, {100,200, 90}, { 99,201, 92},
    { 98,201, 94}, { 96,202, 96}, { 95,203, 98}, { 94,204,101}, { 92,204,103}, { 91,205,105},
    { 90,206,108}, { 88,206,110}, { 87,207,112}, { 85,208,115}, { 84,208,117}, { 82,209,119},
    { 81,210,122}, { 79,210,124}, { 78,211,127}, { 76,212,129}, { 75,212,132}, { 73,213,134},
    { 72,213,137}, { 70,214,139}, { 68,215,142}, { 67,215,144}, { 65,216,147}, { 63,216,149},
    { 62,217,152}, { 60,217,155}, { 58,218,157}, { 57,218,160}, { 55,219,163}, { 53,219,165},
    { 51,220,168}, { 50,220,171}, { 48,221,173}, { 46,221,176}, { 45,221,179}, { 43,222,181},
    { 41,222,184}, { 39,223,187}, { 38,223,189}, { 36,223,192}, { 35,224,195}, { 33,224,197},
    { 32,225,200}, { 30,225,203}, { 29,225,205}, { 28,226,208}, { 27,226,211}, { 26,226,213},
    { 25,227,216}, { 24,227,219}, { 24,227,221}, { 24,228,224}, { 24,228,226}, { 24,228,229},
    { 25,229,232}, { 25,229,234}, { 26,229,237}, { 27,230,239}, { 28,230,242}, { 30,230,244},
    { 31,230,247}, { 33,231,249}, { 35,231,251}, { 36,231,254}
};

extern "C" // no C++ name mangling
__global__ void mandel_float(uchar4 *ptr, int max_w, int max_h, int w, int h, float cx, float cy, float zoom, int iter_mult)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(x < w && y < h) {
        uchar4 bgra = {0x0,0x0,0x0,0x0}; // inside = black
        unsigned int max_iterations = iter_mult*256;
        zoom = 1.0f/zoom;
		float zoomX = zoom;
        float zoomY = zoom * (float)h / w;
        if(h > w) {
        	zoomX = zoom * (float)w / h;
        	zoomY = zoom;
        }
        float min_re = cx - zoomX;
        float min_im = cy - zoomY;
        float re_factor = 2*zoomX/(w-1);
        float im_factor = 2*zoomY/(h-1);
        float c_re = min_re + x*re_factor;
        float c_im = min_im + y*im_factor;
#if 0
		// testing my vertex shader
		float xf = 4.0*x/w;
		float yf = 4.0*y/h;
	    bgra.x = abs(xf - truncf(xf))*max_iterations;
        bgra.y = abs(yf - truncf(yf))*max_iterations;
        bgra.z = 0;
        *(ptr + offset) = bgra;
#else
        float z_re = c_re, z_im = c_im;
        for(unsigned int n = 0; n < max_iterations; ++n) {
            float z_re2 = z_re*z_re, z_im2 = z_im*z_im;
            if(z_re2 + z_im2 > 4.0f) {
                int nn = n & 0xFF;
                // outside the set, set color
                bgra.x = viridis[nn][0];
                bgra.y = viridis[nn][1];
                bgra.z = viridis[nn][2];
                break;
            }
            z_im = 2*z_re*z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
        *(ptr + offset) = bgra;
#endif
    }
    else if(x < max_w && y < max_h) {
         uchar4 bgra = {0x0,0xff,0xff,0x0};
    	*(ptr + offset) = bgra;
    }
}

extern "C" // no C++ name mangling
__global__ void mandel_double(uchar4 *ptr, int max_w, int max_h, int w, int h, double cx, double cy, double zoom, int iter_mult)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if(x < w && y < h) {
        uchar4 bgra = {0x0,0x0,0x0,0x0};  // inside = black
        unsigned int max_iterations = iter_mult*256;
        zoom = 1.0f/zoom;
 		double zoomX = zoom;
        double zoomY = zoom * (double)h / w;
        if(h > w) {
        	zoomX = zoom * (double)w / h;
        	zoomY = zoom;
        }
        double re_factor = 2*zoomX/(w-1);
        double im_factor = 2*zoomY/(h-1);
        double min_re = cx - zoomX;
        double min_im = cy - zoomY;
        double c_re = min_re + x*re_factor;
        double c_im = min_im + y*im_factor;
        double z_re = c_re, z_im = c_im;
        for(unsigned int n = 0; n < max_iterations; ++n) {
            double z_re2 = z_re*z_re, z_im2 = z_im*z_im;
            if(z_re2 + z_im2 > 4.0f) {
                int nn = n & 0xFF;
                // outside the set, set color
                bgra.x = viridis[nn][0];
                bgra.y = viridis[nn][1];
                bgra.z = viridis[nn][2];
                break;
            }
            z_im = 2*z_re*z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
        *(ptr + offset) = bgra;
    }
    else if(x < max_w && y < max_h) {
         uchar4 bgra = {0x0,0xff,0xff,0x0};
    	*(ptr + offset) = bgra;
    }
}

void mandelbrot(void *devPtr, unsigned winWidth, unsigned winHeight, unsigned texWidth, unsigned texHeight, double centerX, double centerY, double zoom, int iterMult, bool doublePrecision) {
    const int blockSize = 16; // 256 threads per block
    if(doublePrecision) {
        mandel_double<<<dim3(texWidth / blockSize, texHeight / blockSize), dim3(blockSize, blockSize)>>>((uchar4 *) devPtr, winWidth, winHeight, texWidth, texHeight, centerX, centerY, zoom, iterMult);
    }
    else {
        mandel_float<<<dim3(texWidth / blockSize, texHeight / blockSize), dim3(blockSize, blockSize)>>>((uchar4 *) devPtr, winWidth, winHeight, texWidth, texHeight, (float)centerX, (float)centerY, (float)zoom, iterMult);
    }
    cudaErrChk(cudaGetLastError());
    cuErrChk(cuCtxSynchronize());
}
