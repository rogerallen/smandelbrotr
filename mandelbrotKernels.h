#ifndef SMANDELBROTR_MANDELBROT_KERNELS_H
#define SMANDELBROTR_MANDELBROT_KERNELS_H

extern "C" __global__ void mandel_float(uchar4 *ptr, int max_w, int max_h, int w, int h, float cx, float cy, float zoom, int iter_mult);
extern "C" __global__ void mandel_double(uchar4 *ptr, int max_w, int max_h, int w, int h, double cx, double cy, double zoom, int iter_mult);

#endif