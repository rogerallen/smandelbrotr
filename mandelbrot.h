#ifndef SMANDELBROTR_MANDELBROT_H
#define SMANDELBROTR_MANDELBROT_H

void compileMandelbrot();

void mandelbrot(void *devPtr, unsigned winWidth, unsigned winHeight, unsigned texWidth, unsigned texHeight, double centerX, double centerY, double zoom, int iterMult, bool doublePrecision);

#endif
