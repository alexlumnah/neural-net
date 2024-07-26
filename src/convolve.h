#ifndef CONVOLVE_H
#define CONVOLVE_H

#include <complex.h>
#include <fftw3.h>

#include "matrix.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    uint32_t size;          // Size of FFT
    Matrix a;               // Input Matrix
    Matrix k;               // Kernel Matrix
    Matrix k_pad;           // Padded Kernel Matrix
    fftwf_complex* a_fft;   // FFT of Input Matrix
    fftwf_complex* k_fft;   // FFT of Kernel Matrix
    float* result;          // Storage for result
    fftwf_plan a_plan;      // FFT Plan for Input Matrix
    fftwf_plan k_plan;      // FFT Plan for Kernel Matrix
    fftwf_plan i_plan;      // FFT Plan for inverse FFT
} ConvPlan;

ConvPlan create_conv_plan(Matrix a, Matrix k);
void update_kernel(ConvPlan* plan);

void execute_convolution(ConvPlan* plan);
void execute_overlap_conv(ConvPlan* p, Matrix out, bool zero);
void execute_full_conv(ConvPlan* p, Matrix out, bool zero);
void execute_rot_conv(ConvPlan* p, Matrix out, bool zero);

/*
// Convolve two matrices - cut-off non-completely overlapping results
void convolve(Matrix m, Matrix a, Matrix b, bool zero);

// Convolve two matrices - no cut-off
void full_convolve(Matrix m, Matrix a, Matrix b, bool zero);

// Convolve two matrices - cut-off non-completely overlapping results
void rotate_convolve(Matrix m, Matrix a, Matrix b, bool zero);

// Cross-corrrelate two matrices - cut-off non-completely overlapping results
void cross_correlate(Matrix m, Matrix a, Matrix b, bool zero);

// Compute fft of a list of complex numbers
void fft(float complex* out, float complex* in, uint32_t n);

// Compute ifft of a list of complex numbers
void ifft(float complex* out, float complex* in, uint32_t n);

// Compute fft of a matrix as a 1d list
void matrix_fft(float complex* out, Matrix m, uint32_t rows, uint32_t cols);

// Compute ifft of a matrix as a 1d list
void matrix_ifft(Matrix m, float complex* in, uint32_t t_pad, uint32_t l_pad, uint32_t rows, uint32_t cols);
*/
// Convolve two matrices using fft method
void matrix_fft_convolve(Matrix m, Matrix a, Matrix b, bool zero);
void matrix_fft_full_convolve(Matrix m, Matrix a, Matrix b, bool zero);
void matrix_fft_rotate_convolve(Matrix m, Matrix a, Matrix b, bool zero);

#endif // CONVOLVE_H

