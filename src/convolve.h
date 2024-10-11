#ifndef CONVOLVE_H
#define CONVOLVE_H

#include <complex.h>
#include <fftw3.h>

#include "matrix.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef enum {
    CONV_OVERLAP,
    CONV_FULL
} ConvType;

typedef struct {
    ConvType type;          // Type of Convolution
    uint32_t size;          // Size of FFT

    Matrix k;               // Kernel Matrix
    Matrix k_pad;           // Padded Kernel Matrix
    fftwf_complex* k_fft;   // FFT of Kernel Matrix
    fftwf_plan k_plan;      // FFT Plan for Kernel Matrix

    Matrix a;               // Input Matrix
    Matrix a_pad;           // Padded Input Matrix
    uint32_t depth;         // Depth of input matrix

    fftwf_complex** a_ffts; // FFTs of Input Matrices
    fftwf_plan* a_plans;    // FFT Plans for Input Matrices
    fftwf_plan* i_plans;    // FFT Plans for inverse FFTs
    float** results;        // Storage for results
} ConvPlan;

ConvPlan create_conv_plan(Matrix a, uint32_t a_depth, Matrix k, ConvType type);
void update_kernel(ConvPlan* plan);

void execute_convolution(ConvPlan* plan);
void execute_overlap_conv(ConvPlan* p, Matrix out, bool zero);
void execute_full_conv(ConvPlan* p, Matrix out, bool zero);
void execute_rot_conv(ConvPlan* p, Matrix out, bool zero);

#endif // CONVOLVE_H

