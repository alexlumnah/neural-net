#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdio.h>
#include <fftw3.h>

#include "convolve.h"
#include "matrix.h"

// Create structure holding necessary meta data to compute a convolution
ConvPlan create_conv_plan(Matrix a, uint32_t a_depth, Matrix k, ConvType type) {

    assert(a.rows >= k.rows);
    assert(a.cols >= k.cols);

    ConvPlan p = {0};

    p.type = type;

    p.a = a;
    p.depth = a_depth;

    // Determine required padding
    uint32_t rows, cols;
    if (type == CONV_OVERLAP) {
        rows = a.rows;
        cols = a.cols;
        p.a_pad = a;
    } else {
        rows = a.rows + k.rows - 1;
        cols = a.cols + k.cols - 1;

        // Create a padded matrix for a, accounting for depth
        p.a_pad = matrix_create(rows * cols * a_depth, 1);
        p.a_pad.rows = rows;
        p.a_pad.cols = cols;
    }
    uint32_t size = rows * cols;
    p.size = size;

    // Create structures to hold FFT and plans for each input slice
    p.a_ffts = calloc(a_depth, sizeof(fftwf_complex*));
    p.a_plans = calloc(a_depth, sizeof(fftwf_plan));
    p.i_plans = calloc(a_depth, sizeof(fftwf_plan));
    p.results = calloc(a_depth, sizeof(float*));
    for (uint32_t s = 0; s < a_depth; s++) {
        p.a_ffts[s] = calloc(size / 2 + 1, sizeof(fftwf_complex));
        p.results[s] = calloc(size, sizeof(float));
        p.a_plans[s] = fftwf_plan_dft_r2c_1d(size,
                                             p.a_pad.data + s * size,
                                             p.a_ffts[s],
                                             0);
        p.i_plans[s] = fftwf_plan_dft_c2r_1d(size,
                                             p.a_ffts[s],
                                             p.results[s],
                                             0);
    }

    // Create structure to hold FFT and plan for kernel
    p.k = k;
    p.k_pad = matrix_create(rows, cols);
    p.k_fft = calloc(size / 2 + 1, sizeof(fftwf_complex));
    p.k_plan = fftwf_plan_dft_r2c_1d(size, p.k_pad.data, p.k_fft, 0);

    return p;
}

// Update Kernel data in a convolution plan and execute FFT
// This is separate, because kernel is only updated after gradient descent
void update_kernel(ConvPlan* p) {

    Matrix k = p->k;
    Matrix k_pad = p->k_pad;

    // Copy data from kernel to padded kernel matrix
    for (uint32_t i = 0; i < k.rows; i++) {
        memcpy(k_pad.data + i * k_pad.cols,
               k.data + i * k.cols,
               k.cols * sizeof(k.data[0]));
    }

    fftwf_execute(p->k_plan);
}

// Update padded copy of input matrix. Only required for full convolutions
// Called by execute_full_conv
static void update_input_matrix(ConvPlan* p) {

    assert(p->type == CONV_FULL);
    Matrix a = p->a;
    Matrix a_pad = p->a_pad;

    // Copy data from kernel to padded kernel matrix
    uint32_t a_pad_size = a_pad.rows * a_pad.cols;
    uint32_t a_size = a.rows * a.cols;
    for (uint32_t s = 0; s < p->depth; s++) {
        for (uint32_t i = 0; i < a.rows; i++) {
            memcpy(a_pad.data + s * a_pad_size + i * a_pad.cols,
                   a.data     + s * a_size     + i * a.cols,
                   a.cols * sizeof(a.data[0]));
        }
    }
}
// Perform an FFT Convolution of two matrices
void execute_convolution(ConvPlan* p) {

    for (uint32_t s = 0; s < p->depth; s++ ) {

        // Compute forward FFT for input matrix
        fftwf_execute(p->a_plans[s]);

        // Pairwise multiply each element, scale by size of array
        for (uint32_t i = 0; i < p->size / 2 + 1; i++) {
            p->a_ffts[s][i] *= p->k_fft[i] / p->size;
        }

        // Compute inverse fft
        fftwf_execute(p->i_plans[s]);
    }
}

// Perform an FFT Cross-correlation of two matrices
void execute_cross_correlation(ConvPlan* p) {

    for (uint32_t s = 0; s < p->depth; s++ ) {

        // Compute forward FFT for input matrix
        fftwf_execute(p->a_plans[s]);

        // Pairwise multiply each element, scale by size of array
        for (uint32_t i = 0; i < p->size / 2 + 1; i++) {
            p->a_ffts[s][i] *= conj(p->k_fft[i]) / p->size;
        }

        // Compute inverse fft
        fftwf_execute(p->i_plans[s]);
    }
}

// Perform a convolution, keep only fully overlapping parts
void execute_overlap_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(p->type == CONV_OVERLAP);
    assert(out.rows == p->a.rows - p->k.rows + 1);
    assert(out.cols == p->a.cols - p->k.cols + 1);

    // Compute convolution
    execute_convolution(p);

    // Only keep completely overlapping output
    uint32_t t_pad = p->k.rows - 1; // Non-fully overlapping rows on top
    uint32_t l_pad = p->k.cols - 1; // Non-fully overlapping cols on left

    // Zero output if requested
    if (zero)
        memset(out.data, 0, out.rows * out.cols * sizeof(out.data[0]));

    // Copy results to output
    for (uint32_t s = 0; s < p->depth; s++) {
        for (uint32_t i = 0; i < out.rows; i++) {
            float* out_row = out.data + i * out.cols;
            float* res_row = p->results[s] + (i + t_pad) * p->a.cols;
            for (uint32_t j = 0; j < out.cols; j++) {
                out_row[j] += res_row[j + l_pad];
            }
        }
    }
}

// Rotate matrix, then convolve it
void execute_rot_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(p->type == CONV_OVERLAP);
    assert(out.rows == p->a.rows - p->k.rows + 1);
    assert(out.cols == p->a.cols - p->k.cols + 1);

    // Compute cross-correlation
    execute_cross_correlation(p);

    // Only keep completely overlapping output
    uint32_t t_pad = p->k.rows - 1; // Non-fully overlapping rows on top
    uint32_t l_pad = p->k.cols - 1; // Non-fully overlapping cols on left

    // Zero output if requested
    if (zero)
        memset(out.data, 0, out.rows * out.cols * sizeof(out.data[0]));

    // Reverse, and copy results to output
    uint32_t r_rows = p->a.rows;
    uint32_t r_cols = p->a.cols;
    for (uint32_t s = 0; s < p->depth; s++) {
        for (uint32_t i = 0; i < out.rows; i++) {
            float* out_row = out.data + (i) * out.cols;
            float* r_row = p->results[s] + (r_rows - 1 - i - t_pad)*r_cols;
            for (uint32_t j = 0; j < out.cols; j++) {
                if (zero) out_row[j] = 0.0f;
                out_row[j] += r_row[r_cols - 1 - j - l_pad];
            }
        }
    }
}

// Perform a full convolution, keeping non-overlapping parts
// Assumes output matrix is first slice of a larger matrix
// with same depth as the plan
void execute_full_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(p->type == CONV_FULL);
    assert(out.rows == p->a.rows + p->k.rows - 1);
    assert(out.cols == p->a.cols + p->k.cols - 1);

    // First copy input to padded input matrix
    update_input_matrix(p);

    // Compute convolution
    execute_convolution(p);

    // Zero output if requested
    if (zero)
        memset(out.data, 0, out.rows * out.cols * sizeof(out.data[0]));

    // Copy results to output
    for (uint32_t s = 0; s < p->depth; s++) {
        for (uint32_t i = 0; i < out.rows; i++) {
            // Add s * rows * cols to out_matrix, because...
            // TBD if this is right - experiements say no...
            assert(false);
            float* out_row = out.data + s * out.rows * out.cols + i * out.cols;
            float* res_row = p->results[s] + i * p->a.cols;
            for (uint32_t j = 0; j < out.cols; j++) {
                out_row[j] += res_row[j];
            }
        }
    }
}

