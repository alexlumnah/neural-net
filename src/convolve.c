#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdio.h>
#include <fftw3.h>

#include "convolve.h"
#include "matrix.h"

#define PI 3.1415926535
/*
// Cross-correlate matrices
// Only return fully overlapping portions
void cross_correlate(Matrix m, Matrix a, Matrix k, bool zero) {

    assert(m.rows == a.rows - k.rows + 1);
    assert(m.cols == a.cols - k.cols + 1);

    // Row/column of output
    for (uint32_t i = 0; i < m.rows; i++) {
        float* m_r = m.data + i * m.cols;
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m_r[j] = 0.0f;
            // Iterate over each cell in kernel
            for (uint32_t p = 0; p < k.rows; p++) {
                float* k_r = k.data + p * k.cols;
                float* a_r = a.data + (p + i) * a.cols;
                for (uint32_t q = 0; q < k.cols; q++) {
                    m_r[j] += k_r[q] * a_r[j + q];
                }
            }
        }
    }
}

// Convolve matrix with kernel
// Only return fully overlapping portions
void rotate_convolve(Matrix m, Matrix a, Matrix k, bool zero) {

    assert(m.rows == a.rows - k.rows + 1);
    assert(m.cols == a.cols - k.cols + 1);

    // Row/column of output
    for (uint32_t i = 0; i < m.rows; i++) {
        float* m_r = m.data + i * m.cols;
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m_r[j] = 0.0f;
            // Iterate over each cell in kernel
            for (uint32_t p = 0; p < k.rows; p++) {
                float* k_r = k.data + (k.rows - p - 1) * k.cols;
                float* a_r = a.data + (a.rows - 1 - (p + i)) * a.cols;
                for (uint32_t q = 0; q < k.cols; q++) {
                    m_r[j] += k_r[k.cols - q - 1] * a_r[a.cols - 1 - (j + q)];
                }
            }
        }
    }
}

// Convolve matrix with kernel
// Return entire convolution, including non-overlapping parts
void full_convolve(Matrix m, Matrix a, Matrix k, bool zero) {

    assert(m.rows == a.rows + k.rows - 1);
    assert(m.cols == a.cols + k.cols - 1);

    // Row/column of output
    for (uint32_t i = 0; i < m.rows; i++) {
        float* m_r = m.data + i * m.cols;
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m_r[j] = 0.0f;
            // Iterate over each cell in kernel
            for (uint32_t p = 0; p <= i; p++) {
                if (p >= a.rows || (i - p) >= k.rows) continue;
                float* k_r = k.data + (i - p) * k.cols;
                float* a_r = a.data + p * a.cols;
                for (uint32_t q = 0; q <= j; q++) {
                    if (q >= a.cols || (j - q) >= k.cols) continue;
                    m_r[j] += k_r[j - q] * a_r[q];
                }
            }
        }
    }
}

// Convolve matrix with kernel
// Only return fully overlapping portions
void convolve(Matrix m, Matrix a, Matrix k, bool zero) {

    assert(m.rows == a.rows - k.rows + 1);
    assert(m.cols == a.cols - k.cols + 1);

    // Row/column of output
    for (uint32_t i = 0; i < m.rows; i++) {
        float* m_r = m.data + i * m.cols;
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m_r[j] = 0.0f;
            // Iterate over each cell in kernel
            for (uint32_t p = 0; p < k.rows; p++) {
                float* k_r = k.data + (k.rows - p - 1) * k.cols;
                float* a_r = a.data + (p + i) * a.cols;
                for (uint32_t q = 0; q < k.cols; q++) {
                    m_r[j] += k_r[k.cols - q - 1] * a_r[j + q];
                }
            }
        }
    }
}

// Compute dft of a list of complex numbers
void dft(float complex* out, float complex* in, uint32_t n) {

    for (uint32_t k = 0; k < n; k++) {
        out[k] = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            out[k] += in[i] * cexp(-2 * PI * I * k * i / n);
        }
    }
}

// Compute idft of a list of complex numbers
// Result is multiplied by length of input
void idft(float complex* out, float complex* in, uint32_t n) {

    for (uint32_t k = 0; k < n; k++) {
        out[k] = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            out[k] += in[i] * cexp(2 * PI * I * k * i / n);
        }
    }
}

// Compute fft of a list of complex numbers
void fft(float complex* out, float complex* in, uint32_t n) {

    if (n % 2 == 0 && n/2) {

        float complex e_in[n/2];   // Even inputs
        float complex e_out[n/2];   // Even outputs
        float complex o_in[n/2];   // Odd inputs
        float complex o_out[n/2];   // Odd outputs

        for (uint32_t i = 0; i < n/2; i++) {
            e_in[i] = in[2 * i];
            o_in[i] = in[2 * i + 1];
        }
        fft(e_out, e_in, n/2);
        fft(o_out, o_in, n/2);

        for (uint32_t k = 0; k < n/2; k++) {
            float complex odd = cexp(-2 * PI * I * k / n) * o_out[k];
            out[k] = e_out[k] + odd;
            out[k + n/2] = e_out[k] - odd;
        }
    } else if (n == 1) {
        out[0] = in[0];
    } else {
        // Compute dft for non-even lengthed lists
        dft(out, in, n);
    }
}

// Compute ifft for list of complex numbers
// Result is multiplied by length of input
void ifft(float complex* out, float complex* in, uint32_t n) {

    if (n % 2 == 0 && n/2) {

        float complex e_in[n/2];   // Even inputs
        float complex e_out[n/2];   // Even outputs
        float complex o_in[n/2];   // Odd inputs
        float complex o_out[n/2];   // Odd outputs

        for (uint32_t i = 0; i < n/2; i++) {
            e_in[i] = in[2 * i];
            o_in[i] = in[2 * i + 1];
        }
        ifft(e_out, e_in, n/2);
        ifft(o_out, o_in, n/2);

        for (uint32_t k = 0; k < n/2; k++) {
            float complex odd = cexp(2 * PI * I * k / n) * o_out[k];
            out[k] = (e_out[k] + odd);
            out[k + n/2] = (e_out[k] - odd);
        }
    } else if (n == 1) {
        out[0] = in[0];
    } else {
        // Compute dft for non-even lengthed lists
        idft(out, in, n);
    }
}

// Compute 2d fft of a matrix, output as a flat list
// Pad input with zeros to rows, cols
void matrix_fft(float complex* out, Matrix m, uint32_t rows, uint32_t cols) {

    uint32_t n = rows * cols;

    float complex in[n];
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            if (i < m.rows && j < m.cols)
                in[i * (cols) + j] = m.data[i * m.cols + j];
            else
                in[i * (cols) + j] = 0.0f;
        }
    }

    fft(out, in, n);

}

// Compute inverse fft
// Store values in output matrix from row t_pad (top padding) to t_pad + m.rows
// and from column l_pad (left padding) to l_pad + m.cols
// Rows and Cols is the number of rows and columns in the input fft
void matrix_ifft(Matrix m, float complex* in, uint32_t t_pad, uint32_t l_pad, uint32_t rows, uint32_t cols) {

    assert(m.rows + t_pad <= rows);
    assert(m.cols + l_pad <= cols);

    uint32_t n = rows * cols;

    float complex out[n];
    ifft(out, in, n);

    for (uint32_t i = 0; i < m.rows; i++) {
        for (uint32_t j = 0; j < m.cols; j++) {
            m.data[i * m.cols + j] = out[(i + t_pad) * cols + j + l_pad] / n;
        }
    }

}


static uint32_t ceil_power(uint32_t n) {

    assert(n > 0);

    uint32_t val = 1;
    while (val < n) {
        val *= 2;
    }

    return val;
}
*/

// Create structure holding necessary meta data to compute a convolution
ConvPlan create_conv_plan(Matrix a, Matrix k) {

    assert(a.rows >= k.rows);
    assert(a.cols >= k.cols);

    ConvPlan p = {0};

    uint32_t size = a.rows * a.cols;
    p.size = size;

    p.a = a;
    p.k = k;
    p.k_pad = matrix_create(a.rows, a.cols);

    p.a_fft = calloc(size / 2 + 1, sizeof(fftwf_complex));
    p.k_fft = calloc(size / 2 + 1, sizeof(fftwf_complex));
    p.result = calloc(size, sizeof(float));

    p.a_plan = fftwf_plan_dft_r2c_1d(size, p.a.data, p.a_fft, 0);
    p.k_plan = fftwf_plan_dft_r2c_1d(size, p.k_pad.data, p.k_fft, 0);
    p.i_plan = fftwf_plan_dft_c2r_1d(size, p.a_fft, p.result, 0);

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

// Perform an FFT Convolution of two matrices
void execute_convolution(ConvPlan* p) {

    // Compute forward FFT for input matrix
    fftwf_execute(p->a_plan);

    // Pairwise multiply each element, scale by size of array
    for (uint32_t i = 0; i < p->size / 2 + 1; i++) {
        p->a_fft[i] *= p->k_fft[i] / p->size;
    }

    // Compute inverse fft
    fftwf_execute(p->i_plan);

}

// Perform an FFT Cross-correlation of two matrices
void execute_cross_correlation(ConvPlan* p) {

    // Compute forward FFT for input matrix
    fftwf_execute(p->a_plan);

    // Pairwise multiply each element, scale by size of array
    for (uint32_t i = 0; i < p->size / 2 + 1; i++) {
        p->a_fft[i] *= conj(p->k_fft[i]) / p->size;
    }

    // Compute inverse fft
    fftwf_execute(p->i_plan);

}

// Perform a convolution, keep only fully overlapping parts
void execute_overlap_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(out.rows == p->a.rows - p->k.rows + 1);
    assert(out.cols == p->a.cols - p->k.cols + 1);

    // Compute convolution
    execute_convolution(p);

    // Only keep completely overlapping output
    uint32_t t_pad = p->k.rows - 1; // Non-fully overlapping rows on top
    uint32_t l_pad = p->k.cols - 1; // Non-fully overlapping cols on left

    // Copy output to destination
    if (zero) {
        for (uint32_t i = 0; i < out.rows; i++) {
            memcpy(out.data + i * out.cols,
                   p->result + (i + t_pad) * p->a.cols + l_pad,
                   out.cols * sizeof(out.data[0]));
        }
    }
    // If not zeroing matrix, use slow loop to add to each element
    else {
        for (uint32_t i = 0; i < out.rows; i++) {
            float* out_row = out.data + i * out.cols;
            float* res_row = p->result + (i + t_pad) * p->a.cols;
            for (uint32_t j = 0; j < out.cols; j++) {
                out_row[j] += res_row[j + l_pad];
            }
        }
    }
}

void execute_full_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(out.rows == p->a.rows + p->k.rows - 1);
    assert(out.cols == p->a.cols + p->k.cols - 1);

    // Compute convolution
    execute_convolution(p);

    // Copy output to destination
    if (zero) {
        for (uint32_t i = 0; i < out.rows; i++) {
            memcpy(out.data + i * out.cols,
                   p->result + i * p->a.cols,
                   out.cols * sizeof(out.data[0]));
        }
    }
    // If not zeroing matrix, use slow loop to add to each element
    else {
        for (uint32_t i = 0; i < out.rows; i++) {
            float* out_row = out.data + i * out.cols;
            float* res_row = p->result + i * p->a.cols;
            for (uint32_t j = 0; j < out.cols; j++) {
                out_row[j] += res_row[j];
            }
        }
    }
}

// Rotate matrix, then convolve it
void execute_rot_conv(ConvPlan* p, Matrix out, bool zero) {

    assert(out.rows == p->a.rows - p->k.rows + 1);
    assert(out.cols == p->a.cols - p->k.cols + 1);

    // Compute cross-correlation
    execute_cross_correlation(p);

    // Only keep completely overlapping output
    uint32_t t_pad = p->k.rows - 1; // Non-fully overlapping rows on top
    uint32_t l_pad = p->k.cols - 1; // Non-fully overlapping cols on left

    // Reverse, and copy results to output
    uint32_t res_rows = p->a.rows;
    uint32_t res_cols = p->a.cols;
    for (uint32_t i = 0; i < out.rows; i++) {
        float* out_row = out.data + (i) * out.cols;
        float* res_row = p->result + (res_rows - 1 - i - t_pad) * res_cols;
        for (uint32_t j = 0; j < out.cols; j++) {
            if (zero) out_row[j] = 0.0f;
            out_row[j] += res_row[res_cols - 1 - j - l_pad];
        }
    }
}

// Convolve two matrices using fft algorithm
// Assumes inputs are pre-padded to be same size
void matrix_fft_full_convolve(Matrix m, Matrix a, Matrix b, bool zero) {

    assert(b.rows < a.rows && b.cols < a.cols);
    assert(m.rows == a.rows + b.rows - 1);
    assert(m.cols == a.cols + b.cols - 1);

    // Pad a and b to size of m
    uint32_t rows = m.rows;
    uint32_t cols = m.cols;
    uint32_t size = rows * cols;

    // Compute ffts
    fftwf_complex a_fft[size/2 + 1];
    fftwf_complex b_fft[size/2 + 1];

    // Pad inputs
    float b_pad[size];
    float a_pad[size];
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            if (i < a.rows && j < a.cols) a_pad[i * cols + j] = a.data[i * a.cols + j];
            else a_pad[i * cols + j] = 0.0f;
            if (i < b.rows && j < b.cols) b_pad[i * cols + j] = b.data[i * b.cols + j];
            else b_pad[i * cols + j] = 0.0f;
        }
    }
            
    fftwf_plan a_plan = fftwf_plan_dft_r2c_1d(size, a_pad, a_fft, 0);
    fftwf_plan b_plan = fftwf_plan_dft_r2c_1d(size, b_pad, b_fft, 0);
    fftwf_execute(a_plan);
    fftwf_execute(b_plan);

    // Pairwise multiply each element
    for (uint32_t i = 0; i < size / 2 + 1; i++) {
        a_fft[i] *= b_fft[i];
    }

    // Compute inverse fft
    float out[size];
    a_plan = fftwf_plan_dft_c2r_1d(size, a_fft, out, 0);
    fftwf_execute(a_plan);

    // Only keep non-zero output
    for (uint32_t i = 0; i < m.rows; i++) {
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m.data[i * m.cols + j] = out[i * cols + j] / size;
            else m.data[i * m.cols + j] += out[i * cols + j] / size;
        }
    }

}

// Use fft method to convolve two matrices
void matrix_fft_convolve(Matrix m, Matrix a, Matrix b, bool zero) {

    assert(b.rows < a.rows && b.cols < a.cols);
    assert(m.rows == (MAX(a.rows, b.rows) - MIN(a.rows, b.rows) + 1));
    assert(m.cols == (MAX(a.cols, b.cols) - MIN(a.cols, b.cols) + 1));

    // Pad to nearest power of 2
    uint32_t rows = a.rows;//ceil_power(MAX(a.rows, b.rows));
    uint32_t cols = a.cols;//ceil_power(MAX(a.cols, b.cols));
    uint32_t size = rows * cols;

    // Compute ffts
    fftwf_complex a_fft[size/2 + 1];
    fftwf_complex b_fft[size/2 + 1];

    // Pad shorter input - assumes b is smaller than a
    float b_pad[size];
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            if (i < b.rows && j < b.cols) b_pad[i * cols + j] = b.data[i * b.cols + j];
            else b_pad[i * cols + j] = 0.0f;
        }
    }
            
    fftwf_plan a_plan = fftwf_plan_dft_r2c_1d(size, a.data, a_fft, 0);
    fftwf_plan b_plan = fftwf_plan_dft_r2c_1d(size, b_pad, b_fft, 0);
    fftwf_execute(a_plan);
    fftwf_execute(b_plan);

    // Pairwise multiply each element
    for (uint32_t i = 0; i < size / 2 + 1; i++) {
        a_fft[i] *= b_fft[i];
    }

    // Compute inverse fft
    float out[size];
    a_plan = fftwf_plan_dft_c2r_1d(size, a_fft, out, 0);
    fftwf_execute(a_plan);

    // Only keep completely overlapping output
    uint32_t t_pad = MIN(a.rows, b.rows) - 1;
    uint32_t l_pad = MIN(a.cols, b.cols) - 1;
    for (uint32_t i = 0; i < m.rows; i++) {
        for (uint32_t j = 0; j < m.cols; j++) {
            if (zero) m.data[i * m.cols + j] = 0.0f;
            m.data[i * m.cols + j] += out[(i + t_pad) * cols + j + l_pad] / size;
        }
    }

}

void matrix_fft_rotate_convolve(Matrix m, Matrix a, Matrix b, bool zero) {

    float data[a.rows * a.cols];
    Matrix a_rot = {.rows = a.rows,
                    .cols = a.cols,
                    .data = data};

    for (uint32_t i = 0; i < a.rows; i++) {
        for (uint32_t j = 0; j < a.cols; j++) {
            a_rot.data[(a.rows - 1 - i) * a_rot.cols + (a.cols - 1 - j)] = a.data[i * a.cols + j];
        }
    }

    matrix_fft_convolve(m, a_rot, b, zero);
}





