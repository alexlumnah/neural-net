#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <fftw3.h>

#include "convolve.h"
#include "matrix.h"

#define PI 3.1415926535

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
                in[i * (cols) + j] = m.data[i * m.rows + j];
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
            m.data[i * m.cols + j] = out[(i + t_pad) * (cols) + j + l_pad] / n;
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

// Convolve two matrices using fft algorithm
// Only output areas that are fully overlapping
void matrix_fft_convolve(Matrix m, Matrix a, Matrix b) {

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


    //float complex a_fft[size];
    //float complex b_fft[size];
    //matrix_fft(a_fft, a, rows, cols);
    //matrix_fft(b_fft, b, rows, cols);

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
            m.data[i * m.cols + j] = out[(i + t_pad) * (cols) + j + l_pad] / size;
        }
    }


    //uint32_t top_pad = MIN(a.rows, b.rows) - 1;
    //uint32_t left_pad = MIN(a.cols, b.cols) - 1;
    //matrix_ifft(m, a_fft, top_pad, left_pad, rows, cols);

}

