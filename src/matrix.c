#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <cblas.h>

#include "matrix.h"

/*
typedef struct Matrix {
    uint32_t rows, cols;      // Num Rows, Num Cols
    float* data;              // m x n Matrix - index with m*j + i
} Matrix;
*/

// Print Matrix
void matrix_print(Matrix m) {
    printf("%d x %d Matrix:\n", m.rows, m.cols);
    for (uint32_t row = 0; row < m.rows; row++) {
        for (uint32_t col = 0; col < m.cols; col++) {
            printf("% 0.3f ", m.data[row * m.cols + col]);
        }
        printf("\n");
    }
}

// Allocate a matrix, initialize to all zeroes
Matrix matrix_create(int rows, int cols) {

    Matrix m = {.rows = rows,
                .cols = cols,
                .data = calloc(rows * cols * sizeof(float), 1),
               };
    assert(m.data != NULL);

    return m;
}

// Destroy Matrix, frees data, but does not free Matrix itself
void matrix_destroy(Matrix* a) {
    a->rows = 0.0f;
    a->cols = 0.0f;
    if (a != NULL) free(a->data);
}

// Copy values in a matrix
void matrix_copy(Matrix m, Matrix a) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(m.cols == a.cols);

    memcpy(m.data, a.data, sizeof(m.data[0]) * m.rows * m.cols);

}

// Add Matrix  m = m + c * a
void matrix_add(Matrix m, float c, Matrix a) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(m.cols == a.cols);

    // First we store a in m, then add b to m
    // cblas_sgeadd: C = alpha * A + beta * C
    // cblas_sgeadd(CBLAS_ORDER, rows, cols, alpha, *A, a_width, beta, *C, c_width)
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, c, a.data, a.cols, 1, m.data, m.cols);

}

// Compute Sum Matrices m = (a + b)
void matrix_sum(Matrix m, Matrix a, Matrix b) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(a.rows == b.rows);
    assert(m.cols == a.cols);
    assert(a.cols == b.cols);

    // First we store a in m, then add b to m
    // cblas_sgeadd: C = alpha * A + beta * C
    // cblas_sgeadd(CBLAS_ORDER, rows, cols, alpha, *A, a_width, beta, *C, c_width)
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, 1, a.data, a.cols, 0, m.data, m.cols);
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, 1, b.data, b.cols, 1, m.data, m.cols);

}

// Compute Difference of Matrices m = (a - b)
void matrix_diff(Matrix m, Matrix a, Matrix b) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(a.rows == b.rows);
    assert(m.cols == a.cols);
    assert(a.cols == b.cols);

    // First we store a in m, then add -b to m
    // cblas_sgeadd: C = alpha * A + beta * C
    // cblas_sgeadd(CBLAS_ORDER, rows, cols, alpha, *A, a_width, beta, *C, c_width)
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, 1, a.data, a.cols, 0, m.data, m.cols);
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, -1, b.data, b.cols, 1, m.data, m.cols);
}

// Scalar multiply m = (c * a)
void matrix_smult(Matrix m, Matrix a, float c) {

    // Multiply by Scalar
    // cblas_sgeadd: C = alpha * A + beta * C
    // cblas_sgeadd(CBLAS_ORDER, rows, cols, alpha, *A, a_width, beta, *C, c_width)
    cblas_sgeadd(CblasRowMajor, m.rows, m.cols, c, a.data, a.cols, 0, m.data, m.cols);
}

// Matrix multiply m = (a * b)
void matrix_mmult(Matrix m, Matrix a, Matrix b) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(m.cols == b.cols);
    assert(a.cols == b.rows);

    // Use cblas single precision generic matrix multiplication method
    // C = alpha * Op(A) * Op(B) + beta * C, where A = m x k, B = k x n, C = m x n matrix
    // Op(x) = x^T (X-transpose), if specified in argument
    // cblas_sgemm(CBLAS_ORDER, transpose_A, transpose_B, m, n, k, alpha, *A, a_width, *B, b_width, beta, *C, c_width)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m.rows, m.cols, a.cols, 1, a.data, a.cols, b.data, b.cols, 0, m.data, m.cols);

}

// Complex Matrix Product, Using all the tools cblas has to offer
void matrix_cmult(Matrix m, Matrix a, bool a_t, Matrix b, bool b_t, float alpha, float beta) {

    // Assert we have compatible dimensions, even if we are transposing matrices
    uint32_t a_rows = (a_t ? a.cols : a.rows);
    uint32_t a_cols = (a_t ? a.rows : a.cols);
    uint32_t b_rows = (b_t ? b.cols : b.rows);
    uint32_t b_cols = (b_t ? b.rows : b.cols);
    assert(m.rows == a_rows);
    assert(m.cols == b_cols);
    assert(a_cols = b_rows);

    // Use cblas single precision generic matrix multiplication method
    // C = alpha * Op(A) * Op(B) + beta * C, where A = m x k, B = k x n, C = m x n matrix
    // Op(x) = x^T (X-transpose), if specified in argument
    // cblas_sgemm(CBLAS_ORDER, transpose_A, transpose_B, m, n, k, alpha, *A, a_width, *B, b_width, beta, *C, c_width)
    cblas_sgemm(CblasRowMajor, a_t ? CblasTrans : CblasNoTrans, b_t ? CblasTrans : CblasNoTrans, m.rows, m.cols, a_cols, alpha, a.data, a.cols, b.data, b.cols, beta, m.data, m.cols);
}

// Efficient multiplication of a diagonal matrix by a vector
void matrix_dmult(Matrix m, Matrix d, Matrix v) {

    // Assert we have compatible dimensions
    assert(d.rows == d.cols);
    assert(d.cols == v.rows);
    assert(m.rows == d.rows);
    assert(m.cols == v.cols);
    assert(v.cols == 1);

    for (uint32_t i = 0; i < m.rows; i++)
        m.data[i] = d.data[i*d.cols + i] * v.data[i];
}

// Hadamard Product m = (a_i * b_i for all elements i)
void matrix_hprod(Matrix m, Matrix a, Matrix b) {

    // Assert matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(a.rows == b.rows);
    assert(m.cols == a.cols);
    assert(a.cols == b.cols);

    // Multiply elements
    for (uint32_t row = 0; row < m.rows; row++) {
        for (uint32_t col = 0; col < m.cols; col++) {
            int rows = row * m.cols;
            m.data[rows + col] = a.data[rows + col] * b.data[rows + col];
        }
    }

}

// Transpose Matrix a, store in m
void matrix_transpose(Matrix m, Matrix a) {

    // First confirm matrices have compatible dimensions
    assert(m.rows == a.cols);
    assert(m.cols == a.rows);

    for (uint32_t row = 0; row < m.rows; row++) {
        for (uint32_t col = 0; col < m.cols; col++) {
            int rows = row * m.cols;
            m.data[rows + col] = a.data[col * a.cols + row];
        }
    }

}

// Apply activation function to elements of matrix
void matrix_activation(Matrix m, Matrix a, float act(float)) {

    // First confirm matrices have compatible dimensions
    assert(m.rows == a.rows);
    assert(m.cols == a.cols);

    for (uint32_t row = 0; row < a.rows; row++) {
        for (uint32_t col = 0; col < a.cols; col++) {
            int rows = row * m.cols;
            m.data[rows + col] = act(a.data[rows + col]);
        }
    }

}

// Generate random number with guassian distribution using Box_Muller method
float random_gaussian(float mean, float stdev) {

    float u = (float)rand() / (float)RAND_MAX;
    float v = (float)rand() / (float)RAND_MAX;

    return stdev * sqrtf(-2 * logf(u)) * cosf(2 * M_PI * v) + mean;
}

// Initialize all elements to random value
void matrix_initialize_random(Matrix m) {

    // Initialize all values to random float between -1 and 1
    for (uint32_t row = 0; row < m.rows; row++) {
        for (uint32_t col = 0; col < m.cols; col++) {
            int rows = row * m.cols;
            m.data[rows + col] = -1.0 + 2.0*((float)(rand()) / (float)(RAND_MAX));
        }
    }
}

void matrix_initialize_gaussian(Matrix m, float mean, float stdev) {

    // Initialize all values to random float between -1 and 1
    for (uint32_t row = 0; row < m.rows; row++) {
        for (uint32_t col = 0; col < m.cols; col++) {
            int rows = row * m.cols;
            m.data[rows + col] = random_gaussian(mean, stdev);
        }
    }
}

// Set all elements to zero
void matrix_zero(Matrix m) {
    memset(m.data, 0, m.cols * m.rows * sizeof(m.data[0]));
}

// Set all elements to one
void matrix_ones(Matrix m) {
    for (uint32_t i = 0; i < m.rows * m.cols; i++) m.data[i] = 1.0f;
}

