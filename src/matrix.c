#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

#include "matrix.h"
/*
typedef struct Matrix {
    uint32_t rows, cols;      // Num Rows, Num Cols
    float data[];       // m x n Matrix - index with m*j + i
} Matrix;
*/

// Print Matrix
void matrix_print(Matrix* mat) {
    printf("%d x %d Matrix:\n", mat->rows, mat->cols);
    for (uint32_t row = 0; row < mat->rows; row++) {
        for (uint32_t col = 0; col < mat->cols; col++) {
            int rows = row * mat->cols;
            printf("% 0.3f ",mat->data[rows + col]);
        }
        printf("\n");
    }
}

// Allocate a matrix, initialize to all zeroes
Matrix* matrix_create(int rows, int cols) {

    Matrix* mat = calloc(sizeof(Matrix) + rows * cols * sizeof(float), 1);
    mat->rows = rows;
    mat->cols = cols;

    return mat;
}

// Destroy Matrix, freeing memory for data
void matrix_destroy(Matrix* a) {
    if (a != NULL)
        free(a);
}

// Add Matrices (a + b)
MatrixStatus matrix_add(Matrix* m, Matrix* a, Matrix* b) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->rows || a->rows != b->rows ||
        m->cols != a->cols || a->cols != b->cols)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    // First we 
    // cblas_sgeadd c = alpha*a + beta*c
    cblas_sgeadd(CblasRowMajor, m->rows, m->cols, 1, a->data, a->cols, 0, m->data, m->cols);
    cblas_sgeadd(CblasRowMajor, m->rows, m->cols, 1, b->data, b->cols, 1, m->data, m->cols);

    /* void cblas_sgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, float *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST float cbeta, 
          float *c, OPENBLAS_CONST blasint cldc);
    */

    /*
    // Add matrices
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = a->data[rows + col] + b->data[rows + col];
        }
    }*/

    return MATRIX_SUCCESS;

}

// Subtract Matrices (a - b)
MatrixStatus matrix_sub(Matrix* m, Matrix* a, Matrix* b) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->rows || a->rows != b->rows ||
        m->cols != a->cols || a->cols != b->cols)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    // Subtract matrices
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = a->data[rows + col] - b->data[rows + col];
        }
    }

    return MATRIX_SUCCESS;
}

// Scalar multiply (c * a)
MatrixStatus matrix_smult(Matrix* m, Matrix* a, float c) {

    // Multiply by Scalar
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = c * a->data[rows + col];
        }
    }

    return MATRIX_SUCCESS;
}

// Matrix multiply (a * b)
MatrixStatus matrix_mmult(Matrix* m, Matrix* a, Matrix* b) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->rows || m->cols != b->cols || a->cols != b->rows)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    // Use cblas single precision generic matrix multiplication method
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->rows, m->cols, a->cols, 1, a->data, a->cols, b->data, b->cols, 0, m->data, m->cols);

    /*void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);*/
    /*
    // Multiply matrices
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = (row * m->cols);
            m->data[rows + col] = 0;
            for (uint32_t n = 0; n < a->cols; n++) {
                m->data[rows + col] += a->data[(row * a->cols) + n] * b->data[(n * b->cols) + col];
            }
        }
    }*/

    return MATRIX_SUCCESS;
}

// Hadamard Product (a_i * b_i for all elements i)
MatrixStatus matrix_hprod(Matrix* m, Matrix* a, Matrix* b) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->rows || a->rows != b->rows ||
        m->cols != a->cols || a->cols != b->cols)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    // Multiply elements
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = a->data[rows + col] * b->data[rows + col];
        }
    }

    return MATRIX_SUCCESS;
}

// Transpose Matrix a, store in m
MatrixStatus matrix_transpose(Matrix* m, Matrix* a) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->cols || m->cols != a->rows)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = a->data[col * a->cols + row];
        }
    }

    return MATRIX_SUCCESS;
}

// Apply activation function to elements of matrix
MatrixStatus matrix_activation(Matrix* m, Matrix* a, float act(float)) {

    // First confirm matrices have compatible dimensions
    if (m->rows != a->rows || m->cols != a->cols)
        return MATRIX_ERROR_INVALID_DIMENSIONS;

    for (uint32_t row = 0; row < a->rows; row++) {
        for (uint32_t col = 0; col < a->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = act(a->data[rows + col]);
        }
    }

    return MATRIX_SUCCESS;
}

// Generate random number with guassian distribution using Box_Muller method
float random_guassian(void) {

    float u = (float)rand() / (float)RAND_MAX;
    float v = (float)rand() / (float)RAND_MAX;

    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

// Initialize all elements to random value
void matrix_initialize_random(Matrix* m) {

    // Initialize all values to random float between -1 and 1
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = -1.0 + 2.0*((float)(rand()) / (float)(RAND_MAX));
        }
    }
}

void matrix_zero(Matrix* m) {

    // Set all values to zero
    for (uint32_t row = 0; row < m->rows; row++) {
        for (uint32_t col = 0; col < m->cols; col++) {
            int rows = row * m->cols;
            m->data[rows + col] = 0;
        }
    }
}
