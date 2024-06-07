#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct Matrix {
    uint32_t rows, cols;    // Num Rows, Num Cols
    float data[];           // m x n Matrix - index with m*i + j
} Matrix;

typedef enum MatrixStatus {
    MATRIX_SUCCESS,
    MATRIX_ERROR_INVALID_DIMENSIONS,
} MatrixStatus;

void matrix_print(Matrix* mat); // Print Matrix

Matrix* matrix_create(int rows, int cols);  // Allocate a matrix, initialize to all zeroes
void matrix_destroy(Matrix* m);             // Destroy Matrix, freeing memory for data

// Matrix Operations
MatrixStatus matrix_add(Matrix* m, Matrix* a, Matrix* b);   // Add Matrices (a + b)
MatrixStatus matrix_sub(Matrix* m, Matrix* a, Matrix* b);   // Subtract Matrices (a - b)
MatrixStatus matrix_smult(Matrix* m, Matrix* a, float c);   // Scalar multiply (c * a)
MatrixStatus matrix_mmult(Matrix* m, Matrix* a, Matrix* b); // Matrix multiply (a * b)
MatrixStatus matrix_hprod(Matrix* m, Matrix* a, Matrix* b); // Hadamard Product (a_i * b_i for all elements i)
MatrixStatus matrix_transpose(Matrix* m, Matrix* a);        // Transpose matrix
MatrixStatus matrix_activation(Matrix* m, Matrix* a, float act(float));// Apply activation function to elements of matrix

// Matrix Mutations
void matrix_initialize_random(Matrix* m);               // Initialize all elements to random value
void matrix_zero(Matrix* m);                            // Zero elements in matrix
#endif // MATRIX_H
