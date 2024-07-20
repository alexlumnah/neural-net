#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdbool.h>

typedef struct Matrix {
    uint32_t rows, cols;    // Num Rows, Num Cols
    float* data;            // m x n Matrix - index with m*i + j
} Matrix;

void matrix_print(Matrix mat); // Print Matrix

Matrix matrix_create(int rows, int cols);  // Allocate a matrix, initialize to all zeroes
void matrix_destroy(Matrix* m);             // Destroy Matrix, freeing memory for data

// Matrix Operations
void matrix_copy(Matrix m, Matrix a);             // Copy elements from one matrix to another
void matrix_add(Matrix m, float c, Matrix a);     // Add Matrix to output matrix m = m + c * A
void matrix_sum(Matrix m, Matrix a, Matrix b);   // Compute Sum Matrices m = (a + b)
void matrix_diff(Matrix m, Matrix a, Matrix b);  // Compute Difference of Matrices m = (a - b)
void matrix_smult(Matrix m, Matrix a, float c);   // Scalar multiply m = (c * a)
void matrix_mmult(Matrix m, Matrix a, Matrix b); // Matrix multiply m = (a * b)
void matrix_cmult(Matrix m, Matrix a, bool a_t, Matrix b, bool b_t, float alpha, float beta); // Complex multiplication m = alpha * A * B + beta * C, where A/B are optionally transposed
void matrix_dmult(Matrix m, Matrix d, Matrix b); // Matrix multiply m = D * B, where D is a diagonal matrix and B is optionally transposed
void matrix_hprod(Matrix m, Matrix a, Matrix b); // Hadamard Product m_i = (a_i * b_i for all elements i)
void matrix_transpose(Matrix m, Matrix a);        // Transpose matrix
void matrix_activation(Matrix m, Matrix a, float act(float));// Apply activation function to elements of matrix

// Matrix Mutations
void matrix_initialize_random(Matrix m);               // Initialize all elements to random value
void matrix_initialize_gaussian(Matrix m, float mean, float stdev); // Initialize all elements to random gaussian value
void matrix_zero(Matrix m);                            // Zero elements in matrix
void matrix_ones(Matrix m);                            // Set all elements in a matrix to 1
#endif // MATRIX_H
