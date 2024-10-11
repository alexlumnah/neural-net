#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdbool.h>

typedef struct Matrix {
    uint32_t rows, cols;    // Num Rows, Num Cols
    float* data;            // m x n Matrix - index with m*i + j
} Matrix;

void matrix_print(Matrix mat);                      // Print Matrix

Matrix matrix_create(int rows, int cols);           // Create Matrix
void matrix_destroy(Matrix* m);                     // Destroy Matrix

// Matrix Operations
void matrix_copy(Matrix m, Matrix a);
void matrix_add(Matrix m, float c, Matrix a);       // M = M + c*A
void matrix_sum(Matrix m, Matrix a, Matrix b);      // M = A + B
void matrix_diff(Matrix m, Matrix a, Matrix b);     // M = A - B
void matrix_smult(Matrix m, Matrix a, float c);     // M = c*A
void matrix_mmult(Matrix m, Matrix a, Matrix b);    // M = AB
void matrix_dmult(Matrix m, Matrix d, Matrix v);    // M = DV
void matrix_hprod(Matrix m, Matrix a, Matrix b);    // M_i = A_i * B_i
void matrix_transpose(Matrix m, Matrix a);          // M = A^T
void matrix_activation(Matrix m, Matrix a, float act(float)); // M = f(A)
void matrix_cmult(Matrix m, Matrix a, bool a_t, Matrix b, bool b_t, float alpha, float beta);   // M = alpha * A^(?T) B^(?T) + beta * M

// Matrix Mutations
void matrix_zero(Matrix m);                         // Initialize to zeroes
void matrix_ones(Matrix m);                         // Initialize to ones
void matrix_initialize_random(Matrix m);            // Initialize random
void matrix_initialize_gaussian(Matrix m, float mean, float stdev); // Initialize to random guassian

#endif // MATRIX_H

