#include <math.h>
#include <assert.h>

#include "matrix.h"
#include "activation.h"

// Compute sigmoid of src, store in dst
void act_sigmoid(Matrix* a, Matrix* z) {
    assert(z->rows == a->rows);
    assert(z->cols == a->cols);

    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        a->data[i] = 1.0 / (1.0 + exp(-1.0 * z->data[i]));
}

// Compute sigmoid derivative of src, store in dst
void act_sigmoid_prime(Matrix* a_j, Matrix* a) {
    assert(a_j->rows = a->rows);
    assert(a_j->cols == 1);

    // All off diagonal entries are 0
    for (uint32_t i = 0; i < a_j->rows; i++) {
        float s = a->data[i];
        a_j->data[i] = s * (1 - s);
    }
}

void act_relu(Matrix* a, Matrix* z) {
    assert(z->rows == a->rows);
    assert(z->cols == a->cols);

    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        a->data[i] = z->data[i] <= 0.0f ? 0.0f : z->data[i];
}

void act_relu_prime(Matrix* a_j, Matrix* a) {
    assert(a_j->rows = a->rows);
    assert(a_j->cols == 1);

    // All of diagonal entries are 0
    for (uint32_t i = 0; i < a_j->rows; i++) {
        a_j->data[i] = a->data[i] <= 0.0f ? 0.0f : 1.0f;
    }
}

void act_tanh(Matrix* a, Matrix* z) {
    assert(z->rows == a->rows);
    assert(z->cols == a->cols);

    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        a->data[i] = tanhf(z->data[i]);
}

void act_tanh_prime(Matrix* a_j, Matrix* a) {
    assert(a_j->rows = a->rows);
    assert(a_j->cols == 1);

    // All of diagonal entries are 0
    for (uint32_t i = 0; i < a_j->rows; i++) {
        float t = a->data[i];
        a_j->data[i] = 1.0f - (t * t);
    }
}

void act_softmax(Matrix* a, Matrix* z) {
    assert(z->rows == a->rows);
    assert(z->cols == a->cols);

    // First find max
    float max = 0.0f;
    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        if (max < z->data[i]) max = z->data[i];

    // Calculate sum, subtracting max so we dont overflow
    float sum = 0.0f;
    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        sum += exp(z->data[i] - max);

    // Calculate activations
    for (uint32_t i = 0; i < z->rows * z->cols; i++)
        a->data[i] = exp(z->data[i] - max) / sum;
}

void act_softmax_prime(Matrix* a_j, Matrix* a) {
    assert(a->rows == a_j->rows);
    assert(a->rows == a_j->cols);

    for (uint32_t i = 0; i < a->rows * a->cols; i++) {
        for (uint32_t j = 0; j < a_j->cols; j++) {
            if (i == j) {
                a_j->data[i * a_j->cols + j] = a->data[j] * (1 - a->data[j]);
            }
            else {
                a_j->data[i * a_j->cols + j] = -1.0 * a->data[i] * a->data[j];
            }
        }
    }
}
