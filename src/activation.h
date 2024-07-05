#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

typedef enum ActFun {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    ACT_SOFTMAX,
} ActFun;

static const char* ACT_STR[] = {
    "ACT_SIGMOID",
    "ACT_RELU",
    "ACT_TANH",
    "ACT_SOFT_MAX",
};

void act_sigmoid(Matrix* dst, Matrix* src);
void act_sigmoid_prime(Matrix* dst, Matrix* src);

void act_relu(Matrix* dst, Matrix* src);
void act_relu_prime(Matrix* dst, Matrix* src);

void act_tanh(Matrix* dst, Matrix* src);
void act_tanh_prime(Matrix* dst, Matrix* src);

void act_softmax(Matrix* dst, Matrix* src);
void act_softmax_prime(Matrix* dst, Matrix* src);

#endif // ACTIVATIONS_H

