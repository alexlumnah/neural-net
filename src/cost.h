#ifndef COST_H
#define COST_H

#include "matrix.h"

typedef enum CostFun {
    COST_QUADRATIC,
    COST_CROSS_ENTROPY,
    COST_LOG_LIKELIHOOD,
} CostFun;

static const char* COST_STR[] = {
    "COST_QUADRATIC",
    "COST_CROSS_ENTROPY",
    "COST_LOG_LIKEILHOOD",
};

float cost_quadratic(Matrix* exp, Matrix* act);
void cost_grad_quadratic(Matrix* c_g, Matrix* exp, Matrix* act);

float cost_cross(Matrix* exp, Matrix* act);
void cost_grad_cross(Matrix* c_g, Matrix* exp, Matrix* act);

float cost_log(Matrix* exp, Matrix* act);
void cost_grad_log(Matrix* c_g, Matrix* exp, Matrix* act);

#endif  // COST_H

