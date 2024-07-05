#include <math.h>
#include <stdio.h>

#include "cost.h"
#include "assert.h"

float cost_quadratic(Matrix* exp, Matrix* act) {

    assert(exp->rows == act->rows);
    assert(exp->cols == act->cols);

    float cost = 0.0;
    for (uint32_t n = 0; n < exp->rows * exp->cols; n++) {
        float y = exp->data[n];
        float a = act->data[n];
        cost += powf(y - a, 2);
    }

    return cost;
}

void cost_grad_quadratic(Matrix* c_g, Matrix* exp, Matrix* act) {

    assert(exp->rows == act->rows && exp->rows == c_g->rows);
    assert(exp->cols == act->cols && exp->cols == c_g->cols);

    for (uint32_t n = 0; n < exp->rows * exp->cols; n++)
        c_g->data[n] = act->data[n] - exp->data[n];
    
}

float cost_cross(Matrix* exp, Matrix* act) {
    
    assert(exp->rows == act->rows);
    assert(exp->cols == act->cols);

    float cost = 0.0;
    for (uint32_t n = 0; n < exp->rows * exp->cols; n++) {

        float y = exp->data[n];
        float a = act->data[n];

        // Clamp values to prevent gradient from going to inf
        if (a <= 0.001f) a = .001;
        if (a >= 0.999f) a = .999;
        cost -= y * logf(a) + (1.0f - y) * logf(1.0f - a);
    }

    return cost;
}


void cost_grad_cross(Matrix* c_g, Matrix* exp, Matrix* act) {

    assert(exp->rows == act->rows && exp->rows == c_g->rows);
    assert(exp->cols == act->cols && exp->cols == c_g->cols);

    for (uint32_t n = 0; n < exp->rows * exp->cols; n++) {

        float y = exp->data[n];
        float a = act->data[n];

        // Clamp values to prevent gradient from going to inf
        if (a <= 0.001f) a = .001;
        if (a >= 0.999f) a = .999;
        c_g->data[n] = (a - y) / (a * (1.0f - a));

    }
    
}

float cost_log(Matrix* exp, Matrix* act) {
    
    assert(exp->rows == act->rows);
    assert(exp->cols == act->cols);

    float cost = 0.0;
    for (uint32_t n = 0; n < exp->rows * exp->cols; n++) {
        float y = exp->data[n];
        float a = act->data[n];
        cost += -y * logf(a);
    }

    return cost;
}


void cost_grad_log(Matrix* c_g, Matrix* exp, Matrix* act) {

    assert(exp->rows == act->rows && exp->rows == c_g->rows);
    assert(exp->cols == act->cols && exp->cols == c_g->cols);

    for (uint32_t n = 0; n < exp->rows * exp->cols; n++) {

        float y = exp->data[n];
        float a = act->data[n];

        // Clamp values to prevent gradient from going to inf
        if (a <= 0.001f) a = .001;
        c_g->data[n] = -y / a;

    }
    
}
