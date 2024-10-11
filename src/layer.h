#ifndef LAYER_H
#define LAYER_H

#include "convolve.h"

typedef void (*ActPtr)(Matrix, Matrix);
typedef float (*CostPtr)(Matrix, Matrix);
typedef void (*CostGradPtr)(Matrix, Matrix, Matrix);

typedef enum LayerType {
    LAYER_INPUT,
    LAYER_FULLY_CONNECTED,
    LAYER_CONVOLUTIONAL,
    LAYER_MAX_POOLING,
} LayerType;

typedef struct Layer {
    // General Parameters
    LayerType type;
    uint32_t rows;          // Number of rows
    uint32_t cols;          // Number of cols
    uint32_t depth;         // Depth of layer
    uint32_t size;          // Number of neurons
    Matrix z;               // Weighted Sum
    Matrix a;               // Neuron Activation
    Matrix e;               // Neuron Error
    Matrix c_g;             // Cost Gradient
    Matrix a_j;             // Activation jacobian
    
    // Activation Function
    ActFun act_type;        // Activation Function
    ActPtr act_fun;         // Activation Function Pointer
    ActPtr act_pri;         // Derivative Pointer 

    // Regularization Parameters
    float l1_reg;           // L1 Regularization Parameter
    float l2_reg;           // L2 Regularization Parameter
    float drop_rate;        // Dropout Rate

    // Fully Connected Layer
    Matrix w;               // Weights
    Matrix b;               // Biases
    Matrix w_g;             // Weight Gradient
    Matrix b_g;             // Bias Gradient
    Matrix a_m;             // Neuron Mask for Drop Out
    Matrix s;               // Scratch matrix for intermediate calc

    // Convolutional Layer Parameters
    Matrix* map_w;          // Feature Maps
    Matrix* map_wg;         // Map Gradients
    Matrix map_b;           // Map biases
    Matrix map_bg;          // Map bias gradients
    ConvPlan* forw_conv;    // Plans for forward convolution
    ConvPlan* grad_conv;    // Plans for gradient convolution
    ConvPlan* err_conv;     // Plan for error convolution

    // Pool Layer / Convolutional Layer Parameters
    uint32_t width;         // Width of pool window / feature map
    uint32_t height;        // Height of pool window / feature map
    uint32_t stride;        // Stride taken when pooling
} Layer;

static const char* LAYER_STR[] = {
    "INPUT",
    "FULLY_CONNECTED",
    "CONVOLUTIONAL",
    "MAX_POOLING",
};

#endif // LAYER_H

