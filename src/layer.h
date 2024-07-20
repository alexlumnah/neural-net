#ifndef LAYER_H
#define LAYER_H

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
    LayerType type;
    uint32_t size;          // Number of neurons
    uint32_t rows;          // Number of rows
    uint32_t cols;          // Number of cols
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
} Layer;

typedef struct FullLayer {
    Layer layer;

    // Fully Connected Layer
    Matrix w;               // Weights
    Matrix b;               // Biases
    Matrix w_g;             // Weight Gradient
    Matrix b_g;             // Bias Gradient
    Matrix a_m;             // Neuron Mask for Drop Out
    Matrix s;               // Scratch matrix for intermediate calc

} FullLayer;

typedef struct ConvLayer {
    Layer layer;

    // Convolutional Layer Parameters
    uint32_t num_maps;      // Number of Feature Maps
    Matrix* map_w;          // Feature Maps
    Matrix* map_wg;         // Map Gradients
    Matrix map_b;           // Map biases
    Matrix map_bg;          // Map bias gradients

} ConvLayer;

typedef struct PoolLayer {
    Layer layer;

    // Pool Layer Parameters
    uint32_t depth;         // Number of Feature Maps
    uint32_t width;
    uint32_t height;
    uint32_t stride;

} PoolLayer;

static const char* LAYER_STR[] = {
    "INPUT",
    "FULLY_CONNECTED",
    "CONVOLUTIONAL",
    "MAX_POOLING",
};

#define LAYER(l) ((Layer*)(l))
#define INPUT(l) ((InputLayer*)(l))
#define FULL(l)  ((FullLayer*)(l))
#define CONV(l)  ((ConvLayer*)(l))
#define POOL(l)  ((PoolLayer*)(l))

#endif // LAYER_H

