#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"
#include "activation.h"
#include "cost.h"
#include "layer.h"

#define MAX_LAYERS (10)

typedef struct NeuralNetwork {
    Layer* layers[MAX_LAYERS];          // Array of layers
    uint32_t layer_count;   // Number of layers in the network

    // Cost Function Variables
    CostFun cost_type;      // Cost Function
    CostPtr cost_fun;       // Cost Function Pointer
    CostGradPtr cost_grad;  // Cost Gradient Pointer

    int apply_dropout;      // Flag to apply dropout during training
} NeuralNetwork;

void print_neural_network(NeuralNetwork n);

NeuralNetwork create_neural_network(uint32_t rows, uint32_t cols, CostFun cost_type);
void destroy_neural_network(NeuralNetwork n);

void fully_connected_layer(NeuralNetwork* n, ActFun type, uint32_t rows, uint32_t cols);
void convolutional_layer(NeuralNetwork* n, ActFun type, uint32_t num_maps, uint32_t field_rows, uint32_t field_cols);

void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output);
void back_propogate(NeuralNetwork n, Matrix* exp_output, Matrix* act_output);
void gradient_descent(NeuralNetwork n, size_t batch_size, size_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta);
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** training_outputs, size_t batch_size, float eta);

void set_l1_reg(NeuralNetwork *n, uint32_t l, float lambda);
void set_l2_reg(NeuralNetwork *n, uint32_t l, float lambda);
void set_drop_out(NeuralNetwork *n, float drop_rate);

void init_drop_masks(NeuralNetwork n);
void update_drop_masks(NeuralNetwork n);
void scale_weights_up(NeuralNetwork n);
void scale_weights_down(NeuralNetwork n);

int save_neural_network(NeuralNetwork n, const char* path);
int load_neural_network(const char* path, NeuralNetwork* np);

#endif  // NEURAL_NET_H

