#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"

typedef enum ActivationFunction {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
} ActivationFunction;

typedef struct Layer {
    uint32_t num_nodes; // Number of nodes
    Matrix* w;          // Weights
    Matrix* b;          // Biases
    Matrix* z;          // Z
    Matrix* a;          // Neuron Activation
    Matrix* e;          // Layer Error
    Matrix* w_g;        // Weight Gradient
    Matrix* b_g;        // Bias Gradient
} Layer;

typedef struct NeuralNetwork {
    uint32_t num_inputs;    // Number of inputs
    uint32_t num_layers;    // Number of layers in the network
    Layer* layers;          // Array of layers
} NeuralNetwork;

float calculate_cost(Matrix* exp_output, Matrix* act_output);

NeuralNetwork create_neural_network(uint32_t num_inputs, uint32_t num_layers, const uint32_t* num_nodes); // Instantiate neural network, using random values for weights and biases
void destroy_neural_network(NeuralNetwork n);   // Destroy neural network and all layers

void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output);
void back_propogate(NeuralNetwork n, Matrix* exp_output, Matrix* act_output);
void gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta);
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** training_outputs, size_t num_batches, float eta);

int save_neural_network(NeuralNetwork n, const char* path);
int load_neural_network(const char* path, NeuralNetwork* np);

#endif  // NEURAL_NET_H
