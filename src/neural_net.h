#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"

typedef struct Layer {
    uint32_t num_nodes; // Number of nodes in this layer
    Matrix* weights;    // Matrix containing input weights into this layer
    Matrix* biases;     // Matrix containing output bias for this layer
    Matrix* zed;        // Matrix storing result of weights * input + bias
    Matrix* output;     // Matrix for storing output of layer during computations
    Matrix* w_grad;     // Matrix for storing weight gradient during back propagation
    Matrix* b_grad;     // Matrix for storing bias gradient during back propagation
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
