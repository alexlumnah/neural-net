#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "neural_net.h"
#include "matrix.h"

/*
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
    Layer* layers;        // Array of layers
} NeuralNetwork;
*/

float sigmoid(float f) {
    return 1.0 / (1.0 + exp(-1.0 * f));
}

float sigmoid_prime(float f) {
    return sigmoid(f) * (1 - sigmoid(f));
}

// Instantiate neural network, using random values for weights and biases
NeuralNetwork create_neural_network(uint32_t num_inputs, uint32_t num_layers, const uint32_t* num_nodes) {

    NeuralNetwork n = {0};

    // Instantiate our neural network
    n.num_inputs = num_inputs;
    n.num_layers = num_layers;
    n.layers = calloc(sizeof(Layer), num_layers);

    // Create first layer with random values for weights and biases
    n.layers[0].num_nodes = num_nodes[0];
    n.layers[0].weights = matrix_create(num_nodes[0], num_inputs);
    matrix_initialize_random(n.layers[0].weights);
    n.layers[0].biases = matrix_create(num_nodes[0],1);
    matrix_initialize_random(n.layers[0].biases);
    n.layers[0].zed = matrix_create(num_nodes[0],1);
    n.layers[0].output = matrix_create(num_nodes[0],1);
    n.layers[0].w_grad = matrix_create(num_nodes[0], num_inputs);
    n.layers[0].b_grad = matrix_create(num_nodes[0],1);

    // Then create remaining layers
    for (uint32_t i = 1; i < num_layers; i++) {
        n.layers[i].num_nodes = num_nodes[i];
        n.layers[i].weights = matrix_create(num_nodes[i], num_nodes[i-1]);
        matrix_initialize_random(n.layers[i].weights);
        n.layers[i].biases = matrix_create(num_nodes[i], 1);
        matrix_initialize_random(n.layers[i].biases);
        n.layers[i].zed = matrix_create(num_nodes[i], 1);
        n.layers[i].output = matrix_create(num_nodes[i], 1);
        n.layers[i].w_grad = matrix_create(num_nodes[i], num_nodes[i-1]);
        n.layers[i].b_grad = matrix_create(num_nodes[i], 1);
    }

    return n;
}

void destroy_neural_network(NeuralNetwork n) {

    // Free weights and biases matrices
    for (uint32_t i = 0; i < n.num_layers; i++) {
        matrix_destroy(n.layers[i].weights);
        matrix_destroy(n.layers[i].biases);
    }

    // Free list of layers
    free(n.layers);

}

void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output) {

    // TODO: Add dimension checking

    // Propogate inputs through first layer
    matrix_mmult(n.layers[0].zed, n.layers[0].weights, input);
    matrix_add(n.layers[0].zed, n.layers[0].zed, n.layers[0].biases);
    matrix_activation(n.layers[0].output, n.layers[0].zed, sigmoid);

    // Propogate through remaining layers
    for (uint32_t i = 1; i < n.num_layers; i++) {
        // Multiply matrics
        matrix_mmult(n.layers[i].zed, n.layers[i].weights, n.layers[i - 1].output);
        matrix_add(n.layers[i].zed, n.layers[i].zed, n.layers[i].biases);
        matrix_activation(n.layers[i].output, n.layers[i].zed, sigmoid);
    }

    // Now copy data to output
    for (uint32_t i = 0; i < n.layers[n.num_layers - 1].num_nodes; i++) {
        output->data[i] = n.layers[n.num_layers - 1].output->data[i];
    }

}

// Apply Back propogation algorithm
void back_propogate(NeuralNetwork n, Matrix* input, Matrix* exp_output) {

    Matrix* act_output = matrix_create(exp_output->rows, exp_output->cols);

    forward_propogate(n, input, act_output);

    // Calculate error for the last layer
    Matrix* err_last = matrix_create(exp_output->rows, exp_output->cols);
    matrix_sub(err_last, act_output, exp_output);
    matrix_activation(n.layers[n.num_layers - 1].zed, n.layers[n.num_layers - 1].zed, sigmoid_prime);
    matrix_hprod(err_last, err_last, n.layers[n.num_layers - 1].zed);

    // Calculate gradients
    Matrix* w_grad = matrix_create(n.layers[n.num_layers - 1].weights->rows, n.layers[n.num_layers - 1].weights->cols);
    Matrix* a_in_t = matrix_create(n.layers[n.num_layers - 2].output->cols, n.layers[n.num_layers - 2].output->rows);
    matrix_transpose(a_in_t, n.layers[n.num_layers - 2].output);
    matrix_mmult(w_grad, err_last, a_in_t);
    matrix_add(n.layers[n.num_layers - 1].w_grad, n.layers[n.num_layers - 1].w_grad, w_grad);
    matrix_add(n.layers[n.num_layers - 1].b_grad, n.layers[n.num_layers - 1].b_grad, err_last);
    matrix_destroy(w_grad);
    matrix_destroy(a_in_t);

    // Now propogate backwards through remaining layers
    Matrix* err;
    for (int i = n.num_layers - 2; i >= 1; i--) {
        // Calculate error for this layer
        err = matrix_create(n.layers[i].output->rows, n.layers[i].output->cols);
        Matrix* w_t = matrix_create(n.layers[i+1].weights->cols, n.layers[i+1].weights->rows);
        matrix_transpose(w_t, n.layers[i+1].weights);
        matrix_mmult(err, w_t, err_last);
        matrix_activation(n.layers[i].zed, n.layers[i].zed, sigmoid_prime);
        matrix_hprod(err, err, n.layers[i].zed);
        matrix_destroy(w_t);
        
        // Calculate gradients
        w_grad = matrix_create(n.layers[i].weights->rows, n.layers[i].weights->cols);
        a_in_t = matrix_create(n.layers[i-1].output->cols, n.layers[i-1].output->rows);
        matrix_transpose(a_in_t, n.layers[i-1].output);
        matrix_mmult(w_grad, err, a_in_t);
        matrix_add(n.layers[i].w_grad, n.layers[i].w_grad, w_grad);
        matrix_add(n.layers[i].b_grad, n.layers[i].b_grad, err);
        matrix_destroy(w_grad);
        matrix_destroy(a_in_t);

        matrix_destroy(err_last);
        err_last = err;
    }

    // Now propogate through first layer
    err = matrix_create(n.layers[0].output->rows, n.layers[0].output->cols);
    Matrix* w_t = matrix_create(n.layers[1].weights->cols, n.layers[1].weights->rows);
    matrix_transpose(w_t, n.layers[1].weights);
    matrix_mmult(err, w_t, err_last);
    matrix_activation(n.layers[0].zed, n.layers[0].zed, sigmoid_prime);
    matrix_hprod(err, err, n.layers[0].zed);
    matrix_destroy(w_t);
    
    // Calculate gradients
    w_grad = matrix_create(n.layers[0].weights->rows, n.layers[0].weights->cols);
    a_in_t = matrix_create(input->cols, input->rows);
    matrix_transpose(a_in_t, input);
    matrix_mmult(w_grad, err, a_in_t);
    matrix_add(n.layers[0].w_grad, n.layers[0].w_grad, w_grad);
    matrix_add(n.layers[0].b_grad, n.layers[0].b_grad, err);
    matrix_destroy(w_grad);
    matrix_destroy(a_in_t);

    matrix_destroy(err_last);
    matrix_destroy(err);
    matrix_destroy(act_output);

}

void gradient_descent(NeuralNetwork n, uint32_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta) {

    // First ensure the gradient matrices in the neural network are zeroed
    for (uint32_t i = 0; i < n.num_layers; i++) {
        matrix_zero(n.layers[i].w_grad);
        matrix_zero(n.layers[i].b_grad);
    }

    // Now for all training data, back propogate to calculate gradient
    for (uint32_t i = 0; i < training_size; i++) {
        back_propogate(n, training_inputs[i], expected_outputs[i]);
    }

    // Scale gradient by learning rate, and number of training inputs, then apply gradient
    float learning_rate = eta / (float)training_size;
    //printf("Learning rate: %f\n", learning_rate);
    for (uint32_t i = 0; i < n.num_layers; i++) {

        // Scale gradients
        matrix_smult(n.layers[i].w_grad, n.layers[i].w_grad, learning_rate);
        matrix_smult(n.layers[i].b_grad, n.layers[i].b_grad, learning_rate);

        // Subtract gradients from parameters
        matrix_sub(n.layers[i].weights, n.layers[i].weights, n.layers[i].w_grad);
        matrix_sub(n.layers[i].biases, n.layers[i].biases, n.layers[i].b_grad);
    }

}

void shuffle_arrays(uint32_t array_size, Matrix** array1, Matrix** array2) {

    int rand_loc;
    Matrix* m1;
    Matrix* m2;
    for (uint32_t i = 0; i < array_size; i++) {
        m1 = array1[i];
        m2 = array2[i];
        rand_loc = rand() % array_size;
        array1[i] = array1[rand_loc];
        array2[i] = array2[rand_loc];
        array1[rand_loc] = m1;
        array2[rand_loc] = m2;
    }
}


void stochastic_gradient_descent(NeuralNetwork n, uint32_t training_size, Matrix** training_inputs, Matrix** training_outputs, uint32_t num_batches, float eta) {

    // First get subsets of training data for training
    Matrix** inputs = calloc(sizeof(Matrix*), training_size);
    Matrix** outputs = calloc(sizeof(Matrix*), training_size);

    for (uint32_t i = 0; i < training_size; i++) {
        inputs[i] = training_inputs[i];
        outputs[i] = training_outputs[i];
    }

    // Now shuffle data
    shuffle_arrays(training_size, inputs, outputs);

    // For each batch, apply gradient descent algorithm
    uint32_t batch_size = training_size / num_batches;
    for (uint32_t i = 0; i < num_batches; i++) {
        gradient_descent(n, batch_size, inputs + i*batch_size, outputs + i*batch_size, eta);
    }

    // Free arrays
    free(inputs);
    free(outputs);

}
