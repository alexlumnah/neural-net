#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "neural_net.h"
#include "matrix.h"

/* TODO
   x Use new error matrix inside each layer
   - Test save/load functionality
   - Allow each layer to have a unique activation function
   - Rename nodes to neurons
   - Different cost functions?

Eventually:
  - Plot cost over time
  - Plot image of neural network
  - Plot weights and activations as an image
*/

/*
typedef enum ActFun {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    ACT_SOFTMAX,
} ActFun;

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
*/

float sigmoid(float f) {
    return 1.0 / (1.0 + exp(-1.0 * f));
}

float sigmoid_prime(float f) {
    return sigmoid(f) * (1 - sigmoid(f));
}

float calculate_cost(Matrix* exp_output, Matrix* act_output) {

    assert(exp_output->rows == act_output->rows);
    assert(exp_output->cols == act_output->cols);

    float cost = 0.0;
    for (uint32_t n = 0; n < exp_output->rows * exp_output->cols; n++) {
        cost += pow(exp_output->data[n] - act_output->data[n], 2);
    }

    return cost;

}

// Instantiate neural network, using random values for weights and biases
NeuralNetwork create_neural_network(uint32_t num_inputs, uint32_t num_layers, const uint32_t* num_nodes) {

    NeuralNetwork n = {0};

    // Instantiate our neural network
    n.num_inputs = num_inputs;
    n.num_layers = num_layers;
    n.layers = calloc(sizeof(Layer), num_layers + 1);

    // First layer just stores inputs in activation matrix
    n.layers[0].num_nodes = num_inputs;
    n.layers[0].a = matrix_create(num_inputs, 1);
    
    // Then create remaining layers
    for (uint32_t i = 1; i <= num_layers; i++) {
        uint32_t nodes = num_nodes[i - 1];
        n.layers[i].num_nodes = nodes;
        n.layers[i].w = matrix_create(nodes, n.layers[i - 1].num_nodes);
        n.layers[i].b = matrix_create(nodes, 1);
        n.layers[i].z = matrix_create(nodes, 1);
        n.layers[i].a = matrix_create(nodes, 1);
        n.layers[i].e = matrix_create(nodes, 1);
        n.layers[i].w_g = matrix_create(nodes, n.layers[i - 1].num_nodes);
        n.layers[i].b_g = matrix_create(nodes, 1);
        matrix_initialize_random(n.layers[i].w);
        matrix_initialize_random(n.layers[i].b);
    }

    return n;
}

// Free memory allocated to neural network
void destroy_neural_network(NeuralNetwork n) {

    // Free weights and biases matrices
    for (uint32_t i = 0; i <= n.num_layers; i++) {
        matrix_destroy(n.layers[i].w);
        matrix_destroy(n.layers[i].b);
        matrix_destroy(n.layers[i].z);
        matrix_destroy(n.layers[i].a);
        matrix_destroy(n.layers[i].e);
        matrix_destroy(n.layers[i].w_g);
        matrix_destroy(n.layers[i].b_g);
    }

    // Free list of layers
    free(n.layers);

}

// Evaluate inputs to neural network, store in output if provided
void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output) {

    // Assert input and output have expected dimensions
    assert(input->rows == n.num_inputs);
    assert(input->cols == 1);

    if (output != NULL) {
        assert(output->rows == n.layers[n.num_layers].num_nodes);
        assert(output->cols == 1);
    }

    // Copy inputs into first layer
    matrix_copy(n.layers[0].a, input);

    // Propogate through remaining layers
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        // Multiply matrics
        matrix_mmult(n.layers[i].z, n.layers[i].w, n.layers[i - 1].a);
        matrix_add(n.layers[i].z, 1, n.layers[i].b);
        matrix_activation(n.layers[i].a, n.layers[i].z, sigmoid);
    }

    // If output matrix is provided, copy data to output
    if (output != NULL) {
        matrix_copy(output, n.layers[n.num_layers].a);
    }
}

// Apply Back propogation algorithm
void back_propogate(NeuralNetwork n, Matrix* input, Matrix* exp_output) {

    // Assert input and output have expected dimensions
    assert(input->rows == n.num_inputs);
    assert(input->cols == 1);
    assert(exp_output->rows == n.layers[n.num_layers].num_nodes);
    assert(exp_output->cols == 1);

    forward_propogate(n, input, NULL);

    // Calculate error for the last layer
    uint32_t L = n.num_layers;
    matrix_diff(n.layers[L].e, n.layers[L].a, exp_output);
    matrix_activation(n.layers[L].z, n.layers[L].z, sigmoid_prime);
    matrix_hprod(n.layers[L].e, n.layers[L].e, n.layers[L].z);

    // Calculate gradients
    matrix_cmult(n.layers[L].w_g, n.layers[L].e, false, n.layers[L-1].a, true, 1.0, 1.0);
    matrix_add(n.layers[L].b_g, 1.0, n.layers[L].e);

    // Now propogate backwards through remaining layers
    for (int i = n.num_layers - 1; i >= 1; i--) {
        // Calculate error for this layer
        matrix_cmult(n.layers[i].e, n.layers[i+1].w, true, n.layers[i+1].e, false, 1.0, 0.0);
        matrix_activation(n.layers[i].z, n.layers[i].z, sigmoid_prime);
        matrix_hprod(n.layers[i].e, n.layers[i].e, n.layers[i].z);
        
        // Calculate gradients
        matrix_cmult(n.layers[i].w_g, n.layers[i].e, false, n.layers[i-1].a, true, 1.0, 1.0);
        matrix_add(n.layers[i].b_g, 1.0, n.layers[i].e);
    }

}

// Use back-propogation to evaluate gradient, and adjust weights/biases
void gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta) {

    // First ensure the gradient matrices in the neural network are zeroed
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        matrix_zero(n.layers[i].w_g);
        matrix_zero(n.layers[i].b_g);
    }

    // Now for all training data, back propogate to calculate gradient
    for (uint32_t i = 0; i < training_size; i++) {
        back_propogate(n, training_inputs[i], expected_outputs[i]);
    }

    // Scale gradient by learning rate, and number of training inputs, then apply gradient
    float learning_rate = eta / (float)training_size;
    for (uint32_t i = 1; i <= n.num_layers; i++) {

        // Subtract gradients from parameters
        matrix_add(n.layers[i].w, -learning_rate, n.layers[i].w_g);
        matrix_add(n.layers[i].b, -learning_rate, n.layers[i].b_g);
    }

}

// Shuffle two input arrays identically
void shuffle_arrays(size_t array_size, Matrix** array1, Matrix** array2) {

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

// Apply stochastic gradient descent algorithm
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** training_outputs, size_t num_batches, float eta) {

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

void write_matrix(FILE* f, Matrix* m) {

    // Write rows
    uint32_t rows = htons(m->rows);
    fwrite(&rows, sizeof(rows), 1, f);

    // Write cols
    uint32_t cols = htons(m->cols);
    fwrite(&cols, sizeof(cols), 1, f);

    // Write data
    fwrite(m->data, sizeof(m->data[0]), m->rows * m->cols, f);
}

// Read matrix from file, and store in provided matrix
int read_matrix(FILE* f, Matrix* m) {

    size_t bytes;

    // Read rows
    uint32_t rows;
    bytes = fread(&rows, sizeof(rows), 1, f);
    rows = ntohs(rows);
    if (bytes != 1) return -1;
    
    // Read cols
    uint32_t cols;
    bytes = fread(&cols, sizeof(cols), 1, f);
    cols = ntohs(cols);
    if (bytes != 1) return -1;

    // Confirm matrix has proper dimensions
    if (m->rows != rows || m->cols != cols) {
        return -1;
    }

    // Read data
    bytes = fread(m->data, sizeof(m->data[0]), rows * cols, f);
    if (bytes != rows * cols) return -1;

    return 0;
}

// Save neural network to file
int save_neural_network(NeuralNetwork n, const char* path) {

    FILE *f = fopen(path, "wb");

    if (f == NULL) return -1;

    // Write number of inputs
    uint32_t num_inputs = htons(n.num_inputs);
    fwrite(&num_inputs, sizeof(num_inputs), 1, f);

    // Write total number of layers
    uint32_t num_layers = htons(n.num_layers);
    fwrite(&num_layers, sizeof(num_layers), 1, f);

    // Write total number of nodes in each layer
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        uint32_t num_nodes = htons(n.layers[i].num_nodes);
        fwrite(&num_nodes, sizeof(num_nodes), 1, f);
    }

    // Now save each parameter set
    for (uint32_t i = 1; i <= n.num_layers; i++) {

        // Write parameter matrices
        write_matrix(f, n.layers[i].w);
        write_matrix(f, n.layers[i].b);
    }

    return 0;
}

// Load neural network from file, store in pointer
int load_neural_network(const char* path, NeuralNetwork* np) {

    FILE *f = fopen(path, "rb");

    if (f == NULL) return -1;

    int bytes;

    // Write number of inputs
    uint32_t num_inputs;
    bytes = fread(&num_inputs, sizeof(num_inputs), 1, f);
    num_inputs = ntohs(num_inputs);
    if (bytes != 1) return -1;

    // Write total number of layers
    uint32_t num_layers;
    bytes = fread(&num_layers, sizeof(num_layers), 1, f);
    num_layers = ntohs(num_layers);
    if (bytes != 1) return -1;

    // Read total number of nodes in each layer
    uint32_t num_nodes[num_layers];
    for (uint32_t i = 0; i < num_layers; i++) {
        bytes = fread(&num_nodes[i], sizeof(num_nodes[0]), 1, f);
        num_nodes[i] = ntohs(num_nodes[i]);
        if (bytes != 1) return -1;
    }

    // Create the neural network
    NeuralNetwork n = create_neural_network(num_inputs, num_layers, num_nodes);

    // Now save each parameter set
    for (uint32_t i = 1; i <= num_layers; i++) {
        // Read weights
        int status = read_matrix(f, n.layers[i].w);
        if (status != 0) return -1;

        // Read biases
        status = read_matrix(f, n.layers[i].b);
        if (status != 0) return -1;
    }

    *np = n;

    return 0;
}

