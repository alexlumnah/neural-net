#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "neural_net.h"
#include "matrix.h"
#include "cost.h"

/* TODO
   x Use new error matrix inside each layer
   x Allow each layer to have a unique activation function
   x Rename nodes to neurons
   x Different cost functions
   x Make multiplication more efficient for diagonal matrices
   - Option for different drop-out rates per layer
   - Momentum-based gradient descent
   - Update save/load functionality
   - Model validation checks - e.g. need layer with output between 0 and 1 for cross entropy

Eventually:
  - Plot cost over time
  x Plot image of neural network
  - Plot weights and activations as an image
*/


// Print Neural Network
void print_neural_network(NeuralNetwork n) {

    // Print a summary of the network
    printf("%u -> {", n.num_inputs);
    for (uint32_t i = 1; i < n.num_layers; i++) {
        printf("%u", n.layers[i].num_neurons);
        if (i + 1 < n.num_layers) printf(", ");
    }
    printf("} -> %u\n", n.layers[n.num_layers].num_neurons);
    
    // Cost Function and Activation Functions
    printf("Cost Function: %s\n", COST_STR[n.cost_type]);
    printf("Activation Functions:\n");
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        printf("\t%u - %s\n", n.layers[i].num_neurons, ACT_STR[n.layers[i].act_type]);
    }

    // Print Regularization Settings
    printf("Regularization Settings:\n");
    printf("\tL1 Regularization Parameter: %f\n", n.l1_reg);
    printf("\tL2 Regularization Parameter: %f\n", n.l2_reg);
    printf("\tDrop Out: %f\n", n.drop_rate);
}

// Instantiate neural network, using random values for weights and biases
NeuralNetwork create_neural_network(uint32_t num_inputs, uint32_t num_layers, const uint32_t* num_neurons) {

    NeuralNetwork n = {0};

    // Instantiate our neural network
    n.num_inputs = num_inputs;
    n.num_layers = num_layers;
    n.layers = calloc(sizeof(Layer), num_layers + 1);
    set_cost_fun(&n, COST_QUADRATIC);
    n.l1_reg = 0.0f;
    n.l2_reg = 0.0f;
    n.drop_rate = 0.0f;

    // First layer just stores inputs in activation matrix
    n.layers[0].num_neurons = num_inputs;
    n.layers[0].a = matrix_create(num_inputs, 1);
    
    // Then create remaining layers
    for (uint32_t i = 1; i <= num_layers; i++) {
        uint32_t nodes = num_neurons[i - 1];
        n.layers[i].num_neurons = nodes;
        n.layers[i].w = matrix_create(nodes, n.layers[i - 1].num_neurons);
        n.layers[i].b = matrix_create(nodes, 1);
        n.layers[i].z = matrix_create(nodes, 1);
        n.layers[i].a = matrix_create(nodes, 1);
        n.layers[i].e = matrix_create(nodes, 1);
        n.layers[i].w_g = matrix_create(nodes, n.layers[i - 1].num_neurons);
        n.layers[i].b_g = matrix_create(nodes, 1);
        n.layers[i].c_g = matrix_create(nodes, 1);
        n.layers[i].a_j = matrix_create(nodes, nodes);
        n.layers[i].a_m = matrix_create(nodes, 1);
        if (i != n.num_layers) {
            n.layers[i].s = matrix_create(nodes, num_neurons[i]);
        }
        
        // Initialize weights as random guassian variables
        float mean = 0.0f;
        float stdev = 1.0f / sqrtf(n.layers[i-1].num_neurons);
        matrix_initialize_gaussian(n.layers[i].w, mean, stdev);
        matrix_initialize_gaussian(n.layers[i].b, 0.0, 1.0);
        set_act_fun(&n, i, ACT_SIGMOID);
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
        matrix_destroy(n.layers[i].s);
        matrix_destroy(n.layers[i].w_g);
        matrix_destroy(n.layers[i].b_g);
        matrix_destroy(n.layers[i].c_g);
        matrix_destroy(n.layers[i].a_j);
        matrix_destroy(n.layers[i].a_m);
    }

    // Free list of layers
    free(n.layers);

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

// Set Cost Function
void set_cost_fun(NeuralNetwork *n, CostFun type) {

    n->cost_type = type;
    switch (type) {
        case COST_QUADRATIC:
            n->cost_fun = cost_quadratic;
            n->cost_grad = cost_grad_quadratic;
            break;
        case COST_CROSS_ENTROPY:
            n->cost_fun = cost_cross;
            n->cost_grad = cost_grad_cross;
            break;
        case COST_LOG_LIKELIHOOD:
            n->cost_fun = cost_log;
            n->cost_grad = cost_grad_log;
            break;
    }
}

// Set Activation Function for Layer
void set_act_fun(NeuralNetwork *n, uint32_t layer, ActFun type) {

    assert(layer != 0 && layer <= n->num_layers);

    n->layers[layer].act_type = type;
    switch (type) {
        case ACT_SIGMOID:
            n->layers[layer].act_fun = act_sigmoid;
            n->layers[layer].act_pri = act_sigmoid_prime;
            break;
        case ACT_RELU:
            n->layers[layer].act_fun = act_relu;
            n->layers[layer].act_pri = act_relu_prime;
            break;
        case ACT_TANH:
            n->layers[layer].act_fun = act_tanh;
            n->layers[layer].act_pri = act_tanh_prime;
            break;
        case ACT_SOFTMAX:
            n->layers[layer].act_fun = act_softmax;
            n->layers[layer].act_pri = act_softmax_prime;
            break;
    }
}

// Set L1 Regularization Parameter
void set_l1_reg(NeuralNetwork *n, float lambda) {
    n->l1_reg = lambda;
}

// Set L2 Regularization Parameter
void set_l2_reg(NeuralNetwork *n, float lambda) {
    n->l2_reg = lambda;
}

// Set Dropout Rate
void set_drop_out(NeuralNetwork *n, float drop_rate) {
    assert(drop_rate >= 0.0f && drop_rate < 1.0f);
    n->drop_rate = drop_rate;
    init_drop_masks(*n);
}

// Initialize Drop Masks
void init_drop_masks(NeuralNetwork n) {
  
    // Randomly initialize percentage of masks to 1
    for (uint32_t l = 1; l < n.num_layers; l++) {
        matrix_zero(n.layers[l].a_m);
        uint32_t count = n.layers[l].num_neurons * (1.0f - n.drop_rate);
        for (uint32_t i = 0; i < count; i++) {
            n.layers[l].a_m->data[i] = 1.0f;
        }
    }
}

// Update Masks for Dropping Out Neurons
void update_drop_masks(NeuralNetwork n) {
  
    for (uint32_t l = 1; l < n.num_layers; l++) {
        // Shuffle existing mask to create a new one
        for (uint32_t i = 0; i < n.layers[l].num_neurons; i++) {
            float a_m = n.layers[l].a_m->data[i];
            int rand_loc = rand() % n.layers[l].num_neurons;
            n.layers[l].a_m->data[i] = n.layers[l].a_m->data[rand_loc];
            n.layers[l].a_m->data[rand_loc] = a_m;
        }
    }

}

// Scale Hidden Neuron Weights Up By Drop Out Factor
void scale_weights_up(NeuralNetwork n) {

    for (uint32_t l = 1; l < n.num_layers; l++) {
        int w_size = n.layers[l].w->rows * n.layers[l].w->cols;
        for (int i = 0; i < w_size; i++) {
            n.layers[l].w->data[i] = n.layers[l].w->data[i] / n.drop_rate;
        }
    }
}

// Scale Hidden Neuron Weights Down By Drop Out Factor
void scale_weights_down(NeuralNetwork n) {

    for (uint32_t l = 1; l < n.num_layers; l++) {
        int w_size = n.layers[l].w->rows * n.layers[l].w->cols;
        for (int i = 0; i < w_size; i++) {
            n.layers[l].w->data[i] = n.layers[l].w->data[i] * n.drop_rate;
        }
    }
}

// Evaluate inputs to neural network, store in output if provided
void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output) {

    // Assert input and output have expected dimensions
    assert(input->rows == n.num_inputs);
    assert(input->cols == 1);

    if (output != NULL) {
        assert(output->rows == n.layers[n.num_layers].num_neurons);
        assert(output->cols == 1);
    }

    // Copy inputs into first layer
    matrix_copy(n.layers[0].a, input);

    // Propogate through remaining layers
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        // Multiply matrics
        matrix_mmult(n.layers[i].z, n.layers[i].w, n.layers[i - 1].a);
        matrix_add(n.layers[i].z, 1, n.layers[i].b);
        n.layers[i].act_fun(n.layers[i].a, n.layers[i].z);
        
        // If dropping out neurons, apply mask now
        if (n.apply_dropout && i != n.num_layers) {
            matrix_hprod(n.layers[i].a, n.layers[i].a, n.layers[i].a_m);
        }
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
    assert(exp_output->rows == n.layers[n.num_layers].num_neurons);
    assert(exp_output->cols == 1);

    forward_propogate(n, input, NULL);

    // Compute Cost Gradient
    uint32_t L = n.num_layers;
    n.cost_grad(n.layers[L].c_g, exp_output, n.layers[L].a);

    // Now Compute Error
    n.layers[L].act_pri(n.layers[L].a_j, n.layers[L].a);
    switch (n.layers[L].act_type) {
        case ACT_SIGMOID:   // Intentional Fall Through
        case ACT_RELU:      // Intentional Fall Through
        case ACT_TANH: 
            // Take advantage of the fact that a_j is diagonal
            matrix_dmult(n.layers[L].e,
                         n.layers[L].a_j,
                         n.layers[L].c_g);
            break;
        case ACT_SOFTMAX:
            matrix_cmult(n.layers[L].e,
                         n.layers[L].a_j, false, 
                         n.layers[L].c_g, false, 
                         1.0f, 0.0f);
            break;
    }

    // Calculate gradients
    matrix_cmult(n.layers[L].w_g,
                 n.layers[L].e, false, 
                 n.layers[L-1].a, true, 
                 1.0f, 1.0f);
    matrix_add(n.layers[L].b_g, 1.0, n.layers[L].e);

    // Now propogate backwards through remaining layers
    for (int l = n.num_layers - 1; l >= 1; l--) {
        // Calculate error for this layer
        n.layers[l].act_pri(n.layers[l].a_j, n.layers[l].a);
        switch (n.layers[l].act_type) {
            case ACT_SIGMOID:   // Intentional Fall Through
            case ACT_RELU:      // Intentional Fall Through
            case ACT_TANH: 
                matrix_cmult(n.layers[l].e,
                             n.layers[l+1].w, true,
                             n.layers[l+1].e, false,
                             1.0, 0.0);
                // Take advantage of the fact that a_j is diagonal
                matrix_dmult(n.layers[l].e,
                             n.layers[l].a_j,
                             n.layers[l].e);
                break;
            case ACT_SOFTMAX: {
                matrix_cmult(n.layers[l].s,
                             n.layers[l].a_j, false, 
                             n.layers[l+1].w, true,
                             1.0, 0.0);
                matrix_cmult(n.layers[l].e,
                             n.layers[l].s, false, 
                             n.layers[l+1].e, false,
                             1.0, 0.0);
                break;
          }
        }
        
        // Calculate gradients
        matrix_cmult(n.layers[l].w_g,
                     n.layers[l].e, false,
                     n.layers[l-1].a, true,
                     1.0, 1.0);
        matrix_add(n.layers[l].b_g, 1.0, n.layers[l].e);
    }

}

// Use back-propogation to evaluate gradient, and adjust weights/biases
void gradient_descent(NeuralNetwork n, size_t batch_size, size_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta) {

    // First ensure the gradient matrices in the neural network are zeroed
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        matrix_zero(n.layers[i].w_g);
        matrix_zero(n.layers[i].b_g);
    }

    // Now for all training data, back propogate to calculate gradient
    for (uint32_t i = 0; i < batch_size; i++) {
        back_propogate(n, training_inputs[i], expected_outputs[i]);
    }

    // Scale gradient by learning rate, and number of training inputs, then apply gradient
    float learning_rate = eta / (float)batch_size;
    for (uint32_t i = 1; i <= n.num_layers; i++) {

        // Apply l1 regularization, if applicable
        if (n.l1_reg > 0.0f) {
            for (uint32_t j = 0; j < n.layers[i].w->rows; j++) {
                float scale = eta * n.l1_reg / (float)training_size;
                float w = n.layers[i].w->data[j];
                float sgn = (w > 0) - (w < 0);
                n.layers[i].w->data[j] -= scale * sgn;
            }
        }

        // Apply l2 regularization, if applicable
        if (n.l2_reg > 0.0f) {
            float scale = 1.0f - (eta * n.l2_reg / (float)training_size);
            matrix_smult(n.layers[i].w, n.layers[i].w, scale);
        }

        // Subtract gradients from parameters
        matrix_add(n.layers[i].w, -learning_rate, n.layers[i].w_g);
        matrix_add(n.layers[i].b, -learning_rate, n.layers[i].b_g);
    }

}

// Apply stochastic gradient descent algorithm
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** training_outputs, size_t batch_size, float eta) {

    // First get subsets of training data for training
    Matrix** inputs = calloc(sizeof(Matrix*), training_size);
    Matrix** outputs = calloc(sizeof(Matrix*), training_size);

    for (uint32_t i = 0; i < training_size; i++) {
        inputs[i] = training_inputs[i];
        outputs[i] = training_outputs[i];
    }

    // Now shuffle data
    shuffle_arrays(training_size, inputs, outputs);

    // If dropping neurons, set flag to apply dropout and scale weights
    if (n.drop_rate > 0.0f) {
        n.apply_dropout = 1; 
        scale_weights_up(n);
    }

    // For each batch, apply gradient descent algorithm
    size_t num_batches = training_size / batch_size;
    for (uint32_t i = 0; i < num_batches; i++) {
        if (n.apply_dropout) update_drop_masks(n);
        gradient_descent(n, batch_size, training_size, inputs + i*batch_size, outputs + i*batch_size, eta);
    }

    // If dropping neurons, scale weights back down
    if (n.apply_dropout) {
        n.apply_dropout = 0;
        scale_weights_down(n);
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
        uint32_t num_neurons = htons(n.layers[i].num_neurons);
        fwrite(&num_neurons, sizeof(num_neurons), 1, f);
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
    uint32_t num_neurons[num_layers];
    for (uint32_t i = 0; i < num_layers; i++) {
        bytes = fread(&num_neurons[i], sizeof(num_neurons[0]), 1, f);
        num_neurons[i] = ntohs(num_neurons[i]);
        if (bytes != 1) return -1;
    }

    // Create the neural network
    NeuralNetwork n = create_neural_network(num_inputs, num_layers, num_neurons);

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

