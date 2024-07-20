#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "neural_net.h"
#include "convolve.h"
#include "matrix.h"
#include "cost.h"

/* TODO
   - Combine layers into one struct
   - Consider separating activation functions into their own layer
   - Add depth to all layers
   - Better optimize the convolution functions. Explore FFT method
   - Store layer data, not pointers to layers inside neural net
   - Momentum-based gradient descent
   - Update save/load functionality
   - Model validation checks - e.g. need layer with output between 0 and 1 for cross entropy
*/


// Print Neural Network
void print_neural_network(NeuralNetwork n) {

    // Print a summary of the network
    printf("%u x %u -> {", n.layers[0]->rows, n.layers[0]->cols);
    for (uint32_t i = 1; i < n.layer_count - 1; i++) {
        printf("%u", n.layers[i]->size);
        if (i + 1 < n.layer_count - 1) printf(", ");
    }
    printf("} -> %u\n", n.layers[n.layer_count - 1]->size);
    
    // Cost Function
    printf("Cost Function: %s\n", COST_STR[n.cost_type]);

    // Print Details of Each Layer
    for (uint32_t i = 0; i < n.layer_count; i++) {
        Layer* layer = n.layers[i];
        printf("Layer %u:\n",i);
        printf("    Type: %s\n", LAYER_STR[layer->type]);
        printf("    Activation: %s\n", ACT_STR[layer->act_type]);
        printf("    Size: %u\n", layer->size);
        if (layer->type == LAYER_CONVOLUTIONAL)
            printf("    Dimensions: %u x %u x %u\n",CONV(layer)->num_maps, layer->rows, layer->cols);
        else
            printf("    Dimensions: %u x %u\n",layer->rows, layer->cols);
        printf("    L1 Reg: %f\n", layer->l1_reg);
        printf("    L2 Reg: %f\n", layer->l2_reg);
        printf("    Dropout: %f\n", layer->drop_rate);
    }
    
}

// Instantiate neural network, using random values for weights and biases
NeuralNetwork create_neural_network(uint32_t rows, uint32_t cols, CostFun cost_type) {

    NeuralNetwork n = {0};

    // Instantiate our neural network
    n.cost_type = cost_type;
    switch (cost_type) {
        case COST_QUADRATIC:
            n.cost_fun = cost_quadratic;
            n.cost_grad = cost_grad_quadratic;
            break;
        case COST_CROSS_ENTROPY:
            n.cost_fun = cost_cross;
            n.cost_grad = cost_grad_cross;
            break;
        case COST_LOG_LIKELIHOOD:
            n.cost_fun = cost_log;
            n.cost_grad = cost_grad_log;
            break;
        default:
            assert(false && "Undefined cost function");
            break;
    }

    // First layer just stores inputs in activation matrix
    Layer* layer = calloc(sizeof(Layer), 1);
    layer->type = LAYER_INPUT;
    layer->act_type = ACT_NONE;
    layer->size = rows * cols;
    layer->rows = rows;
    layer->cols = cols;
    layer->a = matrix_create(rows * cols, 1);

    n.layers[0] = layer;
    n.layer_count = 1;
    
    return n;
}

// Free memory allocated to neural network
void destroy_neural_network(NeuralNetwork n) {

    // Free weights and biases matrices
    for (uint32_t i = 0; i < n.layer_count; i++) {
        assert(false && "Not implemented yet");
    }

    // Free list of layers
    free(n.layers);

}

// Add a fully connected layer to the neural network
void fully_connected_layer(NeuralNetwork* n, ActFun type, uint32_t rows, uint32_t cols) {

    assert(n->layer_count < MAX_LAYERS);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(FullLayer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    uint32_t size = rows * cols;
    layer->type = LAYER_FULLY_CONNECTED;
    layer->size = size;
    layer->rows = rows;
    layer->cols = cols;
    layer->z = matrix_create(size, 1);
    layer->a = matrix_create(size, 1);
    layer->e = matrix_create(size, 1);
    layer->c_g = matrix_create(size, 1);

    // Regularization Parameters
    layer->l1_reg = 0.0f;
    layer->l2_reg = 0.0f;
    layer->drop_rate = 0.0f;

    // Set Activation Function
    layer->act_type = type;
    switch (type) {
        case ACT_SIGMOID:
            layer->a_j = matrix_create(size, 1);
            layer->act_fun = act_sigmoid;
            layer->act_pri = act_sigmoid_prime;
            break;
        case ACT_RELU:
            layer->a_j = matrix_create(size, 1);
            layer->act_fun = act_relu;
            layer->act_pri = act_relu_prime;
            break;
        case ACT_TANH:
            layer->a_j = matrix_create(size, 1);
            layer->act_fun = act_tanh;
            layer->act_pri = act_tanh_prime;
            break;
        case ACT_SOFTMAX:
            layer->a_j = matrix_create(size, size);
            layer->act_fun = act_softmax;
            layer->act_pri = act_softmax_prime;
            break;
        default:
            assert(false && "Undefined activation function");
            break;
    }

    // Create matrices
    uint32_t prev_size = n->layers[l - 1]->size;
    FULL(layer)->w = matrix_create(size, prev_size);
    FULL(layer)->b = matrix_create(size, 1);
    FULL(layer)->w_g = matrix_create(size, prev_size);
    FULL(layer)->b_g = matrix_create(size, 1);
    FULL(layer)->a_m = matrix_create(size, 1);
    FULL(layer)->s = matrix_create(prev_size, size);
    
    // Initialize weights as random guassian variables
    float mean = 0.0f;
    float stdev = 1.0f / sqrtf(n->layers[l - 1]->size);
    matrix_initialize_gaussian(FULL(layer)->w, mean, stdev);
    matrix_initialize_gaussian(FULL(layer)->b, 0.0, 1.0);

}

// Add a convolutional layer to the neural network
void convolutional_layer(NeuralNetwork* n, ActFun type, uint32_t num_maps, uint32_t field_rows, uint32_t field_cols) {

    assert(n->layer_count < MAX_LAYERS);
    assert(n->layers[n->layer_count - 1]->act_type != ACT_SOFTMAX);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(ConvLayer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    // Calculate number of rows and cols in this layer
    int s_rows = n->layers[l - 1]->rows - field_rows + 1;
    int s_cols = n->layers[l - 1]->cols - field_cols + 1;
    assert(s_rows > 0 && "Invalid receptive field dimensions");
    assert(s_cols > 0 && "Invalid receptive field dimensions");

    uint32_t rows = (uint32_t)s_rows;
    uint32_t cols = (uint32_t)s_cols;

    uint32_t size = rows * cols * num_maps;
    layer->type = LAYER_CONVOLUTIONAL;
    layer->size = size;
    layer->rows = rows;
    layer->cols = cols;
    layer->z = matrix_create(size, 1);
    layer->a = matrix_create(size, 1);
    layer->e = matrix_create(size, 1);
    layer->c_g = matrix_create(size, 1);
    layer->a_j = matrix_create(size, 1);
    
    // Set Activation Function
    layer->act_type = type;
    switch (type) {
        case ACT_SIGMOID:
            layer->act_fun = act_sigmoid;
            layer->act_pri = act_sigmoid_prime;
            break;
        case ACT_RELU:
            layer->act_fun = act_relu;
            layer->act_pri = act_relu_prime;
            break;
        case ACT_TANH:
            layer->act_fun = act_tanh;
            layer->act_pri = act_tanh_prime;
            break;
        case ACT_SOFTMAX:
            assert(!"Softmax not supported with convolutional layer");
            break;
        default:
            assert(!"Undefined activation function");
            break;
    }

    // Create feature maps
    CONV(layer)->num_maps = num_maps;
    CONV(layer)->map_w = calloc(sizeof(Matrix), num_maps);
    CONV(layer)->map_wg = calloc(sizeof(Matrix), num_maps);
    for (uint32_t i = 0; i < num_maps; i++) {
        // Create weight matrices
        CONV(layer)->map_w[i] = matrix_create(field_rows, field_cols);
        CONV(layer)->map_wg[i] = matrix_create(field_rows, field_cols);

        // Initialize weights
        float mean = 0.0f;
        float stdev = 1.0f / sqrtf(rows * cols);
        matrix_initialize_gaussian(CONV(layer)->map_w[i], mean, stdev);
    }
    CONV(layer)->map_b = matrix_create(num_maps, 1);
    CONV(layer)->map_bg = matrix_create(num_maps, 1);
    matrix_initialize_gaussian(CONV(layer)->map_b, 0.0, 1.0 / sqrtf(rows * cols));

}

// Add a max pooling layer to network
void max_pooling_layer(NeuralNetwork* n, uint32_t field_rows, uint32_t field_cols, uint32_t stride) {

    assert(n->layer_count < MAX_LAYERS);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(PoolLayer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    // Calculate number of rows and cols in this layer
    Layer* prev_layer = n->layers[l - 1];
    int s_rows = (prev_layer->rows - field_rows) / stride + 1;
    int s_cols = (prev_layer->cols - field_cols) / stride + 1;
    uint32_t depth = prev_layer->type == LAYER_CONVOLUTIONAL ? CONV(prev_layer)->num_maps : 1;
    assert(s_rows > 0 && "Invalid receptive field dimensions");
    assert(s_cols > 0 && "Invalid receptive field dimensions");

    uint32_t rows = (uint32_t)s_rows;
    uint32_t cols = (uint32_t)s_cols;

    uint32_t size = rows * cols * depth;
    layer->type = LAYER_MAX_POOLING;
    layer->act_type = ACT_NONE;
    layer->size = size;
    layer->rows = rows;
    layer->cols = cols;
    layer->z = matrix_create(size, 1);
    layer->a = matrix_create(size, 1);
    layer->e = matrix_create(size, 1);
    layer->a_j = matrix_create(size, 1);
    matrix_ones(layer->a_j);

    POOL(layer)->depth = depth;
    POOL(layer)->height = field_rows;
    POOL(layer)->width = field_cols;
    POOL(layer)->stride = stride;

}

// Shuffle two input arrays identically
static void shuffle_arrays(size_t array_size, Matrix* array1, Matrix* array2) {

    int rand_loc;
    Matrix m1;
    Matrix m2;
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

// Set L1 Regularization Parameter
void set_l1_reg(NeuralNetwork *n, uint32_t l, float lambda) {
    n->layers[l]->l1_reg = lambda;
}

// Set L2 Regularization Parameter
void set_l2_reg(NeuralNetwork *n, uint32_t l, float lambda) {
    n->layers[l]->l2_reg = lambda;
}

/*
// Set Dropout Rate
void set_drop_out(NeuralNetwork *n, float drop_rate) {
    assert(drop_rate >= 0.0f && drop_rate < 1.0f);
    n->drop_rate = drop_rate;
    init_drop_masks(*n);
}

// Initialize Drop Masks
void init_drop_masks(NeuralNetwork n) {
  
    // Randomly initialize percentage of masks to 1
    for (uint32_t l = 1; l < n.layer_count; l++) {
        matrix_zero(n.layers[l].a_m);
        uint32_t count = n.layers[l].size * (1.0f - n.drop_rate);
        for (uint32_t i = 0; i < count; i++) {
            n.layers[l].a_m.data[i] = 1.0f;
        }
    }
}

// Update Masks for Dropping Out Neurons
void update_drop_masks(NeuralNetwork n) {
  
    for (uint32_t l = 1; l < n.layer_count; l++) {
        // Shuffle existing mask to create a new one
        for (uint32_t i = 0; i < n.layers[l].size; i++) {
            float a_m = n.layers[l].a_m.data[i];
            int rand_loc = rand() % n.layers[l].size;
            n.layers[l].a_m.data[i] = n.layers[l].a_m.data[rand_loc];
            n.layers[l].a_m.data[rand_loc] = a_m;
        }
    }

}

// Scale Hidden Neuron Weights Up By Drop Out Factor
void scale_weights_up(NeuralNetwork n) {

    for (uint32_t l = 1; l < n.layer_count; l++) {
        int w_size = n.layers[l].w.rows * n.layers[l].w.cols;
        for (int i = 0; i < w_size; i++) {
            n.layers[l].w.data[i] = n.layers[l].w.data[i] / n.drop_rate;
        }
    }
}

// Scale Hidden Neuron Weights Down By Drop Out Factor
void scale_weights_down(NeuralNetwork n) {

    for (uint32_t l = 1; l < n.layer_count; l++) {
        int w_size = n.layers[l].w.rows * n.layers[l].w.cols;
        for (int i = 0; i < w_size; i++) {
            n.layers[l].w.data[i] = n.layers[l].w.data[i] * n.drop_rate;
        }
    }
}
*/

// Propogate through convolutional layer
static void propogate_conv(NeuralNetwork n, uint32_t l) {

    Layer* layer = n.layers[l];
    Layer* prev_layer = n.layers[l - 1];

    // Loop over each feature map
    for (uint32_t f = 0; f < CONV(layer)->num_maps; f++) {

        uint32_t map_size = layer->rows * layer->cols;
        Matrix z = {layer->rows, layer->cols, layer->z.data + f * map_size};
        Matrix a = {prev_layer->rows, prev_layer->cols, prev_layer->a.data};
        Matrix w = CONV(layer)->map_w[f];

        matrix_fft_convolve(z, a, w);

        // Now add b to each element
        float b = CONV(layer)->map_b.data[f];
        for (uint32_t i = 0; i < z.rows * z.cols; i++) z.data[i] += b;
    }

}

// Propogate through pooling layer
void propogate_pool(NeuralNetwork n, uint32_t l) {

    Layer* layer = n.layers[l];
    Layer* prev_layer = n.layers[l - 1];

    for (uint32_t d = 0; d < POOL(layer)->depth; d++) {

        // Temporary assert - only support pooling on convolutional layers
        assert(prev_layer->type == LAYER_CONVOLUTIONAL);

        uint32_t pool_map_size = layer->rows * layer->cols;
        uint32_t prev_map_size = prev_layer->rows * prev_layer->cols;
        Matrix a_out = {.rows = layer->rows,
                        .cols = layer->cols,
                        .data = layer->a.data + d * pool_map_size};
        Matrix a_in  = {.rows = prev_layer->rows,
                        .cols = prev_layer->cols,
                        .data = prev_layer->a.data + d * prev_map_size};
        uint32_t s = POOL(layer)->stride;
        for (uint32_t i = 0; i < a_out.rows; i++) {
            for (uint32_t j = 0; j < a_out.cols; j++) {
                float max = -INFINITY;
                for (uint32_t p = 0; p < POOL(layer)->height; p++) {
                    float* a_in_r  = a_in.data + (s * i + p) * a_in.cols;
                    for (uint32_t q = 0; q < POOL(layer)->width; q++) {
                        if (a_in_r[s * j + q] > max) max = a_in_r[s * j + q];
                    }
                }
                a_out.data[i * a_out.cols + j] = max;
            }
        }
    }
}

// Evaluate inputs to neural network, store in output if provided
void forward_propogate(NeuralNetwork n, Matrix input, Matrix output) {

    // Assert input and output have expected dimensions
    assert(input.rows == n.layers[0]->a.rows);
    assert(input.cols == n.layers[0]->a.cols);

    if (output.data != NULL) {
        assert(output.rows == n.layers[n.layer_count-1]->size);
        assert(output.cols == 1);
    }

    // Copy inputs into first layer
    matrix_copy(n.layers[0]->a, input);

    // Propogate through remaining layers
    for (uint32_t l = 1; l < n.layer_count; l++) {
        Layer* layer = n.layers[l];
        Layer* prev_layer = n.layers[l - 1];
        switch (n.layers[l]->type) {
        case LAYER_FULLY_CONNECTED: {
            // Multiply matrics
            matrix_mmult(layer->z, FULL(layer)->w, prev_layer->a);
            matrix_add(layer->z, 1, FULL(layer)->b);
            layer->act_fun(layer->a, layer->z);
            break;
        }
        case LAYER_CONVOLUTIONAL:
            propogate_conv(n, l);
            layer->act_fun(layer->a, layer->z);
            break;
        case LAYER_MAX_POOLING:
            propogate_pool(n, l);
            break;
        default:
            assert(false && "Invalid layer type");
        }
        
        /*
        // If dropping out neurons, apply mask now
        if (n.apply_dropout && i != n.layer_count) {
            matrix_hprod(n.layers[i].a, n.layers[i].a, n.layers[i].a_m);
        }
        */
    }

    // If output matrix is provided, copy data to output
    if (output.data != NULL) {
        matrix_copy(output, n.layers[n.layer_count-1]->a);
    }
}

// Compute layer error
void compute_error(NeuralNetwork n, uint32_t l) {

    Layer* layer = n.layers[l]; 

    // Compute jacobian of activation
    if (layer->act_type != ACT_NONE)
        layer->act_pri(layer->a_j, layer->a);

    // Case: Final Layer
    if (l == n.layer_count - 1) {
        assert(layer->type == LAYER_FULLY_CONNECTED);
        if (layer->a_j.cols == 1)
            matrix_hprod(layer->e, layer->a_j, layer->c_g);
        else
            matrix_cmult(layer->e,
                     layer->a_j, false, 
                     layer->c_g, false, 
                     1.0f, 0.0f);
        return;
    }

    // All other layers
    Layer* next_layer = n.layers[l + 1]; 
    switch (next_layer->type) {
        case LAYER_FULLY_CONNECTED:
            if (layer->a_j.cols == 1) {
                matrix_cmult(layer->e, 
                             FULL(next_layer)->w, true,
                             next_layer->e, false,
                             1.0, 0.0);
                matrix_hprod(layer->e, layer->e, layer->a_j);
            } else {
                matrix_cmult(FULL(next_layer)->s,
                         layer->a_j, false, 
                         FULL(next_layer)->w, true,
                         1.0, 0.0);
                matrix_cmult(layer->e,
                         FULL(next_layer)->s, false, 
                         next_layer->e, false,
                         1.0, 0.0);
            }
            break;
        case LAYER_CONVOLUTIONAL:
            matrix_zero(layer->e);
            for (uint32_t f = 0; f < CONV(next_layer)->num_maps; f++) {
                uint32_t map_size = next_layer->rows * next_layer->cols;
                Matrix e_curr = { layer->rows,
                                  layer->cols, 
                                  layer->e.data};
                Matrix e_next = { next_layer->rows,
                                  next_layer->cols, 
                                  next_layer->e.data + f * map_size };
                Matrix w = CONV(next_layer)->map_w[f];
                full_convolve(e_curr, e_next, w, false);
            }
            matrix_hprod(layer->e, layer->e, layer->a_j);
            break;
        case LAYER_MAX_POOLING:
            matrix_zero(layer->e);
            for (uint32_t d = 0; d < POOL(next_layer)->depth; d++) {

                assert(layer->act_type != ACT_SOFTMAX);
                uint32_t curr_map = layer->rows * layer->cols;
                uint32_t pool_map = next_layer->rows * next_layer->cols;
                Matrix e_curr = {.rows = layer->rows,
                                 .cols = layer->cols,
                                 .data = layer->e.data + d * curr_map};
                Matrix a_curr = {.rows = layer->rows,
                                 .cols = layer->cols,
                                 .data = layer->a.data + d * curr_map};
                Matrix e_next = {.rows = next_layer->rows,
                                 .cols = next_layer->cols,
                                 .data = next_layer->e.data + d * pool_map};
                Matrix a_next = {.rows = next_layer->rows,
                                 .cols = next_layer->cols,
                                 .data = next_layer->a.data + d * pool_map};
                
                // Iterate over each neuron in the next layer
                uint32_t s = POOL(next_layer)->stride;
                for (uint32_t i = 0; i < e_next.rows; i++) {
                for (uint32_t j = 0; j < e_next.cols; j++) {
                    float max = a_next.data[i * a_next.cols + j];
                    // Find the max neuron, and add the associated erro
                    for (uint32_t p = 0; p < POOL(next_layer)->height; p++) {
                    for (uint32_t q = 0; q < POOL(next_layer)->width; q++) {
                        uint32_t c = (s * i + p) * a_curr.cols + s * j + q;
                        if (a_curr.data[c] == max) 
                            e_curr.data[c] += e_next.data[i * e_next.cols + j];
                    }
                    }
                }
                }
                matrix_hprod(layer->e, layer->e, layer->a_j);
            }
            break;
        default:
            assert(!"Not implemented yet");
    }
}

// Compute gradient for layer
void compute_gradient(NeuralNetwork n, uint32_t l) {

    Layer* layer = n.layers[l]; 
    Layer* prev_layer = n.layers[l - 1]; 

    switch (layer->type) {
    case LAYER_FULLY_CONNECTED:
        // Calculate gradients
        matrix_cmult(FULL(layer)->w_g,
                     layer->e, false, 
                     prev_layer->a, true, 
                     1.0f, 1.0f);
        matrix_add(FULL(layer)->b_g, 1.0, layer->e);
        break;
    case LAYER_CONVOLUTIONAL:
        for (uint32_t f = 0; f < CONV(layer)->num_maps; f++) {

            // Calculate w gradient
            uint32_t map_size = layer->rows * layer->cols;
            Matrix w_g = CONV(layer)->map_wg[f];
            Matrix e = {layer->rows, layer->cols, layer->e.data + f * map_size};
            Matrix a = {prev_layer->rows, prev_layer->cols, prev_layer->a.data};
            rotate_convolve(w_g, a, e, false);

            // Calculate b gradient
            float* b_g = CONV(layer)->map_bg.data + f;
            for (uint32_t r = 0; r < layer->rows; r++) {
                float* e_r = e.data + r * layer->rows;
                for (uint32_t c = 0; c < layer->cols; c++) {
                    *b_g += e_r[c];
                }
            }
        }
        break;
    case LAYER_MAX_POOLING:
        break;  // No parameters
    default:
        assert(!"Not implemented yet");
    }

}

// Apply Back propogation algorithm
void back_propogate(NeuralNetwork n, Matrix input, Matrix exp_output) {

    // Assert input and output have expected dimensions
    assert(input.rows == n.layers[0]->a.rows);
    assert(input.cols == n.layers[0]->a.cols);
    assert(exp_output.rows == n.layers[n.layer_count-1]->size);
    assert(exp_output.cols == 1);

    forward_propogate(n, input, (Matrix){0});

    // Compute Cost Gradient
    uint32_t L = n.layer_count - 1;
    n.cost_grad(n.layers[L]->c_g, exp_output, n.layers[L]->a);

    // Compute Gradients for Each Layer
    for (int l = n.layer_count - 1; l >= 1; l--) {
        compute_error(n, l);
        compute_gradient(n, l);
    }

}

// Use back-propogation to evaluate gradient, and adjust weights/biases
void gradient_descent(NeuralNetwork n, size_t batch_size, size_t training_size, Matrix* training_inputs, Matrix* expected_outputs, float eta) {

    (void)training_size; // Supress compiler warnings

    // First ensure the gradient matrices in the neural network are zeroed
    for (uint32_t l = 1; l < n.layer_count; l++) {
        switch (n.layers[l]->type) {
        case LAYER_FULLY_CONNECTED:
            matrix_zero(FULL(n.layers[l])->w_g);
            matrix_zero(FULL(n.layers[l])->b_g);
            break;
        case LAYER_CONVOLUTIONAL:
            for (uint32_t f = 0; f < CONV(n.layers[l])->num_maps; f++) {
                matrix_zero(CONV(n.layers[l])->map_wg[f]);
            }
            matrix_zero(CONV(n.layers[l])->map_bg);
            break;
        case LAYER_MAX_POOLING:
            break;  // No parameters to train
        default:
            assert(!"Invalid layer type.");
        }
    }

    // Now for all training data, back propogate to calculate gradient
    for (uint32_t i = 0; i < batch_size; i++) {
        back_propogate(n, training_inputs[i], expected_outputs[i]);
    }

    // Scale gradient by learning rate, and number of training inputs, then apply gradient
    float learning_rate = eta / (float)batch_size;
    for (uint32_t l = 1; l < n.layer_count; l++) {

        // Subtract gradients from parameters
        switch (n.layers[l]->type) {
        case LAYER_FULLY_CONNECTED: {
            
            FullLayer* layer = FULL(n.layers[l]);

            // Apply l1 regularization, if applicable
            if (LAYER(layer)->l1_reg > 0.0f) {
                for (uint32_t j = 0; j < layer->w.rows; j++) {
                    float scale = eta * LAYER(layer)->l1_reg / (float)training_size;
                    float w = layer->w.data[j];
                    float sgn = (w > 0) - (w < 0);
                    layer->w.data[j] -= scale * sgn;
                }
            }

            // Apply l2 regularization, if applicable
            if (LAYER(layer)->l2_reg > 0.0f) {
                float scale = 1.0f - (eta * LAYER(layer)->l2_reg / (float)training_size);
                matrix_smult(layer->w, layer->w, scale);
            }

            matrix_add(layer->w, -learning_rate, layer->w_g);
            matrix_add(layer->b, -learning_rate, layer->b_g);
            break;
        }
        case LAYER_CONVOLUTIONAL: {
            ConvLayer* layer = CONV(n.layers[l]);
            for (uint32_t f = 0; f < layer->num_maps; f++) {
                // Apply l1 regularization, if applicable
                if (LAYER(layer)->l1_reg > 0.0f) {
                    for (uint32_t j = 0; j < layer->map_w[f].rows; j++) {
                        float scale = eta * LAYER(layer)->l1_reg / (float)training_size;
                        float w = layer->map_w[f].data[j];
                        float sgn = (w > 0) - (w < 0);
                        layer->map_w[f].data[j] -= scale * sgn;
                    }
                }

                // Apply l2 regularization, if applicable
                if (LAYER(layer)->l2_reg > 0.0f) {
                    float scale = 1.0f - (eta * LAYER(layer)->l2_reg / (float)training_size);
                    matrix_smult(layer->map_w[f], layer->map_w[f], scale);
                }
                matrix_add(layer->map_w[f], -learning_rate, layer->map_wg[f]);
            }
            matrix_add(layer->map_b, -learning_rate, layer->map_bg);
        }
            break;
        case LAYER_MAX_POOLING:
            break;  // No parameters to train
        default:
            assert(!"Not implemented yet");
        }
    }

}

// Apply stochastic gradient descent algorithm
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix* training_inputs, Matrix* training_outputs, size_t batch_size, float eta) {

    // First get subsets of training data for training
    Matrix* inputs = calloc(sizeof(Matrix), training_size);
    Matrix* outputs = calloc(sizeof(Matrix), training_size);

    for (uint32_t i = 0; i < training_size; i++) {
        inputs[i] = training_inputs[i];
        outputs[i] = training_outputs[i];
    }

    // Now shuffle data
    shuffle_arrays(training_size, inputs, outputs);

    /*
    // If dropping neurons, set flag to apply dropout and scale weights
    if (n.drop_rate > 0.0f) {
        n.apply_dropout = 1; 
        scale_weights_up(n);
    }
    */

    // For each batch, apply gradient descent algorithm
    size_t num_batches = training_size / batch_size;
    for (uint32_t i = 0; i < num_batches; i++) {
        //if (n.apply_dropout) update_drop_masks(n);
        gradient_descent(n, batch_size, training_size, inputs + i*batch_size, outputs + i*batch_size, eta);
    }

    /*
    // If dropping neurons, scale weights back down
    if (n.apply_dropout) {
        n.apply_dropout = 0;
        scale_weights_down(n);
    }
    */

    // Free arrays
    free(inputs);
    free(outputs);

}

void write_matrix(FILE* f, Matrix m) {

    // Write rows
    uint32_t rows = htons(m.rows);
    fwrite(&rows, sizeof(rows), 1, f);

    // Write cols
    uint32_t cols = htons(m.cols);
    fwrite(&cols, sizeof(cols), 1, f);

    // Write data
    fwrite(m.data, sizeof(m.data[0]), m.rows * m.cols, f);
}

// Read matrix from file, and store in provided matrix
int read_matrix(FILE* f, Matrix m) {

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
    if (m.rows != rows || m.cols != cols) {
        return -1;
    }

    // Read data
    bytes = fread(m.data, sizeof(m.data[0]), rows * cols, f);
    if (bytes != rows * cols) return -1;

    return 0;
}

/*
// Save neural network to file
int save_neural_network(NeuralNetwork n, const char* path) {

    FILE *f = fopen(path, "wb");

    if (f == NULL) return -1;

    // Write number of inputs
    uint32_t num_inputs = htons(n.num_inputs);
    fwrite(&num_inputs, sizeof(num_inputs), 1, f);

    // Write total number of layers
    uint32_t num_layers = htons(n.layer_count);
    fwrite(&num_layers, sizeof(num_layers), 1, f);

    // Write total number of neurons in each layer
    for (uint32_t i = 1; i <= n.layer_count; i++) {
        uint32_t size = htons(n.layers[i].size);
        fwrite(&size, sizeof(size), 1, f);
    }

    // Now save each parameter set
    for (uint32_t i = 1; i <= n.layer_count; i++) {

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

    // Read total number of neurons in each layer
    uint32_t size[num_layers];
    for (uint32_t i = 0; i < num_layers; i++) {
        bytes = fread(&size[i], sizeof(size[0]), 1, f);
        size[i] = ntohs(size[i]);
        if (bytes != 1) return -1;
    }

    // Create the neural network
    NeuralNetwork n = create_neural_network(num_inputs, num_layers, size);

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
*/

