#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "neural_net.h"
#include "convolve.h"
#include "matrix.h"
#include "cost.h"

#define SLICE(layer, m, depth) ((Matrix){layer->rows, layer->cols, layer->m.data + layer->rows * layer->cols * depth})
#define MAT2D(layer, m) ((Matrix){layer->rows, layer->cols, layer->m.data})

/* TODO
   x Combine layers into one struct
   x Add depth to all layers
   x Better optimize the convolution functions. Explore FFT method
        x Store fft data in special struct
        x Update copys to be memcpys
   - Refactor and cleanup code
        x Make function calls consistent wrt pointer vs not
        x Show only public functions in header file
        x Update print function
        x Clean up drop out code - only scale if >0 layers have droprate
   - Momentum-based gradient descent
   - Update save/load functionality
   - Consider separating activation functions into their own layer
   - Consider adding stride and padding to convolutional layers
   - Consider inputs with depth
   - Model validation checks - e.g. need layer with output between 0 and 1 for cross entropy
*/

// Instantiate neural network, using random values for weights and biases
NeuralNetwork* create_neural_network(CostFun cost_type) {

    NeuralNetwork* n = calloc(1, sizeof(NeuralNetwork));

    // Instantiate our neural network
    n->cost_type = cost_type;
    switch (cost_type) {
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
        default:
            assert(false && "Undefined cost function");
            break;
    }

    return n;
}

// Free memory allocated to neural network
void destroy_neural_network(NeuralNetwork* n) {

    // Free weights and biases matrices
    for (uint32_t i = 0; i < n->layer_count; i++) {
        assert(false && "Not implemented yet");
    }

    // Free list of layers
    free(n->layers);

}

// Add input layer to neural network
void input_layer(NeuralNetwork* n, uint32_t rows, uint32_t cols) {

    assert(n->layer_count == 0);

    // First layer just stores inputs in activation matrix
    Layer* layer = calloc(sizeof(Layer), 1);
    layer->type = LAYER_INPUT;
    layer->act_type = ACT_NONE;
    layer->rows = rows;
    layer->cols = cols;
    layer->depth = 1;
    layer->size = rows * cols;
    layer->a = matrix_create(rows * cols, 1);

    n->layers[0] = layer;
    n->layer_count = 1;
}

// Add a fully connected layer to the neural network
void fully_connected_layer(NeuralNetwork* n, ActFun type, uint32_t size) {

    assert(n->layer_count > 0);
    assert(n->layer_count < MAX_LAYERS);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(Layer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    layer->type = LAYER_FULLY_CONNECTED;
    layer->size = size;
    layer->rows = size;
    layer->cols = 1;
    layer->depth = 1;
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
    layer->w = matrix_create(size, prev_size);
    layer->b = matrix_create(size, 1);
    layer->w_g = matrix_create(size, prev_size);
    layer->b_g = matrix_create(size, 1);
    layer->a_m = matrix_create(size, 1);
    layer->s = matrix_create(prev_size, size);
    
    // Initialize weights as random guassian variables
    float mean = 0.0f;
    float stdev = 1.0f / sqrtf(n->layers[l - 1]->size);
    matrix_initialize_gaussian(layer->w, mean, stdev);
    matrix_initialize_gaussian(layer->b, 0.0, 1.0);

}

// Add a convolutional layer to the neural network
void convolutional_layer(NeuralNetwork* n, ActFun type, uint32_t num_maps, uint32_t map_height, uint32_t map_width) {

    assert(n->layer_count > 0);
    assert(n->layer_count < MAX_LAYERS);
    assert(n->layers[n->layer_count - 1]->act_type != ACT_SOFTMAX);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(Layer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    Layer* prev_layer = n->layers[l - 1];

    // Calculate number of rows and cols in this layer
    int s_rows = prev_layer->rows - map_height + 1;
    int s_cols = prev_layer->cols - map_width + 1;
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
    layer->depth = num_maps;
    layer->height = map_height;
    layer->width = map_width;
    layer->map_w = calloc(num_maps, sizeof(Matrix));
    layer->map_wg = calloc(num_maps, sizeof(Matrix));
    layer->forw_conv = calloc(num_maps, sizeof(ConvPlan));
    layer->grad_conv = calloc(num_maps, sizeof(ConvPlan));
    layer->err_conv = calloc(num_maps, sizeof(ConvPlan));

    for (uint32_t i = 0; i < num_maps; i++) {
        // Create weight matrices, padded to size of a for faster computing
        layer->map_w[i] = matrix_create(map_height, map_width);
        layer->map_wg[i] = matrix_create(map_height, map_width);

        // Create convolution plans
        Matrix a = SLICE(prev_layer, a, 0);
        Matrix e = SLICE(layer, e, i);
        layer->forw_conv[i] = create_conv_plan(a,
                                               prev_layer->depth,
                                               layer->map_w[i],
                                               CONV_OVERLAP);
        layer->grad_conv[i] = create_conv_plan(a,
                                               prev_layer->depth,
                                               e,
                                               CONV_OVERLAP);
        e = SLICE(layer, e, 0);
        layer->err_conv[i] = create_conv_plan(e,
                                              layer->depth,
                                              layer->map_w[i],
                                              CONV_FULL);

        // Initialize weights
        float mean = 0.0f;
        float stdev = 1.0f / sqrtf(rows * cols);
        matrix_initialize_gaussian(layer->map_w[i], mean, stdev);

        // Update padded kernels in convoluton plans
        update_kernel(&layer->forw_conv[i]);
        update_kernel(&layer->grad_conv[i]);
        update_kernel(&layer->err_conv[i]);
    }

    // Create matrices to hold biases
    layer->map_b = matrix_create(num_maps, 1);
    layer->map_bg = matrix_create(num_maps, 1);
    matrix_initialize_gaussian(layer->map_b, 0.0, 1.0 / sqrtf(rows * cols));

}

// Add a max pooling layer to network
void max_pooling_layer(NeuralNetwork* n, uint32_t map_height, uint32_t map_width, uint32_t stride) {

    assert(n->layer_count > 0);
    assert(n->layer_count < MAX_LAYERS);

    // Create layer
    uint32_t l = n->layer_count;
    Layer* layer = calloc(sizeof(Layer), 1);
    n->layers[l] = layer;
    n->layer_count += 1;

    // Calculate number of rows and cols in this layer
    Layer* prev_layer = n->layers[l - 1];
    int s_rows = (prev_layer->rows - map_height) / stride + 1;
    int s_cols = (prev_layer->cols - map_width) / stride + 1;
    assert(s_rows > 0 && "Invalid receptive field dimensions");
    assert(s_cols > 0 && "Invalid receptive field dimensions");

    uint32_t rows = (uint32_t)s_rows;
    uint32_t cols = (uint32_t)s_cols;
    uint32_t depth = prev_layer->depth;

    uint32_t size = rows * cols * depth;
    layer->type = LAYER_MAX_POOLING;
    layer->act_type = ACT_NONE;
    layer->size = size;
    layer->rows = rows;
    layer->cols = cols;
    layer->depth = depth;
    layer->z = matrix_create(size, 1);
    layer->a = matrix_create(size, 1);
    layer->e = matrix_create(size, 1);
    layer->a_j = matrix_create(size, 1);
    matrix_ones(layer->a_j);

    layer->height = map_height;
    layer->width = map_width;
    layer->stride = stride;

}

// Set L1 Regularization Parameter
void set_l1_reg(NeuralNetwork* n, uint32_t l, float lambda) {

    assert(l > 0);
    assert(l < n->layer_count);

    n->layers[l]->l1_reg = lambda;
}

// Set L2 Regularization Parameter
void set_l2_reg(NeuralNetwork* n, uint32_t l, float lambda) {

    assert(l > 0);
    assert(l < n->layer_count);

    n->layers[l]->l2_reg = lambda;
}

// Initialize Drop Masks
static void init_drop_masks(NeuralNetwork* n, uint32_t l) {
  
    Layer* layer = n->layers[l];

    // Initialize percentage of masks to 1, to be shuffled later
    matrix_zero(layer->a_m);
    uint32_t count = layer->size * (1.0f - layer->drop_rate);
    for (uint32_t i = 0; i < count; i++) {
        layer->a_m.data[i] = 1.0f;
    }
}

// Set Dropout Rate
void set_drop_out(NeuralNetwork* n, uint32_t l, float drop_rate) {

    assert(l > 0);
    assert(l < n->layer_count);
    assert(n->layers[l]->type == LAYER_FULLY_CONNECTED);
    assert(drop_rate >= 0.0f && drop_rate < 1.0f);

    n->layers[l]->drop_rate = drop_rate;
    init_drop_masks(n, l);
}

// Update Masks for Dropping Out Neurons
static void update_drop_masks(NeuralNetwork* n) {
  
    for (uint32_t l = 1; l < n->layer_count; l++) {

        Layer* layer = n->layers[l];
        if (layer->drop_rate == 0.0f) continue;

        // Shuffle existing mask to create a new one
        for (uint32_t i = 0; i < layer->size; i++) {
            float a_m = layer->a_m.data[i];
            int rand_loc = rand() % layer->size;
            layer->a_m.data[i] = layer->a_m.data[rand_loc];
            layer->a_m.data[rand_loc] = a_m;
        }
    }

}

// Scale Hidden Neuron Weights Up By Drop Out Factor
static void scale_weights_up(NeuralNetwork* n) {

    for (uint32_t l = 1; l < n->layer_count; l++) {

        Layer* layer = n->layers[l];
        if (layer->drop_rate == 0.0f) continue;

        int w_size = layer->w.rows * layer->w.cols;
        for (int i = 0; i < w_size; i++) {
            layer->w.data[i] = layer->w.data[i] / layer->drop_rate;
        }
    }
}

// Scale Hidden Neuron Weights Down By Drop Out Factor
static void scale_weights_down(NeuralNetwork* n) {

    for (uint32_t l = 1; l < n->layer_count; l++) {

        Layer* layer = n->layers[l];
        if (layer->drop_rate == 0.0f) continue;

        int w_size = layer->w.rows * layer->w.cols;
        for (int i = 0; i < w_size; i++) {
            layer->w.data[i] = layer->w.data[i] * layer->drop_rate;
        }
    }
}

// Propogate through convolutional layer
static void propogate_conv(NeuralNetwork* n, uint32_t l) {

    Layer* layer = n->layers[l];

    // Loop over each feature map
    for (uint32_t f = 0; f < layer->depth; f++) {

        Matrix z = SLICE(layer, z, f);
        execute_overlap_conv(&layer->forw_conv[f], z, true);

        // Now add b to each element
        float b = layer->map_b.data[f];
        for (uint32_t i = 0; i < z.rows * z.cols; i++) z.data[i] += b;
    }

}

// Propogate through pooling layer
static void propogate_pool(NeuralNetwork* n, uint32_t l) {

    Layer* layer = n->layers[l];
    Layer* prev_layer = n->layers[l - 1];

    for (uint32_t d = 0; d < layer->depth; d++) {

        Matrix a_out = SLICE(layer, a, d);
        Matrix a_in  = SLICE(prev_layer, a, d);

        uint32_t s = layer->stride;
        for (uint32_t i = 0; i < a_out.rows; i++) {
            for (uint32_t j = 0; j < a_out.cols; j++) {
                float max = -INFINITY;
                for (uint32_t p = 0; p < layer->height; p++) {
                    float* a_in_r  = a_in.data + (s * i + p) * a_in.cols;
                    for (uint32_t q = 0; q < layer->width; q++) {
                        if (a_in_r[s * j + q] > max) max = a_in_r[s*j + q];
                    }
                }
                a_out.data[i * a_out.cols + j] = max;
            }
        }
    }
}

// Evaluate inputs to neural network, store in output if provided
void forward_propogate(NeuralNetwork* n, Matrix input, Matrix output) {

    // Assert input and output have expected dimensions
    assert(input.rows == n->layers[0]->rows);
    assert(input.cols == n->layers[0]->cols);

    if (output.data != NULL) {
        assert(output.rows == n->layers[n->layer_count-1]->size);
        assert(output.cols == 1);
    }

    // Copy inputs into first layer
    matrix_copy(MAT2D(n->layers[0], a), input);

    // Propogate through remaining layers
    for (uint32_t l = 1; l < n->layer_count; l++) {

        Layer* layer = n->layers[l];
        Layer* prev_layer = n->layers[l - 1];

        switch (n->layers[l]->type) {
        case LAYER_FULLY_CONNECTED: {
            // Multiply matrics
            matrix_mmult(layer->z, layer->w, prev_layer->a);
            matrix_add(layer->z, 1, layer->b);
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
            assert(!"Invalid layer type");
        }
        
        // If dropping out neurons, apply mask now
        if (n->apply_dropout && layer->drop_rate > 0.0f) {
            matrix_hprod(n->layers[l]->a, n->layers[l]->a, n->layers[l]->a_m);
        }
    }

    // If output matrix is provided, copy data to output
    if (output.data != NULL) {
        matrix_copy(output, n->layers[n->layer_count-1]->a);
    }
}

// Compute layer error
static void compute_error(NeuralNetwork* n, uint32_t l) {

    Layer* layer = n->layers[l]; 

    // Compute jacobian of activation
    if (layer->act_type != ACT_NONE)
        layer->act_pri(layer->a_j, layer->a);

    // Case: Final Layer
    if (l == n->layer_count - 1) {
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

    // Case: All other layers
    Layer* next_layer = n->layers[l + 1]; 

    switch (next_layer->type) {
    case LAYER_FULLY_CONNECTED:
        if (layer->a_j.cols == 1) {
            matrix_cmult(layer->e, 
                         next_layer->w, true,
                         next_layer->e, false,
                         1.0, 0.0);
            matrix_hprod(layer->e, layer->e, layer->a_j);
        } else {
            matrix_cmult(next_layer->s,
                     layer->a_j, false, 
                     next_layer->w, true,
                     1.0, 0.0);
            matrix_cmult(layer->e,
                     next_layer->s, false, 
                     next_layer->e, false,
                     1.0, 0.0);
        }
        break;

    case LAYER_CONVOLUTIONAL:
        matrix_zero(layer->e);
        for (uint32_t f = 0; f < next_layer->depth; f++) {
            Matrix e_curr = SLICE(layer, e, f);
            // I think this is calculating all errors and adding together
            // But we need to separate them out
            assert(false);
            execute_full_conv(&next_layer->err_conv[f], e_curr, false);
        }
        matrix_hprod(layer->e, layer->e, layer->a_j);
        break;

    case LAYER_MAX_POOLING:
        matrix_zero(layer->e);
        for (uint32_t d = 0; d < next_layer->depth; d++) {
            // Interpret neuron error and activation as 2d matrix
            Matrix e_curr = SLICE(layer, e, d);
            Matrix a_curr = SLICE(layer, a, d);
            Matrix e_next = SLICE(next_layer, e, d);
            Matrix a_next = SLICE(next_layer, a, d);
            
            // Iterate over each neuron in the next layer
            uint32_t s = next_layer->stride;
            for (uint32_t i = 0; i < e_next.rows; i++) {
            for (uint32_t j = 0; j < e_next.cols; j++) {
                float max = a_next.data[i * a_next.cols + j];
                // Find the max neuron, and add the associated error
                for (uint32_t p = 0; p < next_layer->height; p++) {
                for (uint32_t q = 0; q < next_layer->width; q++) {
                    uint32_t c = (s * i + p) * a_curr.cols + s * j + q;
                    if (a_curr.data[c] == max) 
                        e_curr.data[c] += e_next.data[i*e_next.cols + j];
                }
                }
            }
            }
            matrix_hprod(layer->e, layer->e, layer->a_j);
        }
        break;

    default:
        assert(!"Invalid layer type");
    }
}

// Compute gradient for layer
static void compute_gradient(NeuralNetwork* n, uint32_t l) {

    Layer* layer = n->layers[l]; 
    Layer* prev_layer = n->layers[l - 1]; 

    switch (layer->type) {
    case LAYER_FULLY_CONNECTED:
        // Calculate gradients
        matrix_cmult(layer->w_g,
                     layer->e, false, 
                     prev_layer->a, true, 
                     1.0f, 1.0f);
        matrix_add(layer->b_g, 1.0, layer->e);
        break;
    case LAYER_CONVOLUTIONAL:
        for (uint32_t f = 0; f < layer->depth; f++) {

            // Calculate w gradient
            Matrix w_g = layer->map_wg[f];
            update_kernel(&layer->grad_conv[f]);
            execute_rot_conv(&layer->grad_conv[f], w_g, false);

            // Calculate b gradient
            Matrix e = SLICE(layer, e, f);
            float* b_g = layer->map_bg.data + f;
            for (uint32_t r = 0; r < layer->rows; r++) {
                float* e_r = e.data + r * layer->cols;
                for (uint32_t c = 0; c < layer->cols; c++) {
                    *b_g += e_r[c];
                }
            }
        }
        break;
    case LAYER_MAX_POOLING:
        break;  // No parameters
    default:
        assert(!"Invalid layer type");
    }

}

// Apply back propogation algorithm
static void back_propogate(NeuralNetwork* n, Matrix input, Matrix exp_output) {

    // Assert input and output have expected dimensions
    assert(input.rows == n->layers[0]->rows);
    assert(input.cols == n->layers[0]->cols);
    assert(exp_output.rows == n->layers[n->layer_count-1]->size);
    assert(exp_output.cols == 1);

    forward_propogate(n, input, (Matrix){0});

    // Compute Cost Gradient
    uint32_t L = n->layer_count - 1;
    n->cost_grad(n->layers[L]->c_g, exp_output, n->layers[L]->a);

    // Compute Gradients for Each Layer
    for (int l = n->layer_count - 1; l >= 1; l--) {
        compute_error(n, l);
        compute_gradient(n, l);
    }

}

// Use back-propogation to evaluate gradient, and adjust weights/biases
void gradient_descent(NeuralNetwork* n, size_t batch_size, size_t training_size, Matrix* training_inputs, Matrix* expected_outputs, float eta) {

    (void)training_size; // Supress compiler warnings

    // Zero all gradient matrices
    for (uint32_t l = 1; l < n->layer_count; l++) {
        switch (n->layers[l]->type) {
        case LAYER_FULLY_CONNECTED:
            matrix_zero(n->layers[l]->w_g);
            matrix_zero(n->layers[l]->b_g);
            break;

        case LAYER_CONVOLUTIONAL:
            for (uint32_t f = 0; f < n->layers[l]->depth; f++) {
                matrix_zero(n->layers[l]->map_wg[f]);
            }
            matrix_zero(n->layers[l]->map_bg);
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

    // Scale gradient by learning rate, and number of training inputs
    // then apply gradient
    float learning_rate = eta / (float)batch_size;
    for (uint32_t l = 1; l < n->layer_count; l++) {

        // Subtract gradients from parameters
        switch (n->layers[l]->type) {
        case LAYER_FULLY_CONNECTED: {
            
            Layer* layer = n->layers[l];

            // Apply l1 regularization, if applicable
            if (layer->l1_reg > 0.0f) {
                for (uint32_t j = 0; j < layer->w.rows; j++) {
                    float scale = eta * layer->l1_reg / training_size;
                    float w = layer->w.data[j];
                    float sgn = (w > 0) - (w < 0);
                    layer->w.data[j] -= scale * sgn;
                }
            }

            // Apply l2 regularization, if applicable
            if (layer->l2_reg > 0.0f) {
                float scale = 1.0f - (eta * layer->l2_reg / training_size);
                matrix_smult(layer->w, layer->w, scale);
            }

            matrix_add(layer->w, -learning_rate, layer->w_g);
            matrix_add(layer->b, -learning_rate, layer->b_g);
            break;
        }
        case LAYER_CONVOLUTIONAL: {
            Layer* layer = n->layers[l];
            for (uint32_t f = 0; f < layer->depth; f++) {
                // Apply l1 regularization, if applicable
                if (layer->l1_reg > 0.0f) {
                    for (uint32_t j = 0; j < layer->map_w[f].rows; j++) {
                        float scale = eta * layer->l1_reg / training_size;
                        float w = layer->map_w[f].data[j];
                        float sgn = (w > 0) - (w < 0);
                        layer->map_w[f].data[j] -= scale * sgn;
                    }
                }

                // Apply l2 regularization, if applicable
                if (layer->l2_reg > 0.0f) {
                    float scale = 1.0f - (eta*layer->l2_reg/training_size);
                    matrix_smult(layer->map_w[f], layer->map_w[f], scale);
                }
                matrix_add(layer->map_w[f],-learning_rate,layer->map_wg[f]);
                update_kernel(&layer->forw_conv[f]);
                update_kernel(&layer->err_conv[f]);
            }
            matrix_add(layer->map_b, -learning_rate, layer->map_bg);
        }
            break;
        case LAYER_MAX_POOLING:
            break;  // No parameters to train
        default:
            assert(!"Invalid layer type.");
        }
    }

}

// Shuffle two input arrays identically
static void shuffle_arrays(size_t size, Matrix* array1, Matrix* array2) {
    int rand_loc;
    Matrix m1;
    Matrix m2;
    for (uint32_t i = 0; i < size; i++) {
        m1 = array1[i];
        m2 = array2[i];
        rand_loc = rand() % size;
        array1[i] = array1[rand_loc];
        array2[i] = array2[rand_loc];
        array1[rand_loc] = m1;
        array2[rand_loc] = m2;
    }
}

// Apply stochastic gradient descent algorithm
void stochastic_gradient_descent(NeuralNetwork* n, size_t training_size, Matrix* training_inputs, Matrix* training_outputs, size_t batch_size, float eta) {

    // First get subsets of training data for training
    Matrix* inputs = calloc(sizeof(Matrix), training_size);
    Matrix* outputs = calloc(sizeof(Matrix), training_size);

    for (uint32_t i = 0; i < training_size; i++) {
        inputs[i] = training_inputs[i];
        outputs[i] = training_outputs[i];
    }

    // Now shuffle data
    shuffle_arrays(training_size, inputs, outputs);

    // Check each layer for a non-zero dropout rate
    bool drop_neurons = false;
    for (uint32_t l = 1; l < n->layer_count; l++) {
        if (n->layers[l]->drop_rate > 0.0f) {
            drop_neurons = true;
            break;
        }
    }

    // If dropping neurons, set flag to apply dropout and scale weights
    if (drop_neurons) {
        n->apply_dropout = true; 
        scale_weights_up(n);
    }

    // For each batch, apply gradient descent algorithm
    size_t num_batches = training_size / batch_size;
    for (uint32_t i = 0; i < num_batches; i++) {
        if (drop_neurons) update_drop_masks(n);
        gradient_descent(n,
                         batch_size,
                         training_size,
                         inputs + i * batch_size,
                         outputs + i * batch_size,
                         eta);
    }

    // If dropping neurons, scale weights back down
    if (drop_neurons) {
        n->apply_dropout = 0;
        scale_weights_down(n);
    }

    // Free arrays
    free(inputs);
    free(outputs);
}

// Find max element in a column matrix
static int find_max(Matrix m) {
    
    float max = m.data[0];
    int loc = 0;
    for (uint32_t i = 1; i < m.rows; i++) {
        if (m.data[i] > max) {
            max = m.data[i];
            loc = i;
        }
    }
    return loc;
}

// Evaluate network, store number correct and cost inside pointers
// Assumes outputs are one-hot encoded, thus guess is represented by max
void evaluate_network(NeuralNetwork* n, size_t test_size, Matrix* test_inputs, Matrix* expected_outputs, size_t* num_correct, float* cost) {

    *num_correct = 0;
    *cost = 0;

    // Now evaluate every single input, test the output
    Matrix output = matrix_create(expected_outputs[0].rows,
                                  expected_outputs[0].cols);
    for (size_t i = 0; i < test_size; i++) {
        forward_propogate(n, test_inputs[i], output);
        if (find_max(expected_outputs[i]) == find_max(output))
            *num_correct = (*num_correct) + 1;
        *cost += n->cost_fun(expected_outputs[i], output);
    }
    matrix_destroy(&output);
}

// Print Neural Network
void print_neural_network(NeuralNetwork* n) {

    // Print a summary of the network
    if (n->layers[0]->cols > 1)
        printf("(%u x %u) -> {", n->layers[0]->rows, n->layers[0]->cols);
    else
        printf("%u -> {", n->layers[0]->size);
    for (uint32_t i = 1; i < n->layer_count - 1; i++) {
        // Print out layer
        Layer* l = n->layers[i];
        if (l->cols > 1)
            printf("(%u x %u)", l->rows, l->cols);
        else
            printf("%u", l->size);

        // Print out separator
        if (i + 1 < n->layer_count - 1) printf(", ");
    }
    printf("} -> %u\n", n->layers[n->layer_count - 1]->size);
    
    // Cost Function
    printf("Cost Function: %s\n", COST_STR[n->cost_type]);

    // Print Details of Each Layer
    for (uint32_t i = 0; i < n->layer_count; i++) {
        Layer* l = n->layers[i];
        printf("Layer %u:\n",i);
        printf("  Type: %s\n", LAYER_STR[l->type]);
        printf("  Activation: %s\n", ACT_STR[l->act_type]);
        printf("  Size: %u\n", l->size);
        printf("  Dimensions: %u x %u x %u\n", l->depth, l->rows, l->cols);
        printf("  Regularization Parameters:\n");
        printf("    L1: %f\n", l->l1_reg);
        printf("    L2 Reg: %f\n", l->l2_reg);
        printf("    Dropout: %f\n", l->drop_rate);
    }
    
}

/*
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

// Save neural network to file
int save_neural_network(NeuralNetwork* n, const char* path) {

    FILE *f = fopen(path, "wb");

    if (f == NULL) return -1;

    // Write number of inputs
    uint32_t num_inputs = htons(n->num_inputs);
    fwrite(&num_inputs, sizeof(num_inputs), 1, f);

    // Write total number of layers
    uint32_t num_layers = htons(n->layer_count);
    fwrite(&num_layers, sizeof(num_layers), 1, f);

    // Write total number of neurons in each layer
    for (uint32_t i = 1; i <= n->layer_count; i++) {
        uint32_t size = htons(n->layers[i].size);
        fwrite(&size, sizeof(size), 1, f);
    }

    // Now save each parameter set
    for (uint32_t i = 1; i <= n->layer_count; i++) {

        // Write parameter matrices
        write_matrix(f, n->layers[i].w);
        write_matrix(f, n->layers[i].b);
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
    NeuralNetwork* n = create_neural_network(num_inputs, num_layers, size);

    // Now save each parameter set
    for (uint32_t i = 1; i <= num_layers; i++) {
        // Read weights
        int status = read_matrix(f, n->layers[i].w);
        if (status != 0) return -1;

        // Read biases
        status = read_matrix(f, n->layers[i].b);
        if (status != 0) return -1;
    }

    *np = n;

    return 0;
}
*/

