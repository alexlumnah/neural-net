#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"
#include "activation.h"
#include "cost.h"

typedef void (*ActPtr)(Matrix*, Matrix*);
typedef float (*CostPtr)(Matrix*, Matrix*);
typedef void (*CostGradPtr)(Matrix*, Matrix*, Matrix*);

typedef struct Layer {
    uint32_t num_neurons;   // Number of nodes
    ActFun act_type;        // Activation Function
    ActPtr act_fun;         // Activation Function Pointer
    ActPtr act_pri;         // Derivative Pointer 
    Matrix* w;              // Weights
    Matrix* b;              // Biases
    Matrix* z;              // Z
    Matrix* a;              // Neuron Activation
    Matrix* e;              // Layer Error
    Matrix* s;              // Scratch matrix for intermediate calc
    Matrix* w_g;            // Weight Gradient
    Matrix* b_g;            // Bias Gradient
    Matrix* c_g;            // Cost Gradient
    Matrix* a_j;            // Activation jacobian
    Matrix* a_m;            // Neuron Mask for Drop Out
} Layer;

typedef struct NeuralNetwork {
    uint32_t num_inputs;    // Number of inputs
    uint32_t num_layers;    // Number of layers in the network
    Layer* layers;          // Array of layers

    // Cost Function Variables
    CostFun cost_type;      // Cost Function
    CostPtr cost_fun;       // Cost Function Pointer
    CostGradPtr cost_grad;  // Cost Gradient Pointer

    // Regularization Parameters
    float l1_reg;           // L1 Regularization Parameter
    float l2_reg;           // L2 Regularization Parameter
    float drop_rate;        // Dropout Rate
    int apply_dropout;      // Flag to apply dropout during training
} NeuralNetwork;

void print_neural_network(NeuralNetwork n);

NeuralNetwork create_neural_network(uint32_t num_inputs, uint32_t num_layers, const uint32_t* num_neurons); // Instantiate neural network, using random values for weights and biases
void destroy_neural_network(NeuralNetwork n);   // Destroy neural network and all layers

void forward_propogate(NeuralNetwork n, Matrix* input, Matrix* output);
void back_propogate(NeuralNetwork n, Matrix* exp_output, Matrix* act_output);
void gradient_descent(NeuralNetwork n, size_t batch_size, size_t training_size, Matrix** training_inputs, Matrix** expected_outputs, float eta);
void stochastic_gradient_descent(NeuralNetwork n, size_t training_size, Matrix** training_inputs, Matrix** training_outputs, size_t batch_size, float eta);

void set_act_fun(NeuralNetwork *n, uint32_t layer, ActFun type);
void set_cost_fun(NeuralNetwork *n, CostFun type);
void set_l1_reg(NeuralNetwork *n, float lambda);
void set_l2_reg(NeuralNetwork *n, float lambda);
void set_drop_out(NeuralNetwork *n, float drop_rate);

void init_drop_masks(NeuralNetwork n);
void update_drop_masks(NeuralNetwork n);
void scale_weights_up(NeuralNetwork n);
void scale_weights_down(NeuralNetwork n);

int save_neural_network(NeuralNetwork n, const char* path);
int load_neural_network(const char* path, NeuralNetwork* np);

#endif  // NEURAL_NET_H

