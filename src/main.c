#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <complex.h>

#include "matrix.h"
#include "neural_net.h"
#include "mnist.h"
#include "ui.h"
#include "convolve.h"

#define count(list) sizeof(list)/sizeof(list[0])

// Find max element in a column matrix
int find_max(Matrix m) {
    
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


int main(void) {

    // Load training images
    size_t num_images;
    Matrix* training_images;
    num_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/train-images-idx3-ubyte", &training_images);

    size_t num_labels;
    uint8_t* training_labels;
    num_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/train-labels-idx1-ubyte", &training_labels);

    // Split out a validation set
    size_t num_val_images = 10000;
    num_images = num_images - num_val_images;
    Matrix* val_images = &training_images[num_images];
    uint8_t* val_labels = &training_labels[num_images];
    (void) val_images;  // Suppress compiler warnings when not using validation set
    (void) val_labels;  // Suppress compiler warnings when not using validation set

    // Generate matrix of expected outputs
    Matrix* expected_outputs = calloc(sizeof(Matrix), num_labels);
    for (size_t i = 0; i < num_labels; i++) {
        expected_outputs[i] = matrix_create(10, 1);
        expected_outputs[i].data[training_labels[i]] = 1.0;
    }

    // Load test set
    size_t num_test_images;
    Matrix* test_images;
    num_test_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/t10k-images-idx3-ubyte", &test_images);

    size_t num_test_labels;
    uint8_t* test_labels;
    num_test_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/t10k-labels-idx1-ubyte", &test_labels);

    // Generate matrix of expected outputs
    Matrix* expected_test_outputs = calloc(sizeof(Matrix), num_test_labels);
    for (size_t i = 0; i < num_test_labels; i++) {
        expected_test_outputs[i] = matrix_create(10, 1);
        expected_test_outputs[i].data[test_labels[i]] = 1.0;
    }

    /*
    // Extend training set by manipulating training inputs
    Matrix* variable_images;
    uint8_t* variable_labels;
    size_t num_variable_images;
    printf("Extending MNIST Set...\n");
    num_variable_images = extend_set(training_images, training_labels, num_images, &variable_images, &variable_labels);
    
    // Copy new images to end of training images list
    training_images = realloc(training_images, sizeof(Matrix) * (num_images + num_variable_images));
    expected_outputs = realloc(expected_outputs, sizeof(Matrix) * (num_images + num_variable_images));
    for (size_t i = 0; i < num_variable_images; i++) {
        training_images[num_images + i] = variable_images[i];
        expected_outputs[num_images + i] = expected_outputs[i];
    }
    num_images = num_images + num_variable_images;
    */

    int r = 10;
    printf("Random seed: %d\n", r);
    srand(r);

    // Create neural network
    NeuralNetwork* n = create_neural_network(COST_LOG_LIKELIHOOD);
    input_layer(n, 28, 28);
    convolutional_layer(n, ACT_RELU, 20, 5, 5);
    max_pooling_layer(n, 2, 2, 2);
    convolutional_layer(n, ACT_RELU, 20, 5, 5);
    max_pooling_layer(n, 2, 2, 2);
    fully_connected_layer(n, ACT_SIGMOID, 100);
    //set_l2_reg(n, 1, 5.0f);
    //set_drop_out(n, 1, 0.25);
    //fully_connected_layer(n, ACT_SIGMOID, 100, 1);
    fully_connected_layer(n, ACT_SOFTMAX, 10);

    // Lets train our network
    int num_epochs = 30;
    int batch_size = 10;
    float learning_rate = 0.1;
    int draw_display = 0;

    // Print out hyper parameters
    printf("Training Neural Network\n");
    print_neural_network(n);
    printf("Total Training Images: %lu\n", num_images);
    printf("Total Test Images: %lu\n", num_test_images);
    printf("Hyperparameters\n");
    printf("Epochs: %d\n", num_epochs);
    printf("Batch Size: %d\n", batch_size);
    printf("Learning Rates: %f\n\n", learning_rate);

    // Evaluate starting benchmark
    size_t success;
    float cost;
    evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
    printf("Starting Benchmark - Number Right: %zu Success Rate: %f Cost: %f\n", success, (float)success/(float)num_test_images, cost);

    
    // Initialize screen
    SDL_Rect net_box = RECT(800, 100, 600, 600);
    if (draw_display) {
        init_screen();
        clear_screen();
        draw_fully_connected_network(net_box, *n);
        display_screen();
        handle_inputs();
    }
   

    // Loop over each epoch
    for (int i = 0; i < num_epochs; i++) {

        // Train Network
        clock_t begin = clock();
        stochastic_gradient_descent(n, num_images, training_images, expected_outputs, batch_size, learning_rate);
        
        // Evaluate and print performance
        evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
        printf("Training Round %d Number Right: %zu Success Rate: %f Cost: %f\n", i, success, (float)success/(float)num_test_images, cost);
        printf("Time elapsed: %f\n", (double)(clock() - begin) / (double) CLOCKS_PER_SEC);

        // Draw updated neural network
        if (draw_display) {
            clear_screen();
            draw_fully_connected_network(net_box, *n);
            display_screen();
            handle_inputs();
        }
    }

    // Destroy screen
    if (draw_display) destroy_screen();

    // Save my neural network
    // save_neural_network(n, "test_neural_network_varied.txt");
    

    /*
    // Load neural network
    NeuralNetwork n;
    size_t success;
    float cost;
    load_neural_network("test_neural_network.txt", &n);
    evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
    printf("Training Round %d Number Right: %zu Success Rate: %f Cost: %f\n", 0, success, (float)success/(float)num_test_images, cost);
    */

    // Create window where you can draw numbers and classify them
    if (draw_display) {
        init_screen();

        Matrix output = matrix_create(10, 1);
        Matrix input = matrix_create(28, 28);
        uint8_t guess;

        create_digit_classifier_window(&guess, &input);

        while (handle_inputs()) {
            clear_screen();
            draw_fully_connected_network(net_box, *n);
            display_screen();

            // Guess input
            forward_propogate(n, input, output);
            guess = find_max(output);
        }
        
        destroy_screen();
    }

    return 0;
}
