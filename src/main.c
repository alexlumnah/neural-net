#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#include "matrix.h"
#include "neural_net.h"
#include "mnist.h"
#include "ui.h"

uint8_t guess;

// Find max element in a column matrix
int find_max(Matrix* m) {
    
    float max = m->data[0];
    int loc = 0;
    for (uint32_t i = 1; i < m->rows; i++) {
        if (m->data[i] > max) {
            max = m->data[i];
            loc = i;
        }
    }
    return loc;
}

// Evaluate network
void evaluate_network(NeuralNetwork n, size_t test_size, Matrix** test_inputs, Matrix** expected_outputs, size_t* num_correct, float* cost) {

    *num_correct = 0;
    *cost = 0;
    // Now evaluate every single input, test the output
    Matrix* output = matrix_create(expected_outputs[0]->rows, expected_outputs[0]->cols);
    for (size_t i = 0; i < test_size; i++) {
        forward_propogate(n, test_inputs[i], output);
        if (find_max(expected_outputs[i]) == find_max(output))
            *num_correct = (*num_correct) + 1;
        *cost += calculate_cost(expected_outputs[i], output);
    }
    matrix_destroy(output);
}

void print_guess(void) {
    printf("Is this your number? %d\n", guess);
}

int main(void) {

    srand(1);

    // Load training images
    size_t num_images;
    Matrix** training_images;
    num_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/train-images-idx3-ubyte", &training_images);

    size_t num_labels;
    uint8_t* training_labels;
    num_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/train-labels-idx1-ubyte", &training_labels);

    // Split out a validation set
    size_t num_val_images = 10000;
    num_images = num_images - num_val_images;
    //Matrix** val_images = &training_images[num_images];
    //uint8_t* val_labels = &training_labels[num_images];

    // Generate matrix of expected outputs
    Matrix** expected_outputs = calloc(sizeof(Matrix*), num_labels);
    for (size_t i = 0; i < num_labels; i++) {
        expected_outputs[i] = matrix_create(10, 1);
        expected_outputs[i]->data[training_labels[i]] = 1.0;
    }

    // Load test set
    size_t num_test_images;
    Matrix** test_images;
    num_test_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/t10k-images-idx3-ubyte", &test_images);

    size_t num_test_labels;
    uint8_t* test_labels;
    num_test_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/t10k-labels-idx1-ubyte", &test_labels);

    // Generate matrix of expected outputs
    Matrix** expected_test_outputs = calloc(sizeof(Matrix*), num_test_labels);
    for (size_t i = 0; i < num_test_labels; i++) {
        expected_test_outputs[i] = matrix_create(10, 1);
        expected_test_outputs[i]->data[test_labels[i]] = 1.0;
    }

/*
   // Extend training set by manipulating training inputs
    Matrix** variable_images;
    uint8_t* variable_labels;
    size_t num_variable_images;
    printf("Extending MNIST Set...\n");
    num_variable_images = extend_set(training_images, training_labels, num_images, &variable_images, &variable_labels);
    
    // Copy new images to end of training images list
    
    training_images = realloc(training_images, sizeof(Matrix*) * num_images * 2);
    expected_outputs = realloc(expected_outputs, sizeof(Matrix*) * num_images * 2);
    for (size_t i = 0; i < num_images; i++) {
        training_images[num_images + i] = variable_images[i];
        expected_outputs[num_images + i] = expected_outputs[i];
    }
    num_images *= 2;
   */ 

    // Create neural network
    uint32_t nodes[] = {100, 100, 16, 10};
    NeuralNetwork n = create_neural_network(784, sizeof(nodes)/sizeof(nodes[0]), nodes);

    // Lets train our network
    int training_iterations = 30;
    int num_to_train = num_images;
    size_t success;
    float cost;
    evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
    printf("Starting Benchmark - Number Right: %zu Success Rate: %f Cost: %f\n", success, (float)success/(float)num_test_images, cost);
    for (int i = 0; i < training_iterations; i++) {
        clock_t begin = clock();
        stochastic_gradient_descent(n, num_to_train, training_images, expected_outputs, 2000, 3.0);
        evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
        printf("Training Round %d Number Right: %zu Success Rate: %f Cost: %f\n", i, success, (float)success/(float)num_test_images, cost);
        printf("Time elapsed: %f\n", (double)(clock() - begin) / (double) CLOCKS_PER_SEC);
    }

    // Save my neural network
    //save_neural_network(n, "test_neural_network_varied.txt");
    

    /*
    // Load neural network
    NeuralNetwork n;
    size_t success;
    float cost;
    load_neural_network("test_neural_network.txt", &n);
    evaluate_network(n, num_test_images, test_images, expected_test_outputs, &success, &cost);
    printf("Training Round %d Number Right: %zu Success Rate: %f Cost: %f\n", 0, success, (float)success/(float)num_test_images, cost);
    */
    init_screen();
    SDL_Color red = {255, 0, 0, 255};
    SDL_Color green = {0, 255, 0, 255};

    Matrix* input = get_input_matrix();
    Matrix* output = matrix_create(10, 1);

    add_button(200, 420, 100, 100, red, clear_inputs);
    add_button(200, 540, 100, 100, green, print_guess);
    while (handle_inputs()) {
        //draw_image(0, 0, training_images[0]);
        draw_input(400, 200);
        display_screen();
        clear_screen();

        // Guess input
        forward_propogate(n, input, output);
        guess = find_max(output);
    }
    
    destroy_screen();


}
