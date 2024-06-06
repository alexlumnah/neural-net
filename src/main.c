#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "matrix.h"
#include "neural_net.h"
#include "mnist.h"

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

int evaluate_network(NeuralNetwork n, int test_size, Matrix** test_inputs, Matrix** expected_outputs) {

    int count = 0;
    // Now evaluate every single input, test the output
    Matrix* output = matrix_create(expected_outputs[0]->rows, expected_outputs[0]->cols);
    for (int i = 0; i < test_size; i++) {
        forward_propogate(n, test_inputs[i], output);
        if (find_max(expected_outputs[i]) == find_max(output))
            count++;

        if (i < 5) {
            printf("%d: expected: %d actual: %d\n", i, find_max(expected_outputs[i]), find_max(output));
            //matrix_print(output);
        }
    }

    return count;

}

int main(void) {

    // Load training images
    uint32_t num_images;
    Matrix** training_images;
    num_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/train-images-idx3-ubyte", &training_images);

    uint32_t num_labels;
    uint8_t* training_labels;
    num_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/train-labels-idx1-ubyte", &training_labels);

    // Generate matrix of expected outputs
    Matrix** expected_outputs = calloc(sizeof(Matrix*), num_labels);
    for (uint32_t i = 0; i < num_labels; i++) {
        expected_outputs[i] = matrix_create(10, 1);
        expected_outputs[i]->data[training_labels[i]] = 1.0;
    }

    // Load test set
    uint32_t num_test_images;
    Matrix** test_images;
    num_test_images = load_images("/Users/lumnah/space/c/neural-net/mnist-data/t10k-images-idx3-ubyte", &test_images);

    uint32_t num_test_labels;
    uint8_t* test_labels;
    num_test_labels = load_labels("/Users/lumnah/space/c/neural-net/mnist-data/t10k-labels-idx1-ubyte", &test_labels);

    // Generate matrix of expected outputs
    Matrix** expected_test_outputs = calloc(sizeof(Matrix*), num_test_labels);
    for (uint32_t i = 0; i < num_test_labels; i++) {
        expected_test_outputs[i] = matrix_create(10, 1);
        expected_test_outputs[i]->data[test_labels[i]] = 1.0;
    }

    // Create neural network
    srand(time(NULL));
    uint32_t nodes[4] = {100, 32, 32, 10};
    NeuralNetwork n = create_neural_network(784, 4, nodes);

    // Lets do some predictions
    // Lets train our network
    int training_iterations = 30;
    int num_to_train = num_images;
    int success = evaluate_network(n, num_test_images, test_images, expected_test_outputs);
    printf("Starting Benchmark - Number Right: %d Success Rate: %f\n", success, (float)success/(float)num_test_images);
    for (int i = 0; i < training_iterations; i++) {
        clock_t begin = clock();
        stochastic_gradient_descent(n, num_to_train, training_images, expected_outputs, 2000, 3.0);
        success = evaluate_network(n, num_test_images, test_images, expected_test_outputs);
        printf("Training Round %d Number Right: %d Success Rate: %f\n", i, success, (float)success/(float)num_test_images);
        printf("Time elapsed: %f\n", (double)(clock() - begin) / (double) CLOCKS_PER_SEC);
    }
    //gradient_descent(n, 10, training_images, expected_outputs, 0.1/10.0);

}
