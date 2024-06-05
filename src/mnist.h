#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

// Load dataset, return number of elements, store pointer
uint32_t load_images(const char* path, Matrix*** images);
uint32_t load_labels(const char* path, uint8_t** labels);

void display_image(Matrix* image);

#endif // MNIST_H
