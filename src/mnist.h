#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

// Load dataset, return number of elements, store pointer
size_t load_images(const char* path, Matrix** images);
size_t load_labels(const char* path, uint8_t** labels);

void rotate_image(Matrix dst, Matrix img, float deg);
void rescale_image(Matrix dst, Matrix img, float mag);
void translate_image(Matrix dst, Matrix img, int dx, int dy);

size_t extend_set(Matrix* images, uint8_t* labels, size_t num_images, Matrix** new_images, uint8_t** new_labels);

#endif // MNIST_H
