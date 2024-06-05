#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"


uint32_t load_images(const char* path, Matrix*** images) {

    FILE* f = fopen(path, "rb");

    if (f == NULL) {
        return 0;
    }

    // Read header values
    int32_t magic_number;
    fread(&magic_number, sizeof(magic_number), 1, f);
    magic_number = htonl(magic_number);

    int32_t num_images;
    fread(&num_images, sizeof(num_images), 1, f);
    num_images = htonl(num_images);

    int32_t num_rows;
    fread(&num_rows, sizeof(num_rows), 1, f);
    num_rows = htonl(num_rows);

    int32_t num_cols;
    fread(&num_cols, sizeof(num_cols), 1, f);
    num_cols = htonl(num_cols);

    // Allocate an array of images
    *images = calloc(num_images, sizeof(Matrix*));

    // Read each image in the data set
    for (int n = 0; n < num_images; n++) {
        (*images)[n] = matrix_create(num_rows * num_cols, 1); // Create a column vector of inputs
        (*images)[n]->rows = num_rows * num_cols;
        (*images)[n]->cols = 1;
        for (int j = 0; j < num_rows; j++) {
            for (int i = 0; i < num_cols; i++) {
                uint8_t pixel;
                fread(&pixel, sizeof(pixel), 1, f);
                (*images)[n]->data[j * num_cols + i] = (float)pixel / 255.0;
            }
        }
    }

    fclose(f);

    return num_images;
}


uint32_t load_labels(const char* path, uint8_t** labels) {
    
    FILE* f = fopen(path, "rb");

    if (f == NULL) {
        return 0;
    }

    // Read header values
    int32_t magic_number;
    fread(&magic_number, sizeof(magic_number), 1, f);
    magic_number = htonl(magic_number);

    int32_t num_labels;
    fread(&num_labels, sizeof(num_labels), 1, f);
    num_labels = htonl(num_labels);

    // Allocate an array of images
    *labels = calloc(num_labels, sizeof(int));

    // Read each image in the data set
    for (int n = 0; n < num_labels; n++) {
        uint8_t label;
        fread(&label, sizeof(label), 1, f);
        (*labels)[n] = (uint8_t)label;

    }

    fclose(f);

    return num_labels;
}
