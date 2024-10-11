#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mnist.h"
#include "matrix.h"

#define ROWS (28)
#define COLS (28)

size_t load_images(const char* path, Matrix** images) {

    FILE* f = fopen(path, "rb");

    if (f == NULL) return -1;

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
    *images = calloc(num_images, sizeof(Matrix));

    // Read each image in the data set
    for (int n = 0; n < num_images; n++) {
        (*images)[n] = matrix_create(num_rows, num_cols);
        for (int j = 0; j < num_rows; j++) {
            for (int i = 0; i < num_cols; i++) {
                uint8_t pixel;
                fread(&pixel, sizeof(pixel), 1, f);
                (*images)[n].data[j * num_cols + i] = (float)pixel / 255.0;
            }
        }
    }

    fclose(f);

    return (size_t)num_images;
}

size_t load_labels(const char* path, uint8_t** labels) {
    
    FILE* f = fopen(path, "rb");

    if (f == NULL) return -1;
    

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

    return (size_t)num_labels;
}

// Rotate an mnist digit
void rotate_image(Matrix dst, Matrix img, float deg) {

    // Basic algorithm - loop through each value in destination table
    // Apply a reverse rotation to find the value in the original image to sample
    // Average over all pixels

    // Temporary 2d array for storing results of rotation
    // This allows us to set the source and destination images the same
    float r[ROWS * COLS] = {0};

    float c = cosf(-deg * M_PI / 180.0f);
    float s = sinf(-deg * M_PI / 180.0f);

    for (uint32_t i = 0; i < ROWS; i++) {
        for (uint32_t j = 0; j < COLS; j++) {

            // First translate so origin is center, then rotate, then translate back
            int y = i - (ROWS / 2);
            int x = j - (COLS / 2);

            float i_rot = (float)x * s + (float)y * c + (ROWS / 2);
            float j_rot = (float)x * c - (float)y * s + (COLS / 2);
            
            int i_f = (int)floor(i_rot);
            int i_c = (int)ceil(i_rot);
            float i_p = i_rot - floor(i_rot);

            int j_f = (int)floor(j_rot);
            int j_c = (int)ceil(j_rot);
            float j_p = j_rot - floor(j_rot);

            // Average all values into one
            dst.data[i * COLS + j] = 0;
            if (i_f >= 0 && i_f < ROWS && j_f >= 0 && j_f < COLS)
                r[i * COLS + j] += (1.0 - i_p) * (1.0 - j_p)* img.data[i_f * COLS + j_f];
            if (i_f >= 0 && i_f < ROWS && j_c >= 0 && j_c < COLS)
                r[i * COLS + j] += (1.0 - i_p) * j_p * img.data[i_f * COLS + j_c];
            if (i_c >= 0 && i_c < ROWS && j_f >= 0 && j_f < COLS)
                r[i * COLS + j] += i_p * (1.0 - j_p) * img.data[i_c * COLS + j_f];
            if (i_c >= 0 && i_c < ROWS && j_c >= 0 && j_c < COLS)
                r[i * COLS + j] +=  i_p * j_p * img.data[i_c * COLS + j_c];
        }
    }

    // Now copy the results to the destination
    for (uint32_t n = 0; n < ROWS * COLS; n++) {
        dst.data[n] = r[n];
    }
}

void rescale_image(Matrix dst, Matrix img, float mag) {

    // Basic algorithm - for each value in destination table
    // Translate so origin is in center, then scale the coordinates based 
    // on the magnification. 

    // Temporary 2d array for storing results of rescaling
    // This allows us to set the source and destination images the same
    float s[ROWS * COLS] = {0};

    for (uint32_t i = 0; i < ROWS; i++) {
        for (uint32_t j = 0; j < COLS; j++) {

            // Determine scaled value of i and j
            float i_sc = ((float)i - (ROWS/2)) / mag + (ROWS/2);
            float j_sc = ((float)j - (COLS/2)) / mag + (COLS/2);

            int i_f = (int)floor(i_sc);
            int i_c = (int)ceil(i_sc);
            float i_p = i_sc - floor(i_sc);

            int j_f = (int)floor(j_sc);
            int j_c = (int)ceil(j_sc);
            float j_p = j_sc - floor(j_sc);

            // Average all values into one
            dst.data[i * COLS + j] = 0;
            if (i_f >= 0 && i_f < ROWS && j_f >= 0 && j_f < COLS)
                s[i * COLS + j] += (1.0 - i_p) * (1.0 - j_p)* img.data[i_f * COLS + j_f];
            if (i_f >= 0 && i_f < ROWS && j_c >= 0 && j_c < COLS)
                s[i * COLS + j] += (1.0 - i_p) * j_p * img.data[i_f * COLS + j_c];
            if (i_c >= 0 && i_c < ROWS && j_f >= 0 && j_f < COLS)
                s[i * COLS + j] += i_p * (1.0 - j_p) * img.data[i_c * COLS + j_f];
            if (i_c >= 0 && i_c < ROWS && j_c >= 0 && j_c < COLS)
                s[i * COLS + j] +=  i_p * j_p * img.data[i_c * COLS + j_c];
        }
    }

    // Now copy the results to the destination
    for (uint32_t n = 0; n < ROWS * COLS; n++) {
        dst.data[n] = s[n];
    }
}


void translate_image(Matrix dst, Matrix img, int dx, int dy) {

    // First find the min and max row and cols that have data
    // So we can clamp the translation
    uint32_t min_row = ROWS;
    uint32_t min_col = COLS;
    uint32_t max_row = 0;
    uint32_t max_col = 0;
    for (uint32_t i = 0; i < ROWS; i++) {
        for (uint32_t j = 0; j < COLS; j++) {

            if (fabs(img.data[i * COLS + j]) < 0.0001f)
                continue;
            
            if (i < min_row)
                min_row = i;
            if (i > max_row)
                max_row = i;
            if (j < min_col)
                min_col = j;
            if (j > max_col)
                max_col = j;
        }
    }

    // Clamp values such that we don't shift our digit off screen
    if (dy < 0 && min_row + dy < 0) {
        dy = -1 * min_row;
    } else if (dy > 0 && max_row + dy >= ROWS) {
        dy = ROWS - max_row - 1;
    }

    if (dx < 0 && min_col + dx < 0) {
        dx = -1 * min_col;
    } else if (dx > 0 && max_col + dx >= COLS) {
        dx = COLS - max_col - 1;
    }

    // Temporary 2d array for storing results of translation
    // This allows us to set the source and destination images the same
    float t[ROWS * COLS] = {0};

    // Now translate source to destination
    for (uint32_t i = 0; i < ROWS; i++) {
        for (uint32_t j = 0; j < COLS; j++) {
            if ( i - dy >= 0 && i - dy < ROWS && j - dx >= 0 && j - dx < COLS)
                t[i * COLS + j] = img.data[(i - dy) * COLS + j - dx];
        }
    }

    // Now copy the results to the destination
    for (uint32_t n = 0; n < ROWS * COLS; n++) {
        dst.data[n] = t[n];
    }
}

size_t extend_set(Matrix* images, uint8_t* labels, size_t num_images, Matrix** new_images, uint8_t** new_labels) {

    // Create list to store new images in
    *new_images = calloc(sizeof(Matrix), num_images);
    *new_labels = calloc(sizeof(uint8_t*), num_images);

    // Loop through all images and create a new image by 
    // rotating, scaling, and translating the original
    for (size_t i = 0; i < num_images; i++) {
        (*new_images)[i] = matrix_create(images[i].rows, images[i].cols);
        (*new_labels)[i] = labels[i];

        // Now rotate, scale, and translate image

        // Random rotation between -20 and 20 deg
        float rot_deg = -10.0f + 20.0f*((float)(rand()) / (float)(RAND_MAX));

        // Magnification between .5 and 1.5
        float mag = 0.5f + ((float)(rand()) / (float)(RAND_MAX));

        // Translation between -5 and 5
        int dx = -5 + (rand() % 11);
        int dy = -5 + (rand() % 11);

        rotate_image((*new_images)[i], images[i], rot_deg);
        rescale_image((*new_images)[i], images[i], mag);
        translate_image((*new_images)[i], images[i], dx, dy);

    }

    return num_images;
}

