#ifndef UI_H
#define UI_H

#include "SDL.h"
#include "matrix.h"
#include "neural_net.h"

#define PIXEL_SIZE (10)
#define WIDTH (1600)
#define HEIGHT (800)

#define IMG_ROWS (28)
#define IMG_COLS (28)


#define RED     ((SDL_Color) {255,   0,   0, 255})
#define GREEN   ((SDL_Color) {  0, 255,   0, 255})
#define BLUE    ((SDL_Color) {  0,   0, 255, 255})
#define WHITE   ((SDL_Color) {255, 255, 255, 255})
#define BLACK   ((SDL_Color) {  0,   0,   0, 255})

#define RECT(x, y, w, h) ((SDL_Rect){(x), (y), (w), (h)})
#define COLOR(r, g, b, a) ((SDL_Color){(r), (g), (b), (a)})

typedef struct Button {
    SDL_Rect r;
    SDL_Color c;
    void* state;
    void (*callback)(struct Button*);
} Button;

typedef struct Image {
    SDL_Rect r;
    SDL_Color c;
    Matrix* m;
    void* state;
    void (*callback)(int x, int y, struct Image*);
} Image;

typedef struct Screen {
    SDL_Window* window;
    SDL_Renderer *renderer;

    // States
    bool mouse_pressed;

    // Buttons
    int num_buttons;
    Button buttons[10]; // Max 10 buttons

    // Images
    int num_images;
    Image images[10];
} Screen;



void init_screen(void);
void clear_screen(void);
void display_screen(void);
void destroy_screen(void);
int handle_inputs(void);

void add_button(SDL_Rect r, SDL_Color c, void (*callback)(Button*), void* state);
void add_image(int x, int y, Matrix* m, void (*callback)(int, int, Image*), void* state);

void draw_fully_connected_network(SDL_Rect r, NeuralNetwork n);

void create_digit_classifier_window(uint8_t* guess, Matrix* image);

#endif // UI_H

