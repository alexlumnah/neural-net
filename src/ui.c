#include <stdbool.h>

#include "SDL.h"
#include "ui.h"

#define PIXEL_SIZE (10)
#define WIDTH (1600)
#define HEIGHT (800)

#define IMG_ROWS (28)
#define IMG_COLS (28)

typedef struct Button {
    SDL_Rect b;
    SDL_Color c;
    void (*callback)(void);
} Button;

typedef struct Screen {
    SDL_Window* window;
    SDL_Renderer *renderer;

    // Area for drawing input
    int in_x, in_y;
    Matrix* input;

    // States
    bool mouse_pressed;

    // Buttons
    int num_buttons;
    Button buttons[10]; // Max 10 buttons

} Screen;

Screen screen;

Matrix* get_input_matrix(void) {
    return screen.input;
}

void init_screen(void) {
    // Flags for window and renderer
    int window_flags = 0;
    int renderer_flags = SDL_RENDERER_PRESENTVSYNC;

    // Init SDL
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        printf("Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }

    // Create our window
    screen.window = SDL_CreateWindow(
        "My Window",
        SDL_WINDOWPOS_CENTERED_DISPLAY(0),
        SDL_WINDOWPOS_CENTERED_DISPLAY(0),
        WIDTH,
        HEIGHT,
        window_flags);
    if (!screen.window) {
        printf("Failed to open %d x %d window: %s\n", WIDTH, HEIGHT, SDL_GetError());
        exit(1);
    }

    // Create renderer
    screen.renderer = SDL_CreateRenderer(screen.window, -1, renderer_flags);
    if (!screen.renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        exit(1);
    };

    // Set blend mode to support alpha values
    SDL_SetRenderDrawBlendMode(screen.renderer, SDL_BLENDMODE_BLEND);

    // Clear screen
    SDL_SetRenderDrawColor(screen.renderer,0,0,0,255);
    SDL_RenderClear(screen.renderer);

    // Create a region of screen for input
    screen.input = matrix_create(IMG_ROWS * IMG_COLS, 1);

    // Instantiate button
    screen.num_buttons = 0;
}

void clear_screen(void) {
    SDL_SetRenderDrawColor(screen.renderer,0,0,0,255);
    SDL_RenderClear(screen.renderer);
}

// Draw pixel in image with origin at x, y
void draw_pixel(int x, int y, int col, int row, SDL_Color c) {
    SDL_Rect rect = {x + PIXEL_SIZE * col, y + PIXEL_SIZE * row, PIXEL_SIZE, PIXEL_SIZE};
    SDL_SetRenderDrawColor(screen.renderer,c.r,c.g,c.b,c.a);
    SDL_RenderFillRect(screen.renderer, &rect);
}

void draw_image(int x, int y, Matrix* image) {

    // Now draw diagram
    for (uint32_t i = 0; i < IMG_ROWS; i++) {
        for (uint32_t j = 0; j < IMG_COLS; j++) {

            uint8_t val = (uint8_t)(image->data[i * IMG_COLS + j] * 255);
            SDL_Color c = {.r = val, .g = val, .b = val, .a = 255};

            draw_pixel(x, y, j, i, c);
        }
    }
}

void draw_input(int x, int y) {
    screen.in_x = x;
    screen.in_y = y;
    draw_image(x, y, screen.input);

    // Draw a bounding box
    SDL_Rect rect = {x, y, IMG_COLS * PIXEL_SIZE, IMG_ROWS * PIXEL_SIZE};
    SDL_SetRenderDrawColor(screen.renderer,255, 255, 255, 255);
    SDL_RenderDrawRect(screen.renderer, &rect);
}

void display_screen(void) {

    // Draw buttons
    for (int i = 0; i < screen.num_buttons; i++) {
        SDL_SetRenderDrawColor(screen.renderer,screen.buttons[i].c.r, screen.buttons[i].c.g, screen.buttons[i].c.b, screen.buttons[i].c.a);
        SDL_RenderFillRect(screen.renderer, &screen.buttons[i].b);
    }
    // Render screen
    SDL_RenderPresent(screen.renderer);    
    
}

void add_button(int x, int y, int w, int h, SDL_Color c, void (*callback)(void)) {

    screen.buttons[screen.num_buttons].b = (SDL_Rect){.x = x, .y = y, .w = w, .h = h};
    screen.buttons[screen.num_buttons].c = c;
    screen.buttons[screen.num_buttons].callback = callback;
    screen.num_buttons += 1;

}

void mouse_click(int x, int y) {
    if (x > screen.in_x && x < screen.in_x + IMG_COLS * PIXEL_SIZE &&
        y > screen.in_y && y < screen.in_y + IMG_ROWS * PIXEL_SIZE) {
        uint32_t in_col = (x - screen.in_x - PIXEL_SIZE/2) / PIXEL_SIZE;
        uint32_t in_row = (y - screen.in_y - PIXEL_SIZE/2) / PIXEL_SIZE;
        float col_f = (float)((x - screen.in_x - PIXEL_SIZE/2) % PIXEL_SIZE) / (float)PIXEL_SIZE;
        float row_f = (float)((y - screen.in_y - PIXEL_SIZE/2) % PIXEL_SIZE) / (float)PIXEL_SIZE;

        col_f = col_f < 0 ? 0 : col_f;
        row_f = row_f < 0 ? 0 : row_f;

        screen.input->data[in_row * IMG_COLS + in_col] += 1.0 * (1 - col_f) * (1 - row_f);
        screen.input->data[in_row * IMG_COLS + in_col] = screen.input->data[in_row * IMG_COLS + in_col] > 1 ? 1 : screen.input->data[in_row * IMG_COLS + in_col];

        if (in_col + 1 < IMG_COLS){
            screen.input->data[in_row * IMG_COLS + in_col + 1] += 1.0 * col_f * (1 - row_f);
            screen.input->data[in_row * IMG_COLS + in_col + 1] = screen.input->data[in_row * IMG_COLS + in_col + 1] > 1 ? 1 : screen.input->data[in_row * IMG_COLS + in_col + 1];
        }
        if (in_row + 1 < IMG_ROWS) {
            screen.input->data[(in_row + 1) * IMG_COLS + in_col] += 1.0 * (1 - col_f) * (1 - row_f);
            screen.input->data[(in_row + 1) * IMG_COLS + in_col] = screen.input->data[(in_row + 1) * IMG_COLS + in_col] > 1 ? 1 : screen.input->data[(in_row + 1) * IMG_COLS + in_col];
        }
        if (in_row + 1 < IMG_ROWS && in_col + 1 < IMG_COLS) {
            screen.input->data[(in_row + 1) * IMG_COLS + in_col + 1] += 1.0 * col_f * row_f;
            screen.input->data[(in_row + 1) * IMG_COLS + in_col + 1] = screen.input->data[(in_row + 1) * IMG_COLS + in_col + 1] > 1 ? 1 : screen.input->data[(in_row + 1) * IMG_COLS + in_col + 1];
        }

    }

    // Check if any buttons are pressed
    for (int i = 0; i < screen.num_buttons; i++) {
        if (x >= screen.buttons[i].b.x && x < screen.buttons[i].b.x + screen.buttons[i].b.w &&
            y >= screen.buttons[i].b.y && y < screen.buttons[i].b.y + screen.buttons[i].b.h)
            screen.buttons[i].callback();
    }
}

void clear_inputs(void) {
    for (int i = 0; i < screen.input->rows * screen.input->cols; i++) {
        screen.input->data[i] = 0;
    }
}

int handle_inputs(void) {

    SDL_Event e;
    while( SDL_PollEvent( &e ) ){
        switch (e.type) {
        case SDL_QUIT:
            SDL_DestroyWindow(screen.window);
            return 0;
        case SDL_MOUSEBUTTONDOWN:
            screen.mouse_pressed = true;
            mouse_click(e.button.x, e.button.y);
            break;
        case SDL_MOUSEBUTTONUP:
            screen.mouse_pressed = false;
            break;
        case SDL_MOUSEMOTION: {
            if (screen.mouse_pressed) {
                mouse_click(e.motion.x, e.button.y);
            }
        }
        }
    }

    return 1;
}

void destroy_screen(void) {
    SDL_DestroyWindow(screen.window);
}
