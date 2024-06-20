#include <stdbool.h>
#include <stdio.h>

#include "SDL.h"
#include "ui.h"
#include "neural_net.h"

Screen screen;

void draw_neuron(int x, int y, int rad) {
    for (int i = x - rad; i <= x + rad; i++) {
        for (int j = y - rad; j <= y + rad; j++) {
            if ((i-x)*(i-x) + (j-y)*(j-y) < rad*rad) {
                SDL_RenderDrawPoint(screen.renderer, i, j);
            }
        }
    }
}

void draw_neural_net(SDL_Rect r, NeuralNetwork n) {

    // Preprocess neural network for plotting purposes
    float max_weight = 0;
    float min_weight = 0;
    float max_bias = 0;
    float min_bias = 0;
    for (uint32_t i = 1; i <= n.num_layers; i++) {
        Layer l = n.layers[i];
        for (uint32_t j = 0; j < l.w->rows * l.w->cols; j++) {
            if (l.w->data[j] > max_weight) max_weight = l.w->data[j];
            if (l.w->data[j] < min_weight) min_weight = l.w->data[j];
        }
        for (uint32_t j = 0; j < l.b->rows * l.b->cols; j++) {
            if (l.b->data[j] > max_bias) max_bias = l.b->data[j];
            if (l.b->data[j] < min_bias) min_bias = l.b->data[j];
        }
    }

    // Draw neurons
    int n_rad = 15;
    int l_space = (r.w - 2 * n_rad) / (n.num_layers - 1);
    for (uint32_t i = 1; i < n.num_layers; i++) {
        int n0_space = r.h / (n.layers[i].num_nodes + 1);
        for (uint32_t j = 0; j < n.layers[i].num_nodes; j++) {
            int x0 = r.x + n_rad;
            int y0 = r.y + n_rad + n0_space/2;
            for (uint32_t k = 0; k < n.layers[i+1].num_nodes; k++) {
                int n1_space = r.h / (n.layers[i+1].num_nodes + 1);
                int x1 = r.x + n_rad;
                int y1 = r.y + n_rad + n1_space/2;
                float w = n.layers[i].w->data[j * n.layers[i].w->cols + k];
                uint32_t red = w < 0 ? (uint8_t)255*(w/min_weight) : 0;
                uint32_t blue = w > 0 ? (uint8_t)255*(w/max_weight) : 0;
                SDL_SetRenderDrawColor(screen.renderer, red, 0, blue, 255);
                SDL_RenderDrawLine( screen.renderer,
                                    x0 + l_space * (i - 1),
                                    y0 + n0_space * j,
                                    x1 + l_space * (i),
                                    y1 + n1_space * k);
            }
            float w = n.layers[i].b->data[j];
            uint32_t red = w < 0 ? (uint8_t)255*(w/min_bias) : 0;
            uint32_t blue = w > 0 ? (uint8_t)255*(w/max_bias) : 0;
            SDL_SetRenderDrawColor(screen.renderer, red, 0, blue, 255);
            draw_neuron(r.x + l_space * (i - 1) + n_rad,
                        r.y + n0_space/2 + n0_space * j + n_rad,
                        n_rad);
        }
    }

    // Draw final layer of neurons
    for (uint32_t j = 0; j < n.layers[n.num_layers].num_nodes; j++) {
        int n_space = r.h / (n.layers[n.num_layers].num_nodes + 1);
        int x0 = r.x + n_rad;
        int y0 = r.y + n_rad + n_space/2;
        float w = n.layers[n.num_layers].b->data[j];
        uint32_t red = w < 0 ? (uint8_t)255*(w/min_bias) : 0;
        uint32_t blue = w > 0 ? (uint8_t)255*(w/max_bias) : 0;
        SDL_SetRenderDrawColor(screen.renderer, red, 0, blue, 255);
        draw_neuron(x0 + l_space * (n.num_layers - 1),
                    y0 + n_space * j,
                    n_rad);
    }

    SDL_RenderDrawRect(screen.renderer, &r);
        
}

void draw_cost_plot(SDL_Rect r, Matrix* cost) {
    (void) r;
    (void) cost;
    printf("draw_cost_plot -> not implemented :(\n");
}

void add_button(SDL_Rect r, SDL_Color c, void (*callback)(Button*), void* state) {

    screen.buttons[screen.num_buttons].r = r;
    screen.buttons[screen.num_buttons].c = c;
    screen.buttons[screen.num_buttons].state = state;
    screen.buttons[screen.num_buttons].callback = callback;
    screen.num_buttons += 1;

}

void draw_button(Button b) {
        SDL_SetRenderDrawColor(screen.renderer,b.c.r, b.c.g, b.c.b, b.c.a);
        SDL_RenderFillRect(screen.renderer, &b.r);
}

void add_image(int x, int y, Matrix* m, void (*callback)(int, int, Image*), void* state) {
    screen.images[screen.num_images].r = RECT(x, y, IMG_ROWS * PIXEL_SIZE, IMG_COLS * PIXEL_SIZE);
    screen.images[screen.num_images].m = m;
    screen.images[screen.num_images].state = state;
    screen.images[screen.num_images].callback = callback;
    screen.num_images += 1;

}

void draw_image(Image img) {

    // Now draw diagram
    for (uint32_t i = 0; i < IMG_ROWS; i++) {
        for (uint32_t j = 0; j < IMG_COLS; j++) {

            uint8_t val = (uint8_t)(img.m->data[i * IMG_COLS + j] * 255);

            SDL_Rect rect = {img.r.x + PIXEL_SIZE * j,
                             img.r.y + PIXEL_SIZE * i,
                             PIXEL_SIZE,
                             PIXEL_SIZE};
            SDL_SetRenderDrawColor(screen.renderer,val,val,val,255);
            SDL_RenderFillRect(screen.renderer, &rect);
        }
    }

    SDL_Rect rect = {img.r.x, img.r.y, IMG_COLS * PIXEL_SIZE, IMG_ROWS * PIXEL_SIZE};
    SDL_SetRenderDrawColor(screen.renderer,255, 255, 255, 255);
    SDL_RenderDrawRect(screen.renderer, &rect);
}

void mouse_click(int x, int y) {

    // Check if any buttons are pressed
    for (int i = 0; i < screen.num_buttons; i++) {
        Button but = screen.buttons[i];
        if (x >= but.r.x && x < but.r.x + but.r.w &&
            y >= but.r.y && y < but.r.y + but.r.h &&
            but.callback != NULL)
            but.callback(&but);
    }

    // Check if any images are clicked
    for (int i = 0; i < screen.num_images; i++) {
        Image img = screen.images[i];
        if (x >= img.r.x && x < img.r.x + img.r.w &&
            y >= img.r.y && y < img.r.y + img.r.h &&
            img.callback != NULL)
            screen.images[i].callback(x, y, &screen.images[i]);
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

    // Instantiate button
    screen.num_buttons = 0;
    screen.num_images = 0;
}

void clear_screen(void) {
    SDL_SetRenderDrawColor(screen.renderer,0,0,0,255);
    SDL_RenderClear(screen.renderer);
}

void display_screen(void) {

    // Draw buttons
    for (int i = 0; i < screen.num_buttons; i++) {
        draw_button(screen.buttons[i]);
    }

    // Draw images
    for (int i = 0; i < screen.num_images; i++) {
        draw_image(screen.images[i]);
    }

    // Render screen
    SDL_RenderPresent(screen.renderer);    
    
}

void destroy_screen(void) {
    SDL_DestroyWindow(screen.window);
}
