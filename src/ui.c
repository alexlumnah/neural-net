#include <stdbool.h>

#include "SDL.h"
#include "ui.h"

#define PIXEL_SIZE (10)
#define WIDTH (28*PIXEL_SIZE)
#define HEIGHT (28*PIXEL_SIZE)

void draw_pixel(SDL_Renderer* renderer, int row, int col, uint8_t val) {
    SDL_Rect rect = {PIXEL_SIZE * col, PIXEL_SIZE * row, PIXEL_SIZE, PIXEL_SIZE};
    SDL_SetRenderDrawColor(renderer,val,val,val,255);
    SDL_RenderFillRect(renderer, &rect);
}

void display_image(Matrix* image, uint32_t rows, uint32_t cols) {

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
    SDL_Window* window = SDL_CreateWindow(
        "My Window",
        SDL_WINDOWPOS_CENTERED_DISPLAY(0),
        SDL_WINDOWPOS_CENTERED_DISPLAY(0),
        WIDTH,
        HEIGHT,
        window_flags);
    if (!window) {
        printf("Failed to open %d x %d window: %s\n", WIDTH, HEIGHT, SDL_GetError());
        exit(1);
    }

    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, renderer_flags);
    if (!renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        exit(1);
    };

    // Set blend mode to support alpha values
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // Clear screen
    SDL_SetRenderDrawColor(renderer,0,0,0,255);
    SDL_RenderClear(renderer);

    // Now draw diagram
    for (uint32_t j = 0; j < rows; j++) {
        for (uint32_t i = 0; i < cols; i++) {
            draw_pixel(renderer, j, i, (uint8_t)(image->data[j * cols + i]*255));
        }
    }

    SDL_RenderPresent(renderer);

    SDL_Event e;
    bool quit = false;
    while( quit == false ){
        while( SDL_PollEvent( &e ) ){
            if( e.type == SDL_QUIT )
                quit = true;
        }
    }
    SDL_DestroyWindow(window);
}
