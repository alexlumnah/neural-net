#ifndef UI_H
#define UI_H

#include "SDL.h"
#include "matrix.h"


void init_screen(void);
void clear_screen(void);
void draw_pixel(int x, int y, int col, int row, SDL_Color c);
void draw_input(int x, int y);
void draw_image(int x, int y, Matrix* image);
void display_screen(void);
void destroy_screen(void);

void add_button(int x, int y, int w, int h, SDL_Color c, void (*callback)(void));
int handle_inputs(void);
Matrix* get_input_matrix(void);

void clear_inputs(void);


#endif // UI_H
