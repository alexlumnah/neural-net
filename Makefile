# Define the compiled
CC = gcc

# Compiler Flags:
CFLAGS = -g -Wall -Wpedantic -Wextra -fsanitize=address,undefined,signed-integer-overflow 

# SDL2 flags
CFLAGS += -I/opt/homebrew/Cellar/sdl2/2.28.5/include/SDL2
LDFLAGS = -L/opt/homebrew/Cellar/sdl2/2.28.5/lib/ -lSDL2

# openblas flags for fast matrix mulitplication
CFLAGS += -I/opt/homebrew/opt/openblas/include
LDFLAGS += -L/opt/homebrew/opt/openblas/lib/ -lopenblas

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)

TEST_SRC = $(wildcard test/*.c)
TEST_OBJ = $(TEST_SRC:.c=.o)

all: main

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o main $^ $(CFLAGS) $(LDFLAGS)

test: test/test.o src/matrix.o
	$(CC) -o test $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm main test/test.o $(OBJ)

tidy:
	clang-tidy src/* --

cppcheck:
	cppcheck --enable=portability --check-level=exhaustive --enable=style src/*.c
