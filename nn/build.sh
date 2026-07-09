#!/bin/sh

set -xe


cc -Werror -Wextra -ggdb nn.c -o nn -lm
cc -Werror -Wextra -ggdb adder.c -o adder -lm
cc -Werror -Wextra -ggdb $(sdl2-config --cflags) visual.c -o visual -lm $(sdl2-config --libs)
cc -Werror -Wextra -ggdb $(sdl2-config --cflags) more.c -o more -lm $(sdl2-config --libs)
