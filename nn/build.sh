#!/bin/sh

set -xe


cc -Werror -Wextra -ggdb nn.c -o nn -lm

cc -Werror -Wextra -ggdb adder.c -o adder -lm

