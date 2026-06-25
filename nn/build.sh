#!/bin/sh

set -xe


cc -Werror -Wextra -ggdb nn.c -o nn -lm

