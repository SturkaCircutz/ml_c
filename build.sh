#!/bin/sh

set -xe

cc -Wall -Wextra -ggdb -o twice twice.c -lm
cc -Wall -Wextra -ggdb -o gates gates.c -lm
cc -Wall -Wextra -ggdb -o xor xor.c -lm
