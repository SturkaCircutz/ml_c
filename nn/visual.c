#define NN_IMPLEMENT
#define VISUAL_IMPLEMENT
#define BITS 2

#include "nn.h"
#include "visual.h"

#include <stdlib.h>

Mat adder(void);
Mat output(Mat input);

Mat adder(void)
{
    size_t n = (1u << BITS);
    Mat input = mat_malloc(n * n, 2 * BITS);

    for (size_t i = 0; i < input.rows; ++i) {
        size_t x = i / n;
        size_t y = i % n;

        for (size_t j = 0; j < BITS; ++j) {
            CAL_MAT(input, i, j) = (float)((x >> j) & 1u);
            CAL_MAT(input, i, j + BITS) = (float)((y >> j) & 1u);
        }
    }

    return input;
}

Mat output(Mat input)
{
    size_t n = (1u << BITS);
    Mat output = mat_malloc(input.rows, BITS + 1);

    for (size_t i = 0; i < input.rows; ++i) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        size_t overflow = z >= n;

        for (size_t j = 0; j < BITS; ++j) {
            CAL_MAT(output, i, j) = (float)((z >> j) & 1u);
        }

        CAL_MAT(output, i, BITS) = (float)overflow;
    }

    return output;
}

int main(int argc, char **argv)
{
    int max_frames = 0;

    if (argc == 3 && argv[1][0] == '-' && argv[1][1] == '-' &&
        argv[1][2] == 'f') {
        max_frames = atoi(argv[2]);
    }

    srand(60);

    Mat input = adder();
    Mat out = output(input);

    size_t layers[] = {input.cols, 2 * BITS, out.cols};
    NN nn = NN_config(LIST_COUNT(layers), layers);
    NN gradient = NN_config(LIST_COUNT(layers), layers);
    VisualApp app = {0};

    nn_rand(nn, 0.0f, 1.0f);
    visual_app_init(&app, &nn, &gradient, input, out);

    int result = visual_run(&app, "NN visual", max_frames);


    return result;
}
