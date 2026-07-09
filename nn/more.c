#define NN_IMPLEMENT
#define VISUAL_IMPLEMENT

#include "nn.h"
#include "visual.h"

#include <stdlib.h>

#define MORE_INPUTS 5
#define MORE_HIDDEN 10
#define MORE_OUTPUTS 3
#define MORE_SAMPLES 32

Mat more_input(void);
Mat more_output(Mat input);

Mat more_input(void)
{
    Mat input = mat_malloc(MORE_SAMPLES, MORE_INPUTS);

    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t col = 0; col < input.cols; ++col) {
            CAL_MAT(input, row, col) = (float)((row >> col) & 1u);
        }
    }

    return input;
}

Mat more_output(Mat input)
{
    Mat output = mat_malloc(input.rows, MORE_OUTPUTS);

    for (size_t row = 0; row < input.rows; ++row) {
        float a = CAL_MAT(input, row, 0);
        float b = CAL_MAT(input, row, 1);
        float c = CAL_MAT(input, row, 2);
        float d = CAL_MAT(input, row, 3);
        float e = CAL_MAT(input, row, 4);

        CAL_MAT(output, row, 0) = (a != b) ? 1.0f : 0.0f;
        CAL_MAT(output, row, 1) = (c != d) ? 1.0f : 0.0f;
        CAL_MAT(output, row, 2) = (a + b + c + d + e >= 3.0f) ? 1.0f : 0.0f;
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

    Mat input = more_input();
    Mat out = more_output(input);
    size_t layers[] = {MORE_INPUTS, MORE_HIDDEN, MORE_OUTPUTS};
    NN nn = NN_config(LIST_COUNT(layers), layers);
    NN gradient = NN_config(LIST_COUNT(layers), layers);
    VisualApp app = {0};

    nn_rand(nn, 0.0f, 1.0f);
    visual_app_init(&app, &nn, &gradient, input, out);

    return visual_run(&app, "5-10-3 NN visual", max_frames);
}
