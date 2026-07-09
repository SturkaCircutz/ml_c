#ifndef VISUAL_H
#define VISUAL_H

#include "nn.h"

#include <SDL2/SDL.h>
#include <stdbool.h>
#include <stddef.h>

#ifndef VISUAL_WINDOW_WIDTH
#define VISUAL_WINDOW_WIDTH 3000
#endif  // visual width

#ifndef VISUAL_WINDOW_HEIGHT
#define VISUAL_WINDOW_HEIGHT 2200
#endif  // visual height

#ifndef VISUAL_NODE_RADIUS
#define VISUAL_NODE_RADIUS 40
#endif  // visual node radius

#ifndef VISUAL_HISTORY_COUNT
#define VISUAL_HISTORY_COUNT 1000
#endif  // visual history count

#define VISUAL_GRAPH_MARGIN 70
#define VISUAL_GRAPH_WIDTH 720
#define VISUAL_GRAPH_HEIGHT 420
#define VISUAL_GRAPH_GAP 120

typedef struct {
    float x;
    float y;
} VisualVec2;

typedef struct {
    NN *nn;
    NN *gradient;
    Mat input;
    Mat output;

    size_t selected_sample;
    size_t train_step;
    size_t train_per_frame;

    float rate;
    float cost;
    float cost_history[VISUAL_HISTORY_COUNT];
    size_t cost_history_count;

    bool paused;
    bool running;
} VisualApp;

float visual_clampf(float value, float low, float high);
Uint8 visual_color_channel(float value);
bool visual_has_output(VisualApp *app);
bool visual_can_train(VisualApp *app);
void visual_app_init(VisualApp *app, NN *nn, NN *gradient, Mat input, Mat output);
void visual_record_cost(VisualApp *app);
void visual_train_once(VisualApp *app);
void visual_update_training(VisualApp *app);
void visual_update_forward_sample(VisualApp *app);
void visual_next_sample(VisualApp *app);
void visual_prev_sample(VisualApp *app);
void visual_handle_events(VisualApp *app);
VisualVec2 visual_neuron_pos(NN nn, size_t layer, size_t neuron);
void visual_draw_filled_circle(SDL_Renderer *renderer, int cx, int cy, int radius);
void visual_draw_thick_line(SDL_Renderer *renderer,
                            VisualVec2 from,
                            VisualVec2 to,
                            int width);
void visual_draw_connections(SDL_Renderer *renderer, NN nn);
void visual_draw_neurons(SDL_Renderer *renderer, NN nn);
void visual_draw_cost_history(SDL_Renderer *renderer, VisualApp *app);
void visual_draw_network(SDL_Renderer *renderer, VisualApp *app);
void visual_update_window_title(SDL_Window *window, VisualApp *app);
int visual_run(VisualApp *app, const char *title, int max_frames);

#ifdef VISUAL_IMPLEMENT

#include <math.h>
#include <stdio.h>

// make sure the value is bounded by low and high
float visual_clampf(float value, float low, float high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

Uint8 visual_color_channel(float value)
{
    return (Uint8)visual_clampf(value, 0.0f, 255.0f);
}

bool visual_has_output(VisualApp *app)
{
    return app->output.es != NULL && app->output.rows > 0 && app->output.cols > 0;
}

bool visual_can_train(VisualApp *app)
{
    return app->gradient != NULL && visual_has_output(app);
}

void visual_app_init(VisualApp *app, NN *nn, NN *gradient, Mat input, Mat output)
{
    ASSERT(app != NULL);
    ASSERT(nn != NULL);
    ASSERT(input.es != NULL);
    ASSERT(input.rows > 0);
    ASSERT(input.cols == nn->as[0].cols);

    if (output.es != NULL) {
        ASSERT(output.rows == input.rows);
        ASSERT(output.cols == nn->as[nn->count].cols);
    }

    app->nn = nn;
    app->gradient = gradient;
    app->input = input;
    app->output = output;
    app->selected_sample = 0;
    app->train_step = 0;
    app->train_per_frame = 4;
    app->rate = 1.0f;
    app->cost = 0.0f;
    app->cost_history_count = 0;
    app->paused = !visual_can_train(app);
    app->running = true;

    if (visual_has_output(app)) {
        app->cost = nn_cost(*app->nn, app->input, app->output);
        visual_record_cost(app);
    }

    visual_update_forward_sample(app);
}

void visual_record_cost(VisualApp *app)
{
    if (!visual_has_output(app)) {
        return;
    }

    if (app->cost_history_count < VISUAL_HISTORY_COUNT) {
        app->cost_history[app->cost_history_count++] = app->cost;
        return;
    }

    for (size_t i = 1; i < VISUAL_HISTORY_COUNT; ++i) {
        app->cost_history[i - 1] = app->cost_history[i];
    }
    app->cost_history[VISUAL_HISTORY_COUNT - 1] = app->cost;
}


// train the model using backprop
void visual_train_once(VisualApp *app)
{
    if (!visual_can_train(app)) {
        return;
    }

    nn_backprop(*app->nn, *app->gradient, app->input, app->output);
    nn_learn(*app->nn, *app->gradient, app->rate);
    app->train_step++;
}

void visual_update_training(VisualApp *app)
{
    if (!visual_can_train(app)) {
        return;
    }

    for (size_t i = 0; i < app->train_per_frame; ++i) {
        visual_train_once(app);
    }

    app->cost = nn_cost(*app->nn, app->input, app->output);
    visual_record_cost(app);
}

void visual_update_forward_sample(VisualApp *app)
{
    app->selected_sample %= app->input.rows;
    mat_copy(app->nn->as[0], mat_row(app->input, app->selected_sample));
    nn_forward(*app->nn);
}


// switch to next sample's view
void visual_next_sample(VisualApp *app)
{
    app->selected_sample = (app->selected_sample + 1) % app->input.rows;
}

// switch to previous sample's view
void visual_prev_sample(VisualApp *app)
{
    if (app->selected_sample == 0) {
        app->selected_sample = app->input.rows - 1;
    } else {
        app->selected_sample--;
    }
}
/*
q / Esc    quit
  Space      pause
  n / Right  next sample
  p / Left   previous sample
  t          train once
  r          randomize
  Up         faster training
  Down       slower training
*/
void visual_handle_events(VisualApp *app)
{
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            app->running = false;  // close the window when user click close button
        }

        if (event.type != SDL_KEYDOWN) {
            continue;  // if not key press continue
        }

        switch (event.key.keysym.sym) {
        case SDLK_ESCAPE:
        case SDLK_q:  // quit key
            app->running = false;
            break;
        case SDLK_SPACE:
            app->paused = !app->paused;
            break;
        case SDLK_n:
        case SDLK_RIGHT:
            visual_next_sample(app);
            break;
        case SDLK_p:
        case SDLK_LEFT:
            visual_prev_sample(app);
            break;
        case SDLK_t:
            visual_train_once(app);
            if (visual_has_output(app)) {
                app->cost = nn_cost(*app->nn, app->input, app->output);
                visual_record_cost(app);
            }
            break;
        case SDLK_r:
            nn_rand(*app->nn, 0.0f, 1.0f);
            app->train_step = 0;
            app->cost_history_count = 0;
            if (visual_has_output(app)) {
                app->cost = nn_cost(*app->nn, app->input, app->output);
                visual_record_cost(app);
            }
            break;
        case SDLK_UP:
            if (app->train_per_frame < 512) {
                app->train_per_frame *= 2;
            }
            break;
        case SDLK_DOWN:
            if (app->train_per_frame > 1) {
                app->train_per_frame /= 2;
            }
            break;
        default:
            break;
        }
    }
}

VisualVec2 visual_neuron_pos(NN nn, size_t layer, size_t neuron)
{
    size_t layer_count = nn.count + 1;
    size_t neuron_count = nn.as[layer].cols;
    float left = 120.0f;
    float right = (float)VISUAL_WINDOW_WIDTH - (float)VISUAL_GRAPH_WIDTH - (float)VISUAL_GRAPH_GAP;
    float top = 110.0f;
    float bottom = (float)VISUAL_WINDOW_HEIGHT - 170.0f;
    float x = left;
    float y = (top + bottom) * 0.5f;

    if (layer_count > 1) {
        x = left + ((right - left) * (float)layer) / (float)(layer_count - 1);
    }

    if (neuron_count > 1) {
        y = top + ((bottom - top) * (float)neuron) / (float)(neuron_count - 1);
    }

    return (VisualVec2){x, y};
}

void visual_draw_filled_circle(SDL_Renderer *renderer, int cx, int cy, int radius)
{
    int r2 = radius * radius;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= r2) {
                SDL_RenderDrawPoint(renderer, cx + x, cy + y);
            }
        }
    }
}

void visual_draw_thick_line(SDL_Renderer *renderer,
                            VisualVec2 from,
                            VisualVec2 to,
                            int width)
{
    int half = width / 2;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            SDL_RenderDrawLine(renderer,
                               (int)from.x + dx,
                               (int)from.y + dy,
                               (int)to.x + dx,
                               (int)to.y + dy); 
        }
    }
}

void visual_draw_connections(SDL_Renderer *renderer, NN nn)
{
    for (size_t layer = 0; layer < nn.count; ++layer) {
        for (size_t from = 0; from < nn.ws[layer].rows; ++from) {
            for (size_t to = 0; to < nn.ws[layer].cols; ++to) {
                float w = CAL_MAT(nn.ws[layer], from, to);
                float strength = visual_clampf(fabsf(w) / 2.0f, 0.1f, 1.0f);
                int width = 1 + (int)(strength * 4.0f);

                if (w >= 0.0f) {
		    // postive color
                    SDL_SetRenderDrawColor(renderer,
                                           visual_color_channel(20.0f + strength * 40.0f),
                                           visual_color_channel(190.0f + strength * 65.0f),
                                           visual_color_channel(230.0f + strength * 25.0f),
                                           visual_color_channel(140.0f + strength * 115.0f));
				       } else {
					   //negative color
                    SDL_SetRenderDrawColor(renderer,
                                           visual_color_channel(235.0f + strength * 20.0f),
                                           visual_color_channel(35.0f + strength * 45.0f),
                                           visual_color_channel(45.0f + strength * 35.0f),
                                           visual_color_channel(150.0f + strength * 105.0f));
                }

                visual_draw_thick_line(renderer,
                                       visual_neuron_pos(nn, layer, from),
                                       visual_neuron_pos(nn, layer + 1, to),
                                       width);
            }
        }
    }
}

void visual_draw_neurons(SDL_Renderer *renderer, NN nn)
{
    for (size_t layer = 0; layer <= nn.count; ++layer) {
        for (size_t neuron = 0; neuron < nn.as[layer].cols; ++neuron) {
            VisualVec2 p = visual_neuron_pos(nn, layer, neuron);
            float a = visual_clampf(CAL_MAT(nn.as[layer], 0, neuron), 0.0f, 1.0f);

            SDL_SetRenderDrawColor(renderer, 230, 232, 222, 255); // color for the neuron circle, its white here
            visual_draw_filled_circle(renderer, (int)p.x, (int)p.y, VISUAL_NODE_RADIUS + 3);

            SDL_SetRenderDrawColor(renderer,
                                   visual_color_channel(34.0f + a * 210.0f),
                                   visual_color_channel(76.0f + a * 150.0f),
                                   visual_color_channel(96.0f + a * 95.0f),
                                   255);
            visual_draw_filled_circle(renderer, (int)p.x, (int)p.y, VISUAL_NODE_RADIUS);
            if (layer > 0) { // if its not the input layer
                float b = CAL_MAT(nn.bs[layer - 1], 0, neuron);
                float strength = visual_clampf(fabsf(b) / 2.0f, 0.1f, 1.0f);
                int marker_width = 10 + (int)(strength * 28.0f);
		// bias marking
                SDL_Rect marker = {
                    (int)p.x - marker_width / 2,
                    (int)p.y - VISUAL_NODE_RADIUS - 15,
                    marker_width,
                    5,
                };
                if (b >= 0.0f) {
                    SDL_SetRenderDrawColor(renderer, 50, 255, 95, 255);
                } else {
                    SDL_SetRenderDrawColor(renderer, 255, 55, 210, 255);
                }
                SDL_RenderFillRect(renderer, &marker);
            }
        }
    }
}

void visual_draw_cost_history(SDL_Renderer *renderer, VisualApp *app)
{
    if (app->cost_history_count < 2) {
        return;
    }

    int x = VISUAL_WINDOW_WIDTH - VISUAL_GRAPH_WIDTH - VISUAL_GRAPH_MARGIN;
    int y = (VISUAL_WINDOW_HEIGHT - VISUAL_GRAPH_HEIGHT) / 2;
    int w = VISUAL_GRAPH_WIDTH;
    int h = VISUAL_GRAPH_HEIGHT;
    float max_cost = 0.001f;
    float x_scale = (float)(w - 1) / (float)(VISUAL_HISTORY_COUNT - 1);

    for (size_t i = 0; i < app->cost_history_count; ++i) {
        if (app->cost_history[i] > max_cost) {
            max_cost = app->cost_history[i];
        }
    }

    SDL_Rect frame = {x, y, w, h};
    SDL_SetRenderDrawColor(renderer, 43, 45, 50, 255);
    SDL_RenderFillRect(renderer, &frame);
    SDL_SetRenderDrawColor(renderer, 92, 96, 105, 255);
    SDL_RenderDrawRect(renderer, &frame);

    SDL_SetRenderDrawColor(renderer, 235, 196, 88, 255);
    for (size_t i = 1; i < app->cost_history_count; ++i) {
        float a = app->cost_history[i - 1] / max_cost;
        float b = app->cost_history[i] / max_cost;
        int x0 = x + (int)((float)(i - 1) * x_scale);
        int x1 = x + (int)((float)i * x_scale);
        int y0 = y + h - 6 - (int)(visual_clampf(a, 0.0f, 1.0f) * (float)(h - 12));
        int y1 = y + h - 6 - (int)(visual_clampf(b, 0.0f, 1.0f) * (float)(h - 12));
        SDL_RenderDrawLine(renderer, x0, y0, x1, y1);
    }
}

void visual_draw_network(SDL_Renderer *renderer, VisualApp *app)
{
    visual_draw_connections(renderer, *app->nn);
    visual_draw_neurons(renderer, *app->nn);
    visual_draw_cost_history(renderer, app);
}

void visual_update_window_title(SDL_Window *window, VisualApp *app)
{
    char title[256];

    if (visual_has_output(app)) {
        snprintf(title,
                 sizeof(title),
                 "NN: cost = %.6f | step = %zu | sample = %zu/%zu | speed = %zu | %s",
                 app->cost,
                 app->train_step,
                 app->selected_sample + 1,
                 app->input.rows,
                 app->train_per_frame,
                 app->paused ? "paused" : "training");
    } else {
        snprintf(title,
                 sizeof(title),
                 "NN visual | sample %zu/%zu | forward only",
                 app->selected_sample + 1,
                 app->input.rows);
    }

    SDL_SetWindowTitle(window, title);
}

int visual_run(VisualApp *app, const char *title, int max_frames)
{
    int frames = 0;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(title,
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          VISUAL_WINDOW_WIDTH,
                                          VISUAL_WINDOW_HEIGHT,
                                          0);
    if (window == NULL) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window,
                                                -1,
                                                SDL_RENDERER_ACCELERATED |
                                                    SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    }

    if (renderer == NULL) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    while (app->running) {
        visual_handle_events(app);

        if (!app->paused) {
            visual_update_training(app);
        }

        visual_update_forward_sample(app);
        visual_update_window_title(window, app);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // background color
        SDL_RenderClear(renderer);
        visual_draw_network(renderer, app);
        SDL_RenderPresent(renderer);

        if (max_frames > 0 && ++frames >= max_frames) {
            app->running = false;
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

#endif  // visual implement
#endif //visual h
