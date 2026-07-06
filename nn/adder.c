#define NN_IMPLEMENT
#define BITS 2
#include <stddef.h>
#include "nn.h"

int main(){
size_t n = (1<<BITS);
Mat in = mat_malloc(n*n, 2*BITS);
Mat out = mat_malloc(n*n, BITS + 1);

for(size_t i = 0; i < in.rows; i++){
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x+y;
    size_t overflow = z >= n;
    for(size_t j = 0; j < BITS; j++){
	CAL_MAT(in, i, j) = (x>>j)&1;
	CAL_MAT(in, i, j + BITS) = (y>>j)&1;
	if(overflow){
	    CAL_MAT(out, i, j) = 0;
	}else{
	    CAL_MAT(out, i, j) = (z>>j)&1;
	}
	    CAL_MAT(out, i, BITS) = overflow;
    }
}
MAT_PRINT(in);
MAT_PRINT(out);
float rate = 1e-1;
size_t  layer_struct[] = {2*BITS, BITS+1, BITS+1};

NN nn = NN_config(LIST_COUNT(layer_struct), layer_struct);
NN g = NN_config(LIST_COUNT(layer_struct), layer_struct);
for(size_t i = 0; i < 1000; ++i){
    nn_backprop(nn, g, in, out);
    nn_learn(nn, g, rate);
    printf("cost = %f\n", nn_cost(nn, in, out));    
}

}
