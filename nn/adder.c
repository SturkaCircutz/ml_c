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
	CAL_MAT(out, i, j) = (z>>j)&1;
	CAL_MAT(out, i, BITS) = overflow;
    }
}
MAT_PRINT(in);
MAT_PRINT(out);
float rate = 1;
size_t  layer_struct[] = {2*BITS, 2*BITS, BITS+1};
NN nn = NN_config(LIST_COUNT(layer_struct), layer_struct);
NN g = NN_config(LIST_COUNT(layer_struct), layer_struct);
nn_rand(nn,0,1);
for(size_t i = 0; i < 1000*10; ++i){
    nn_backprop(nn, g, in, out);
    nn_learn(nn, g, rate);
    printf("cost = %f\n", nn_cost(nn, in, out));    
}
for(size_t i = 0; i < n; ++i){
    for(size_t j = 0; j < n; ++j){
	printf("%zu + %zu =", i, j);
	for(size_t k = 0; k < BITS; ++k){
	    CAL_MAT(NN_INPUT(nn), 0, j) = (i>>k)&1;
	    CAL_MAT(NN_INPUT(nn), 0, j+BITS) = (j>>k)&1;
	}
	nn_forward(nn);
	if(CAL_MAT(NN_OUTPUT(nn), 0, BITS) > 0.5f) printf("overflow\n");
	else{size_t z = 0;
	for(size_t m = 0; m < BITS; ++m){
	    size_t b = CAL_MAT(NN_OUTPUT(nn), 0, m) > 0.5f;
	    z |= b<<m;		
	}
	printf("%zu\n", z);
    }
}
}
}
