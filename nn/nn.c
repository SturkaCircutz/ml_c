#define NN_IMPLEMENT
#include "nn.h"
#include "time.h"


int main(){
typedef struct{
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
}Xor;


float forward(Xor m, float x, float y){
    CAL_MAT(m.a0, 0, 0) = x;
    CAL_MAT(m.a0, 0, 1) = y;
    
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
    return *m.a2.es;
    
}
    
Xor m;
m.a0 = mat_malloc(1,2);
m.w1 = mat_malloc(2,2);
m.b1 = mat_malloc(1,2);


m.a1 = mat_malloc(1,2);
m.w2 = mat_malloc(2,1);
m.b2 = mat_malloc(1,1);

m.a2 = mat_malloc(1,1);


mat_rand(m.a0,0, 1);
mat_rand(m.w1,0, 1);
mat_rand(m.b1,0, 1);
mat_rand(m.a1,0, 1);
mat_rand(m.w2,0, 1);
mat_rand(m.b2,0, 1);
mat_rand(m.a2,0, 1);


for(size_t i = 0 ;i < 2;++i){
    for(size_t j = 0 ;j < 2;++j){
	printf("%d ^ %d = %f\n", i, j, forward(m, i, j));
    }
}

return 0;    
}


