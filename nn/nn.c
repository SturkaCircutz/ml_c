#define NN_IMPLEMENT
#include "nn.h"
#include "time.h"


int main(){
typedef struct{
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
}Xor;



Xor Xor_config(Xor m){
    Xor m;
    m.a0 = mat_malloc(1,2);
    m.w1 = mat_malloc(2,2);
    m.b1 = mat_malloc(1,2);
    m.a1 = mat_malloc(1,2);
    m.w2 = mat_malloc(2,1);
    m.b2 = mat_malloc(1,1);
    m.a2 = mat_malloc(1,1);
    return m;
}   
void forward(Xor m){
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}
    

float cost(Xor m, Mat in, Mat out){
    ASSERT(in.rows == out.rows); // make sure that each input gets one output
    ASSERT(m.a2.cols == out.cols); // the output cols number matches expected output cols
    float c = 0;
    size_t n = in.rows;
    for(size_t i = 0; i < out.rows; ++i){
	Mat x = mat_row(in, i);
	Mat y = mat_row(out, i);
	mat_copy(m.a0, x);
	forward(m);
	for(size_t j = 0; j < out.cols; ++j){
	    float d = CAL_MAT(m.a2, 0, j)  - CAL_MAT(y, 0, j);
	    c+= d*d;
	}
    }
    return c/n;
   
}
void grad(Xor m, float eps){
    for(size_t i = 0; i<)
}
void finite_diff(Xor m, ){
    
}


mat_rand(m.a0,0, 1);
mat_rand(m.w1,0, 1);
mat_rand(m.b1,0, 1);
mat_rand(m.a1,0, 1);
mat_rand(m.w2,0, 1);
mat_rand(m.b2,0, 1);
mat_rand(m.a2,0, 1);


size_t stride = 3;
float train[] ={
    0, 1, 1,
    1, 0, 1,
    0, 0, 0,
    1, 1, 1,
};


Mat in ={
    .rows = 4,
    .cols = 2,
    .stride = stride,
    .es = &train[0],

};

Mat out ={
    .rows = 4,
    .cols = 1,
    .stride = stride,
    .es = &train[2],
};

PRINT_MAT(in);
PRINT_MAT(out);

printf("cost = %f\n",cost(m, in, out));

#if 0
for(size_t i = 0 ;i < 2;++i){
    for(size_t j = 0 ;j < 2;++j){
	forward(m, i, j);
	printf("%d ^ %d = %f\n", i, j, *m.a2.es);
    }
}
#endif 
return 0;    
}


