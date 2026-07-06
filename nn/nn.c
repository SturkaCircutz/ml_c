#define NN_IMPLEMENT
#include "nn.h"
#include "time.h"


int main(){
    srand(60);
 
float eps = 1e-1;
float rate = 1e-2;

size_t stride = 3;
float train[] ={
    0, 1, 1,
    1, 0, 1,
    0, 0, 0,
    1, 1, 0,
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

size_t nn[] = {2, 2, 1};
    NN test = NN_config(LIST_COUNT(nn), nn);
    NN g = NN_config(LIST_COUNT(nn), nn);
    nn_rand(test, 0, 1);
    for(size_t i = 0; i < 1000*100; ++i){
#if 1 
	nn_diff(test, g, eps, in, out);
#else
       nn_backprop(test, g, in, out);
#endif
nn_learn(test, g, rate);
    }

NN_PRINT(g);
    for(size_t i = 0; i < 2; ++i){
	for(size_t j = 0; j<2; ++j){
	    CAL_MAT(NN_INPUT(test), 0, 0)= i;
	    CAL_MAT(NN_INPUT(test), 0, 1)= j;
	    nn_forward(test);
	    printf("%zu ^ %zu = %f\n",i,j, CAL_MAT(NN_OUTPUT(test), 0, 0));
	}
    }
    
    

    return 0;
    }
typedef struct{
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
}Xor;



Xor Xor_config(void){
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
Xor diff(Xor m, float eps, Mat in, Mat out){
    Xor g = Xor_config();
    float c = cost(m, in, out);
    float store;
    for(size_t i = 0; i<m.w1.rows; ++i){
	for(size_t j = 0; j<m.w1.cols; ++j){
	    store = CAL_MAT(m.w1, i, j);
	    CAL_MAT(m.w1, i, j) += eps;
	    CAL_MAT(g.w1, i, j) = (cost(m, in, out)-c)/eps;
	    CAL_MAT(m.w1, i, j) = store;
	   }
       }
    for(size_t i = 0; i<m.b1.rows; ++i){
	for(size_t j = 0; j<m.b1.cols; ++j){
	    store = CAL_MAT(m.b1, i, j);
	    CAL_MAT(m.b1, i, j) += eps;
	     CAL_MAT(g.b1, i, j) = (cost(m, in, out)-c)/eps;
	    CAL_MAT(m.b1, i, j) = store;
 	   }
       }       

    for(size_t i = 0; i<m.b2.rows; ++i){
	for(size_t j = 0; j<m.b2.cols; ++j){
	    store = CAL_MAT(m.b2, i, j);
	    CAL_MAT(m.b2, i, j) += eps;
	    CAL_MAT(g.b2, i, j) = (cost(m, in, out)-c)/eps;
	    CAL_MAT(m.b2, i, j) = store;
 	   }
       }
    for(size_t i = 0; i<m.w2.rows; ++i){
	for(size_t j = 0; j<m.w2.cols; ++j){
	    store = CAL_MAT(m.w2, i, j);
	    CAL_MAT(m.w2, i, j) += eps;
	    CAL_MAT(g.w2, i, j) = (cost(m, in, out)-c)/eps;
	    CAL_MAT(m.w2, i, j) = store;
 	   }
       }
    return g;
}
void learn (Xor g, Xor m, float rate){
    for(size_t i = 0; i<m.w1.rows; ++i){
	for(size_t j = 0; j<m.w1.cols; ++j){
	    CAL_MAT(m.w1, i, j)-=rate*CAL_MAT(g.w1, i, j);
	}
       }
    for(size_t i = 0; i<m.b1.rows; ++i){
	for(size_t j = 0; j<m.b1.cols; ++j){
	    CAL_MAT(m.b1, i, j)-=rate*CAL_MAT(g.b1, i, j);
 	}
       }       
    for(size_t i = 0; i<m.b2.rows; ++i){
	for(size_t j = 0; j<m.b2.cols; ++j){
	    CAL_MAT(m.b2, i, j)-=rate*CAL_MAT(g.b2, i, j);
	}
       }
    for(size_t i = 0; i<m.w2.rows; ++i){
	for(size_t j = 0; j<m.w2.cols; ++j){
	    CAL_MAT(m.w2, i, j)-=rate*CAL_MAT(g.w2, i, j);
 	   }
       }
       
}
  
