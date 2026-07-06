#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stddef.h>
#include <math.h>


#ifndef MALLOC
#include <stdlib.h>
#define MALLOC malloc
#endif //malloc

#ifndef ASSERT
#include <assert.h>
#define ASSERT assert
#endif // assert

#define CAL_MAT(a, i , j) (a).es[(i)*(a).stride + (j)] // macro to simple print the mat
#define PRINT_MAT(a) mat_print(a, #a)

#define LIST_COUNT(a) sizeof(a)/sizeof(a[0])

typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float * es;
}Mat;

typedef struct{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;  // counts of a, should be b+1 or w+1.
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
}NN;


Mat mat_malloc(size_t a, size_t b);

void mat_fill(Mat a, float n);
void mat_sig(Mat a);
void mat_dot(Mat des, Mat a, Mat b);
Mat mat_subm(Mat des, Mat a);
Mat mat_row(Mat des, size_t row);
void mat_copy(Mat des, Mat a);
void mat_sum(Mat des, Mat a);
void mat_mul(Mat des, Mat a);
void mat_rand(Mat a, float low, float high);
void mat_print(Mat a, char *, int padding);
#define MAT_PRINT(a) mat_print(a, #a, 3)

// nn settings
NN nn_config(size_t count, size_t * layer_struct);
void nn_print(NN n, char * name);
void nn_rand(NN n, float low, float high);
void nn_forward(NN n);
void nn_diff(NN n, NN g, float eps, Mat in, Mat out);
void nn_backprop(NN n, NN g, Mat in, Mat out);
float nn_cost(NN n, Mat in, Mat out);
void nn_learn (NN n, NN g, float rate);
void nn_zero(NN n);

#define NN_PRINT(a) nn_print(a, #a)
#define NN_INPUT(a)  a.as[0]
#define NN_OUTPUT(a)  a.as[(a).count]



float sigmoidf(float x){
    return 1.f/(1.f + expf(-x));
}
float rand_float(){
    return (float)rand()/ RAND_MAX;
}

#ifdef NN_IMPLEMENT
Mat mat_malloc(size_t a, size_t b){
    Mat m;
    m.rows = a;
    m.cols = b;
    m.stride = b;
    m.es = MALLOC(sizeof(*m.es)* m.rows * m.cols);
    return m;
}

void mat_fill(Mat a, float n){
   for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(a, i, j) = n;
	}
 }   
}
Mat mat_row(Mat des, size_t row){
    return (Mat) {
	.rows = 1,
	.stride = des.stride,
	.cols = des.cols,
	.es = &CAL_MAT(des, row, 0),
    };
}

void mat_copy(Mat des, Mat a){
    ASSERT(a.rows == des.rows);
    ASSERT(a.cols == des.cols);
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(des, i, j) = CAL_MAT(a, i, j);
	}
    }

}

void mat_sum(Mat des, Mat a){
    ASSERT(a.rows == des.rows);
    ASSERT(a.cols == des.cols);
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(des, i, j) += CAL_MAT(a, i, j);
	}
    }
     
}
void mat_sig(Mat a){
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(a, i, j) = sigmoidf(CAL_MAT(a, i, j));
	}
    }
}

void mat_dot(Mat des, Mat a, Mat b){
    ASSERT(a.cols == b.rows);
    ASSERT(b.cols == des.cols);
    ASSERT(a.rows == des.rows);
    for(size_t i = 0; i < des.rows; ++i){
	for(size_t j = 0; j < des.cols; ++j){
	    CAL_MAT(des, i, j)=0;
	    for(size_t k = 0; k < a.cols; ++k){
		CAL_MAT(des, i, j) += CAL_MAT(a, i, k)* CAL_MAT(b, k, j);
	    }
	}
    }
}
void mat_print(Mat a, char *s, int padding){
    printf("%*s%s = [\n", padding,"",s);
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    printf("%*s   %f ",padding,"",CAL_MAT(a, i, j));
	}
	printf("\n");
    }
    printf("%*s]\n", padding, "");
}
void mat_rand(Mat a, float low, float high){
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(a, i, j) = rand_float()*(high - low) + low;
	}
    }
}



NN NN_config(size_t count, size_t * layer_struct){
    NN n;
    n.count = count-1; // number of inner layers
    n.bs = MALLOC(sizeof(*n.bs) * n.count);
    ASSERT(n.bs != NULL);
    n.ws =MALLOC(sizeof(*n.ws) * n.count);
    ASSERT(n.ws != NULL);
    n.as =MALLOC(sizeof(*n.as) * (n.count+1));
    ASSERT(n.as != NULL);

    n.as[0] = mat_malloc(1,layer_struct[0]);

    
    for(size_t i = 1; i < count; ++i){
	n.bs[i-1] = mat_malloc(1, layer_struct[i]);
	n.ws[i-1] = mat_malloc(n.as[i-1].cols, layer_struct[i]);
	n.as[i] = mat_malloc(1, layer_struct[i]);
    }
    return n;
}

void nn_print(NN n, char * name){
    printf("[\n");
    char buf[128]; // truncate to 128
    for(size_t i = 0; i<n.count; ++i){
	snprintf(buf, sizeof(buf), "bs%zu", i);
	mat_print(n.bs[i], buf, 3);
	snprintf(buf, sizeof(buf), "ws%zu", i);
	mat_print(n.ws[i], buf, 3);
    }
    printf("\n]"); 
}

void nn_rand(NN n, float low, float high){
    for(size_t i = 0; i<n.count; ++i){
	mat_rand(n.ws[i],low, high);
	mat_rand(n.bs[i],low, high);
    }
}


void nn_forward(NN n){
    for(size_t i = 0; i < n.count; i++){
	mat_dot(n.as[i+1], n.as[i], n.ws[i]);
	mat_sum(n.as[i+1], n.bs[i]);
	mat_sig(n.as[i+1]);
    }
	
}


float nn_cost(NN n, Mat in, Mat out){
    float res = 0.f;
    ASSERT(in.rows == out.rows);
    ASSERT(NN_OUTPUT(n).cols == out.cols);
    size_t isize = in.rows;
    for(size_t i = 0; i< isize; i++){
	Mat x = mat_row(in, i);
	Mat y = mat_row(out, i);
	mat_copy(n.as[0],x);
	nn_forward(n);
	for(size_t q = 0; q < out.cols; ++q){
	    float diff =CAL_MAT(NN_OUTPUT(n), 0, q) - CAL_MAT(y, 0, q);
	    res += diff * diff;
	    }
	}
	return res/isize; // divide by the input size
    }



void nn_diff(NN n, NN g, float eps, Mat in, Mat out){
    float saved;

    float c = nn_cost(n, in, out);
    for(size_t i = 0; i < n.count; ++i){
	for(size_t j = 0; j < n.ws[i].rows ; ++j){
	    for(size_t k = 0; k < n.ws[i].cols; ++k){
	    saved = CAL_MAT(n.ws[i], j, k);
	    CAL_MAT(n.ws[i], j, k)+= eps;
	    CAL_MAT(g.ws[i], j, k) = (nn_cost(n, in, out)-c)/eps;
	    CAL_MAT(n.ws[i], j, k) = saved;
	}
    }
}


    for(size_t i = 0; i < n.count; ++i){
	for(size_t j = 0; j < n.bs[i].rows ; ++j){
	    for(size_t k = 0; k < n.bs[i].cols; ++k){
	    saved = CAL_MAT(n.bs[i], j, k);
	    CAL_MAT(n.bs[i], j, k)+= eps;
	    CAL_MAT(g.bs[i], j, k) = (nn_cost(n, in, out)-c)/eps;
	    CAL_MAT(n.bs[i], j, k) = saved;
	}
    }
}


}
void nn_zero(NN n){
    for(size_t i = 0; i < n.count; ++i){
	mat_fill(n.as[i], 0);
	mat_fill(n.bs[i], 0);
	mat_fill(n.ws[i], 0);
    }
    mat_fill(n.as[n.count], 0);
}
void nn_learn (NN n, NN g, float rate){
    for(size_t i = 0; i < n.count; ++i){
	for(size_t j = 0; j < n.ws[i].rows ; ++j){
	    for(size_t k = 0; k < n.ws[i].cols; ++k){
		CAL_MAT(n.ws[i], j, k) -= rate * CAL_MAT(g.ws[i], j, k);
	    }
	}
    }


    for(size_t i = 0; i < n.count; ++i){
	for(size_t j = 0; j < n.bs[i].rows ; ++j){
	    for(size_t k = 0; k < n.bs[i].cols; ++k){
		CAL_MAT(n.bs[i], j, k) -= rate * CAL_MAT(g.bs[i], j, k);
	    }
	}
    }    
}
void nn_backprop(NN n, NN g, Mat in, Mat out){
    ASSERT(in.rows == out.rows);
    ASSERT(NN_OUTPUT(n).cols == out.cols);
    nn_zero(g);

    for(size_t i = 0; i < in.rows; ++i){
	mat_copy(NN_INPUT(n), mat_row(in, i));
	nn_forward(n);
	for(size_t j = 0; j < n.count; ++j){
	    mat_fill(g.as[j], 0);
	}
	for(size_t j = 0; j < out.cols; ++j){
	    CAL_MAT(NN_OUTPUT(g), 0, j) = CAL_MAT(NN_OUTPUT(n), 0, j)  - CAL_MAT(out, i, j);
	}

	for(size_t j = n.count; j > 0; --j){
	    for(size_t k = 0; k < n.as[j].cols; ++k){
		float da = CAL_MAT(g.as[j], 0, k);
		float a = CAL_MAT(n.as[j], 0, k);
		CAL_MAT(g.bs[j-1], 0, k) += 2*a*(1-a)*da;
		for(size_t m = 0; m < n.as[j-1].cols; ++m){
		    
		    float pa = CAL_MAT(n.as[j-1], 0, m);
		    float w = CAL_MAT(n.ws[j-1], m, k);
		    CAL_MAT(g.as[j-1], 0, m) += 2*a*(1-a)*da*w;
		    CAL_MAT(g.ws[j-1], m, k) += 2*da*a*(1-a)*pa;		    
		}
	    }
	}
    }

    for(size_t k = 0; k < n.count; ++k){
	for(size_t i = 0; i < g.ws[k].rows; ++i){
	    for(size_t j = 0; j < g.ws[k].cols; ++j){ 
		CAL_MAT(g.ws[k], i, j) /= in.rows;
	    }
	}
	for(size_t i = 0; i < g.bs[k].rows; ++i){
	    for(size_t j = 0; j < g.bs[k].cols; ++j){
		CAL_MAT(g.bs[k], i, j) /= in.rows;
	    }
	}	
    }
}
#endif // nn implement
#endif //nn h
