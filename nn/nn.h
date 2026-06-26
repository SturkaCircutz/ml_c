#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stddef.h>
#include <math.h>


#ifndef MALLOC
#include <stdlib.h>
#define MALLOC malloc
#endif //nn_malloc

#ifndef ASSERT
#include <assert.h>
#define ASSERT assert
#endif // assert

#define CAL_MAT(a, i , j) (a).es[(i)*(a).stride + (j)] // macro to simple print the mat
#define PRINT_MAT(a) mat_print(a, #a)



typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float * es;
}Mat;


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
void mat_print(Mat a, char *);

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
void mat_print(Mat a, char *s){
    printf("%s = [\n", s);
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    printf("%f ",CAL_MAT(a, i, j));
	}
	printf("\n");
    }
    printf("]\n");
}
void mat_rand(Mat a, float low, float high){
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    CAL_MAT(a, i, j) = rand_float()*(high - low) + low;
	}
    }
}
    
#endif // nn implement
#endif //nn h
