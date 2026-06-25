#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stddef.h>

#ifndef MALLOC
#include <stdlib.h>
#define MALLOC malloc
#endif //nn_malloc

#ifndef ASSERT
#include <assert.h>
#define ASSERT assert
#endif // assert

#define PRINT_MAT(a, i , j) (a).es[(i)*(a).cols + (j)] // macro to simple print the mat

typedef struct{
    size_t rows;
    size_t cols;
    float * es;
}Mat;


Mat mat_malloc(size_t a, size_t b);
void mat_fill(Mat a, float n);
void mat_sum(Mat des, Mat a);
void mat_mul(Mat des, Mat a);
void mat_rand(Mat a, float low, float high);
void mat_print(Mat a);

float rand_float(){
    return (float)rand()/ RAND_MAX;
}

#ifdef NN_IMPLEMENT
Mat mat_malloc(size_t a, size_t b){
    Mat m;
    m.rows = a;
    m.cols = b;
    m.es = MALLOC(sizeof(*m.es)* m.rows * m.cols);
    return m;
}

void mat_fill(Mat a, float n){
   for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    PRINT_MAT(a, i, j) = n;
	}
 }   
}
void mat_sum(Mat des, Mat a){
    ASSERT(a.rows == des.rows);
    ASSERT(a.cols == des.cols);
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    PRINT_MAT(des, i, j) += PRINT_MAT(a, i, j);
	}
	printf("\n");
    }
     
}
void mat_dot(Mat des, Mat a){
    
}
void mat_print(Mat a){
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    printf("%f ",PRINT_MAT(a, i, j));
	}
	printf("\n");
    }
}
void mat_rand(Mat a, float low, float high){
    for(size_t i = 0; i < a.rows; ++i){
	for(size_t j = 0; j < a.cols; ++j){
	    PRINT_MAT(a, i, j) = rand_float()*(high - low) + low;
	}
    }
}
    
#endif // nn implement
#endif //nn h
