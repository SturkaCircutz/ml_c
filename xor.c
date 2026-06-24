#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define train_count 4


typedef struct{
  float or_w1;
  float or_w2;
  float or_b;
  
  float nand_w1;
  float nand_w2;
  float nand_b;
  
  float and_w1;
  float and_w2;
  float and_b;
  
} Xor;


float sigmoidf(float x){
  return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x, float y){
  float a = sigmoidf(m.or_w1 * x + m.or_w2 * y + m.or_b);
  float b = sigmoidf(m.nand_w1 * x + m.nand_w2 * y + m.nand_b);
  return sigmoidf(m.and_w1 *a  + m.and_w2 * b + m.and_b);
}

typedef float sample[3];
//XOR-gate
sample xor_train[]={
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0},
    };
sample *train = xor_train;

float cost(Xor m){
  float res = 0.0f;
  for(size_t i = 0 ; i < train_count;++i){
    float fin = train[i][0]; // first input
    float sin = train[i][1]; // second input
    float actual = forward(m, fin, sin);
    float diff = actual - train[i][2];
    res += diff*diff;
   }
  return res; 
}
// y = x * w;
float rand_float(void){
  return (float) rand()/(float) RAND_MAX;
}
Xor rand_xor(void){
    Xor m;
    m.or_w1 = rand_float(); // weight
    m.or_w2 = rand_float();
    m.or_b = rand_float();
    m.and_w1 = rand_float(); // weight
    m.and_w2 = rand_float();
    m.and_b = rand_float();
    m.nand_w1 = rand_float(); // weight
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    return m;
  }

void print_xor(Xor m){
    printf("or_w1 = %f\n",  m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_b = %f\n", m.or_b);
    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2  = %f\n",m.and_w2);
    printf("and_b = %f\n", m.and_b);
    printf("nand_w1 = %f\n",m.nand_w1);
    printf("nand_w2  = %f\n", m.nand_w2);
    printf("nand_b = %f\n", m.nand_b);
    
}

Xor learn_xor(Xor m, float rate, Xor grad){

    m.or_w1 -= rate * grad.or_w1 ; // weight
    m.or_w2 -= rate * grad.or_w2 ;
    m.or_b -= rate * grad.or_b ;
    m.and_w1 -= rate * grad.and_w1 ; // weight
    m.and_w2 -= rate * grad.and_w2 ;
    m.and_b -= rate * grad.and_b ;
    m.nand_w1 -= rate * grad.nand_w1 ; // weight
    m.nand_w2 -= rate * grad.nand_w2 ;
    m.nand_b -= rate * grad.nand_b ;
    return m;
}
Xor diff_xor(Xor m, float eps){
  Xor g;
  float store;
  float c = cost(m);
  store = m.or_w1;
  m.or_w1 += eps;
  g.or_w1 =(cost(m)-c)/eps;
  m.or_w1 = store;

  store = m.or_w2;
  m.or_w2 += eps;
  g.or_w2 =(cost(m)-c)/eps;
  m.or_w2 = store;
  
  store = m.or_b;
  m.or_b += eps;
  g.or_b =(cost(m)-c)/eps;
  m.or_b = store;

  store = m.and_w1;
  m.and_w1 += eps;
  g.and_w1 =(cost(m)-c)/eps;
  m.and_w1 = store;
  
  store = m.and_w2;
  m.and_w2 += eps;
  g.and_w2 =(cost(m)-c)/eps;
  m.and_w2 = store;

  store = m.and_b;
  m.and_b += eps;
  g.and_b =(cost(m)-c)/eps;
  m.and_b = store;
  
 store = m.nand_w1;
  m.nand_w1 += eps;
  g.nand_w1 =(cost(m)-c)/eps;
  m.nand_w1 = store;

 store = m.nand_w2;
  m.nand_w2 += eps;
  g.nand_w2 =(cost(m)-c)/eps;
  m.nand_w2 = store;

  store = m.nand_b;
  m.nand_b += eps;
  g.nand_b =(cost(m)-c)/eps;
  m.nand_b = store;
  return g;
}
  
int main(){
  float eps = 1e-3;
  float rate = 1e-1;

  Xor m = rand_xor();
  for(size_t i = 0; i < 100000; ++i){
    Xor g = diff_xor(m, eps);
    m = learn_xor(m,rate, g);
    printf("cost = %f\n", cost(m));
  }
  for(size_t i = 0 ; i < 2; ++i){
    for(size_t j = 0 ; j < 2; ++j){
      printf("%zu | %zu = %f\n", i, j, forward(m, i, j) );
    }
  }
  
  return 0;
}
