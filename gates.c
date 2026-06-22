#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float sigmoidf(float x){
  return 1.f / (1.f + expf(-x));
}
//OR-gate
float train[][3] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 1},
};
#define train_count (sizeof(train)/sizeof(train[0]))

float cost(float w1, float w2){
  float result = 0.0f;
  for(size_t i = 0; i < train_count; i++){
    float x1= train[i][0];
    float x2= train[i][1];
    float y = sigmoidf(x1 * w1 + x2*w2);
    float d = y - train[i][2];
    result += d * d;
  }
  result /= train_count;
  return result;
}

// y = x * w;
float rand_float(void){
  return (float) rand()/(float) RAND_MAX;
}

int main(){
  for(float x = -10.f; x<=10; x+=1.f){
    printf("%f => %f\n", x, sigmoidf(x));
  }
  srand(time(0));
  float eps = 10e-3;
  float rate = 10e-3;
  float w1 = rand_float(); // weight
  float w2 = rand_float();
  for(size_t i = 0; i<500; ++i){
    printf("cost = %f, w1 = %f, w2 = %f\n", cost(w1, w2), w1, w2);
    float dcost1 = (cost(w1+eps, w2) - cost(w1, w2))/eps;
    float dcost2 = (cost(w1, w2+eps) - cost(w1, w2))/eps; 
    w1 -= rate * dcost1;
    w2 -= rate * dcost2;
     
  }
  return 0;
   }
 
