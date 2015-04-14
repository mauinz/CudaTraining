#include <stdio.h>
#include <cuda.h>
#define N 4096 
#define G 4
#define B 1024

__global__ void vectorAddKernel(int * a, int * b, int * c){
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index];

}
int main(){
  dim3 grid(G, 1, 1); //e.g. dim3 grid(4,1,1)
  dim3 block(B, 1, 1); //e.g. dim3 bock(128,1,1)
  
  int a_h[N];
  int b_h[N];
  int c_h[N];
  int *a_d;
  int *b_d;
  int *c_d;

  for(int i=0; i<N; i++)
  {
    a_h[i] = i;
    b_h[i] = i*2;
  }
  
  cudaMalloc((void**)&a_d, N*sizeof(int));
  cudaMalloc((void**)&b_d, N*sizeof(int));
  cudaMalloc((void**)&c_d, N*sizeof(int));
  cudaMemcpy(a_d, a_h, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, N*sizeof(int), cudaMemcpyHostToDevice);
  

  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime; 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  vectorAddKernel<<<grid,block >>>(a_d, b_d, c_d);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaMemcpy(c_h, c_d, N*sizeof(int), cudaMemcpyDeviceToHost);
  
  for(int i=0; i<N; i++)
  {
    printf("%i+%i = %i\n",a_h[i], b_h[i], c_h[i]);
  }

  printf("Time to calculate results: %f ms.\n", elapsedTime);
  cudaFree(a_h);
  cudaFree(b_h);
  cudaFree(c_h);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
