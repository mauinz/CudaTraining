#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void matrixAddKernel(int * a, int * b, int * c, int  n){
  int col = threadIdx.x + blockDim.x * blockIdx.x; 
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int index = row * N + col;
  if(col < N && row < N)
  {
    c[index] = a[index]+b[index];
  }
  //int index = threadIdx.y*n + threadIdx.x;
  //c[index] = a[index] + b[index]; 

}

void matrixAdd(int *a, int *b, int *c, int n)
{
  int index;
  for(int col=0; col<n; col++)
  {
    for(int row=0; row<n; row++)
    {
      index = row * n + col;
      c[index] = a[index] + b[index];
    }
  }
}
int main(){

  dim3 grid(1, 1, 1);
  dim3 block(N, N, 1);
  int *a_h;
  int *b_h;
  int *c_h;
  int *d_h;
  int *a_d;
  int *b_d;
  int *c_d;
  
  int size;
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime; 
  printf("Number of threads: %i (%ix%i)\n", block.x*block.y, block.x, block.y);
  printf("Number of blocks: %i (%ix%i)\n", grid.x*grid.y, grid.x, grid.y); 

  size = N * N * sizeof(int);
  a_h = (int*) malloc(size);
  b_h = (int*) malloc(size);
  c_h = (int*) malloc(size);
  d_h = (int*) malloc(size);
  for(int i=0; i<N; i++)
  {
    for(int j=0; j<N; j++)
    {
      a_h[i * N + j] = i;
      b_h[i * N + j] = i;
    }
  }

  cudaMalloc((void**)&a_d, size);
  cudaMalloc((void**)&b_d, size);
  cudaMalloc((void**)&c_d, size);

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  matrixAddKernel<<<grid, block>>>(a_d, b_d, c_d, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n",elapsedTime);
  cudaMemcpy(c_h, c_d, size ,cudaMemcpyDeviceToHost);
  cudaEventRecord(start, 0);

  matrixAdd(a_h, b_h, d_h, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop );
  printf("Time to calculate results on CPU: %f ms.\n",elapsedTime); 
  
  for(int i=0; i<N*N; i++)
  {
    if (c_h[i] != d_h[i]) printf("Error: CPU and GPU results do not match\n");
    break;
  }




  return 0;
}
