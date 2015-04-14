#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

const N = 4096000
__global__ void mulKernel(int *a, int *c)
{
  int tdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tdx < N)
  {
    c[tdx] = a[tdx]*2;
  }
}

int main()
{

  int *a_h[2], *c_h[2];
  //device memory pointers
  int *a_d[2];
  int *c_d[2];
  cudaStream_t stream[2];
  for (int i = 0; i < 2; ++i)
  {
    cudaStreamCreate(&stream[i]); //stream creation
    //pinned memory allocation
    cudaMallocHost((void**)&a_h[i], (N/2)*sizeof(int));
    cudaMallocHost((void**)&c_h[i], (N/2)*sizeof(int));
    //allocate device memory
    cudaMalloc((void**)&a_d[i], (N/2)*sizeof(int));
    cudaMalloc((void**)&b_d[i], (N/2)*sizeof(int));

    cudaMalloc((void**)&c_d[i], (N/2)*sizeof(int));
  }
  //load arrays with some numbers
  for(int i=0; i<2; i++)
  {
    for(int ii=0; ii<N/2; ii++)
    {
      a_h[i][ii] = i*N/2+ii;
    }
  }
  //CUDA events to measure time
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime;
  //start timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 grid(4,1,1);
  dim3 block(1024,1,1);
  //stream 2
  cudaMemcpyAsync(a_d[0], a_h[0], (N/2)*sizeof(int), cudaMemcpyHostToDevice, stream[0]);
  vectorAddKernel <<< grid, block, 0, stream[0]>>>(a_d[0], c_d[0]);
  cudaMemcpyAsync(c_h[0], c_d[0], (N/2)*sizeof(int), cudaMemcpyDeviceToHost, stream[0]);
  //stream 1
  cudaMemcpyAsync(a_d[1], a_h[1], (N/2)*sizeof(int), cudaMemcpyHostToDevice, stream[1]);
  vectorAddKernel <<<grid, block, 0, stream[1]>>>(a_d[1], c_d[1]);
  cudaMemcpyAsync(c_h[1], c_d[1], (N/2)*sizeof(int), cudaMemcpyDeviceToHost, stream[1]);
  
  //stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //print out execution time
  printf("Time to calculate results: %f ms.\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  for (int i = 0; i < 2; ++i)
  {
    cudaStreamDestroy(stream[i]);
    //clean up
    cudaFreeHost(a_h[i]);
    cudaFreeHost(c_h[i]);
  }
  cudaDeviceReset();
  return 0;
}
