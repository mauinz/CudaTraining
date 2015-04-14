#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
void matrixMultiplyCPU(float *a, float *b, float *c, int width)
{
  float result;
  for(int row=0; row<width; row++)
  {
    for(int col=0; col<width; col++)
    {
      result = 0;
      for(int k = 0; k<width; k++)
      {
        result += a[row * width + k] * b[k * width + col];
      }
      c[row * width + col] = result;
    }
  }
}

__global__ void matrixMultiplySimple(float *a, float *b, float *c, int width)
{
  //calculate the row and column index of the element
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float result = 0;
  //do dot product between row of a and column of b
  for(int i=0; i<width; i++)
  {
    result += a[row*width+i] * b[i*width+col];
  }
  //write out this thread's result
  c[row*width+col] = result;
}
global__ void matrixMultiplyOptimised(float *a, float *b, float *c, int width)
{
  //create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  //allocate 2D tiles in __shared__ memory
  __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
  //calculate the row & column index of the element
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float result = 0;
  for(int p=0; p<width/TILE_WIDTH; p++)
  {
    //collaboratively load tiles into __shared__
    s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
    s_b[ty][tx] = b[(p*TILE_WIDTH + ty) * width + col];
    //wait until all data is loaded before allowing any thread in this block to continue
    __syncthreads();
    //do dot product between row of s_a and column of s_b
    for(int i=0; i<TILE_WIDTH; i++)
    {
      result += s_a[ty][i] * s_b[i][tx];
    }
    //wait until all threads are finished with the data before allowing any thread in this block to continue
    __syncthreads();
  }
  //write out this thread's result
  c[row*width+col] = result;
}


