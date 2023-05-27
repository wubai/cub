#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// nvcc -arch=sm_70 bank_conflicts.cu -O3

int g_iterations = 100;

template <int STRIDE>
__global__ void ExampleKernel(clock_t *d_elapsed) {
  __shared__ int shared[1024];
  clock_t start = clock();
  shared[threadIdx.x * STRIDE]++;
  clock_t stop = clock();
  *d_elapsed = stop - start;
}

template <int STRIDE>
void Test() {
  clock_t  elapsed_clocks = 0;

  clock_t *d_elapsed  = NULL;
  cudaMalloc((void**)&d_elapsed, sizeof(clock_t));

  for (int i = 0; i < g_iterations; ++i)
  {
    ExampleKernel<STRIDE><<<1, 32>>>(d_elapsed);

    clock_t clocks;
    cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost);
    elapsed_clocks += clocks;
  }
  cudaFree(d_elapsed);
  std::cout <<"stride: " << STRIDE <<  " elapsed_clocks: " \
            << elapsed_clocks  / g_iterations << std::endl;
}


int main(int argc, char** argv)
{
  Test<0>();
  Test<1>();
  Test<2>();
  Test<4>();
  Test<8>();
  Test<16>();
  Test<32>();
  cudaDeviceReset();
}