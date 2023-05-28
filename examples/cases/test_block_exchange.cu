#include <stdio.h>
#include <iostream>

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>



__global__ void ExampleKernel(int *d_data) {
  // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
  typedef cub::BlockExchange<int, 128, 4> BlockExchange;

  // Allocate shared memory for BlockExchange
  __shared__ typename BlockExchange::TempStorage temp_storage;

  // Load a tile of data striped across threads
  int thread_data[4];
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
  if (threadIdx.x == 0) {
    printf("%d, %d, %d, %d\n", thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
  }

  // Collectively exchange data into a blocked arrangement across threads
  BlockExchange(temp_storage).StripedToBlocked(thread_data);

  if (threadIdx.x == 0) {
    printf("%d, %d, %d, %d\n", thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
  }
}

int main(int argc, char** argv)
{

  int *h_in           = new int[128 * 4];
  for (int i = 0; i < 128 * 4; i ++) {
    h_in[i] = i;
  }

  int *d_in           = NULL;
  cudaMalloc((void**)&d_in, sizeof(int) * 128 * 4);
  cudaMemcpy(d_in, h_in, sizeof(int) * 128 * 4, cudaMemcpyHostToDevice);

  ExampleKernel<<<1, 128>>>(d_in);
  cudaDeviceSynchronize();
}