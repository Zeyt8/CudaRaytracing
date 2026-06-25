#include "full_color.h"

#include <cuda/cmath>

__global__ void k_full_color(uchar4 color, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    buffer[y * width + x] = color;
}

void full_color(uchar4 color, uchar4* pbo)
{
    dim3 block(32, 32);
    dim3 grid(cuda::ceil_div(1080, block.x), cuda::ceil_div(720, block.y));
    k_full_color<<<block, grid>>>(color, pbo, 1080, 720);
    cudaDeviceSynchronize();
}
