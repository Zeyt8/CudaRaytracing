#include "Raytracer.h"

#include <cuda/cmath>

#include <chrono>

#include "raytracer_kernels.cuh"
#include "utils/float3_helpers.cuh"

Raytracer::Raytracer(Scene* scene, int width, int height) : Renderer(scene, width, height)
{
    float viewportHeight = 1.0f;
    float viewportWidth = viewportHeight * ((float)width / height);
    float3 viewportU = float3(viewportWidth, 0, 0);
    float3 viewportV = float3(0, viewportHeight, 0);

    float3 pixelDeltaU = viewportU / (float)width;
    float3 pixelDeltaV = viewportV / (float)height;

    float3 viewportUpperLeft = _cameraPos + float3(0, 0, _cameraFocalLength) - (viewportU + viewportV) / 2;
    float3 pixel00Loc = viewportUpperLeft + (pixelDeltaU + pixelDeltaV) / 2;

    cudaMalloc(&_rayDirs, width * height * sizeof(float3));

    dim3 block(32, 32);
    dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
    k_setPixelCenterAndDir<<<block, grid>>>(width, height, pixel00Loc, pixelDeltaU, pixelDeltaV, _cameraPos, _rayDirs);

    std::vector<float3> objects = scene->GetObjects();
    cudaMallocHost(&_h_Objects, objects.size() * sizeof(float3));
    std::memcpy(_h_Objects, objects.data(), objects.size() * sizeof(float3));
    cudaMalloc(&_d_Objects, objects.size() * sizeof(float3));
    cudaMemcpy(_d_Objects, _h_Objects, objects.size() * sizeof(float3), cudaMemcpyDefault);

    cudaDeviceSynchronize();
}

void Raytracer::Draw(uchar4* pbo)
{
    dim3 block(64, 64);
    dim3 grid(cuda::ceil_div(_width, block.x), cuda::ceil_div(_height, block.y));
    RenderingInfo ri = {
        _width,
        _height,
        100,
    };
    SceneInfo si = {
        _d_Objects,
        _scene->GetObjects().size(),
    };
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    k_raytrace<<<block, grid>>>(_cameraPos, _rayDirs, ri, si, ms, pbo);

    cudaDeviceSynchronize();
}
