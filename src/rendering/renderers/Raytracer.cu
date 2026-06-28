#include "Raytracer.h"

#include <cuda/cmath>

#include <chrono>

#include "utils/float3_helpers.cuh"
#include "utils/math_utils.cuh"

Raytracer::Raytracer(Scene* scene, int width, int height, Camera camera) : Renderer(scene, width, height, camera)
{
    float theta = degToRad(camera.fov);
    float h = tan(theta / 2);
    float viewportHeight = 2 * h * _camera.focalLength;
    float viewportWidth = viewportHeight * ((float)width / height);
    float3 w = normalized(_camera.forward);
    float3 u = normalized(cross(_camera.up, w));
    float3 v = cross(w, u);
    float3 viewportU = u * viewportWidth;
    float3 viewportV = v * viewportHeight;

    float3 pixelDeltaU = viewportU / (float)width;
    float3 pixelDeltaV = viewportV / (float)height;

    float3 viewportUpperLeft = camera.pos + w * camera.focalLength - (viewportU + viewportV) / 2;
    float3 pixel00Loc = viewportUpperLeft + (pixelDeltaU + pixelDeltaV) / 2;
    float defocusRadius = camera.focalLength * tan(degToRad(camera.defocusAngle / 2));
    float3 defocusDiskU = u * defocusRadius;
    float3 defocusDiskV = v * defocusRadius;

    cudaMalloc(&_rayDirs, width * height * sizeof(float3));

    dim3 block(32, 32);
    dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
    k_setPixelCenterAndDir<<<block, grid>>>(width, height, pixel00Loc, pixelDeltaU, pixelDeltaV, camera.pos, _rayDirs);

    std::vector<float3> objects = scene->GetObjects();
    cudaMallocHost(&_h_Objects, objects.size() * sizeof(float3));
    std::memcpy(_h_Objects, objects.data(), objects.size() * sizeof(float3));
    cudaMalloc(&_d_Objects, objects.size() * sizeof(float3));
    cudaMemcpy(_d_Objects, _h_Objects, objects.size() * sizeof(float3), cudaMemcpyDefault);

    std::vector<int> objectMaterials = scene->GetObjectMaterials();
    cudaMallocHost(&_h_ObjectMaterials, objectMaterials.size() * sizeof(int));
    std::memcpy(_h_ObjectMaterials, objectMaterials.data(), objectMaterials.size() * sizeof(int));
    cudaMalloc(&_d_ObjectMaterials, objectMaterials.size() * sizeof(int));
    cudaMemcpy(_d_ObjectMaterials, _h_ObjectMaterials, objectMaterials.size() * sizeof(int), cudaMemcpyDefault);

    std::vector<Material> materials = scene->GetMaterials();
    cudaMallocHost(&_h_Materials, materials.size() * sizeof(Material));
    std::memcpy(_h_Materials, materials.data(), materials.size() * sizeof(Material));
    cudaMalloc(&_d_Materials, materials.size() * sizeof(Material));
    cudaMemcpy(_d_Materials, _h_Materials, materials.size() * sizeof(Material), cudaMemcpyDefault);

    _ri = {
        _width,
        _height,
        pixelDeltaU,
        pixelDeltaV,
        100,
        defocusDiskU,
        defocusDiskV,
    };

    cudaDeviceSynchronize();
}

void Raytracer::Draw(uchar4* pbo)
{
    dim3 block(128, 128);
    dim3 grid(cuda::ceil_div(_width, block.x), cuda::ceil_div(_height, block.y));

    SceneInfo si = {
        _d_Objects,
        _scene->GetObjects().size(),
        _d_ObjectMaterials,
        _d_Materials,
    };
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    k_raytrace<<<block, grid>>>(_camera.pos, _rayDirs, _ri, si, ms, pbo);

    cudaDeviceSynchronize();
}
