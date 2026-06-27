#include "raytracer_kernels.cuh"

#include <cstdint>

#include "utils/float3_helpers.cuh"
#include "utils/math_utils.cuh"

__device__ int idx(int x, int y, int width)
{
    return y * width + x;
}

__global__ void k_setPixelCenterAndDir(int width, int height, float3 pixel100Loc, float3 pixelDeltaU, float3 pixelDeltaV, float3 cameraCenter, float3* rayDirs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float3 pixelCenter = pixel100Loc + (pixelDeltaU * x) + (pixelDeltaV * y);
    float3 rayDir = pixelCenter - cameraCenter;

    rayDirs[idx(x, y, width)] = rayDir;
}

__device__ float hit_sphere(const float3& center, double radius, const float3& rayCenter, const float3& rayDir) {
    float3 oc = center - rayCenter;
    float a = lengthSquared(rayDir);
    float h = dot(rayDir, oc);
    float c = lengthSquared(oc) - radius * radius;
    float discriminant = h * h - a * c;
    if (discriminant < 0)
    {
        return -1.0f;
    }
    else
    {
        return (h - sqrtf(discriminant)) / a;
    }
}

struct HitInfo
{
    bool hit = false;
    float3 o;
    float3 d;
};

__constant__ float scales[3] = { 10.0f, 0.2f, 0.2f };

__device__ float3 getRayColor(float3 rayOrigin, float3 rayDir, const SceneInfo& sceneInfo, uint32_t& state, HitInfo& hitInfo)
{
    hitInfo.hit = false;

    int closestHitObj = -1;
    float closestP = 0;
    for (int i = 0; i < sceneInfo.objCount; i++)
    {
        float p = hit_sphere(sceneInfo.objects[i], scales[i], rayOrigin, rayDir);
        if (p > 0)
        {
            closestHitObj = i;
            closestP = p;
        }
    }
    if (closestHitObj != -1)
    {
        float3 n = normalized((rayOrigin + rayDir * closestP - sceneInfo.objects[closestHitObj]));
        float3 dir = randomOnHemisphere(n, state);
        
        hitInfo.hit = true;
        hitInfo.o = rayOrigin + rayDir * closestP;
        hitInfo.d = dir;

        if (dot(rayDir, n) > 0) {}
        else {}

        return float3(0.5f, 0.5f, 0.5f);
    }
    else
    {
        return float3(255, 255, 255);
    }
}

__global__ void k_raytrace(float3 rayOrigin, float3* __restrict__ rayDirs, RenderingInfo renderingInfo, SceneInfo sceneInfo, uint32_t initState, uchar4* __restrict__ buffer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= renderingInfo.width || y >= renderingInfo.height) return;

    float3 color = float3(0, 0, 0);
    uint32_t state = initState + x + y * renderingInfo.width;

    for (int i = 0; i < renderingInfo.sampleCount; i++)
    {
        float3 rayDir = rayDirs[idx(x, y, renderingInfo.width)];
        rayDir.x += ((float)getRandom(state) / UINT32_MAX - 0.5f) * (1.0f / renderingInfo.height);
        rayDir.y += ((float)getRandom(state) / UINT32_MAX - 0.5f) * (1.0f / renderingInfo.height);

        HitInfo hitInfo;
        float3 accColor = getRayColor(rayOrigin, rayDir, sceneInfo, state, hitInfo);
        if (hitInfo.hit)
        {
            for (int depth = 0; depth < 50; depth++)
            {
                accColor = accColor * getRayColor(hitInfo.o, hitInfo.d, sceneInfo, state, hitInfo);
                if (!hitInfo.hit) break;
            }
        }
        color = color + accColor / renderingInfo.sampleCount;
    }

    buffer[idx(x, y, renderingInfo.width)] = uchar4(color.x, color.y, color.z, 255);
}
