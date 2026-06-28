#include "raytracer_kernels.cuh"

#include <cstdint>
#include <cfloat>

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

__constant__ float scales[4] = { 20.0f, 0.2f, 0.2f, 0.2f };

__device__ float3 getRayColor(float3 rayOrigin, float3 rayDir, const SceneInfo& sceneInfo, uint32_t& state, HitInfo& hitInfo)
{
    hitInfo.hit = false;

    int closestHitObj = -1;
    float closestP = FLT_MAX;
    for (int i = 0; i < sceneInfo.objCount; i++)
    {
        float p = hit_sphere(sceneInfo.objects[i], scales[i], rayOrigin, rayDir);
        if (p > 0 && p < closestP)
        {
            closestHitObj = i;
            closestP = p;
        }
    }
    if (closestHitObj != -1)
    {
        float4 albedo4 = sceneInfo.materials[sceneInfo.objectMaterials[closestHitObj]].albedo;
        float roughness = sceneInfo.materials[sceneInfo.objectMaterials[closestHitObj]].roughness;
        float metallic = sceneInfo.materials[sceneInfo.objectMaterials[closestHitObj]].metallic;
        float refraction = sceneInfo.materials[sceneInfo.objectMaterials[closestHitObj]].refractionIndex;
        float3 albedo = make_float3(albedo4.x, albedo4.y, albedo4.z);

        float3 n = normalized((rayOrigin + rayDir * closestP - sceneInfo.objects[closestHitObj]));
        float ri = (dot(rayDir, n) > 0) ? (1 / refraction) : refraction;
        float cos_theta = min(dot(-normalized(rayDir), n), 1.0);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannotRefract = ri * sin_theta > 1.0f;
        float3 scatterDir;
        if (refraction == 0 || cannotRefract || reflectance(cos_theta, ri) > getRandom(state) % UINT32_MAX)
        {
            scatterDir = reflect(normalized(rayDir), n) + randomUnitVector(state) * roughness;
        }
        else
        {
            scatterDir = refract(normalized(rayDir), n, ri);
        }
        
        hitInfo.hit = (refraction != 0 || dot(scatterDir, n) > 0);
        hitInfo.o = rayOrigin + rayDir * closestP + scatterDir * 0.001f;
        hitInfo.d = scatterDir;

        return float3(albedo.x, albedo.y, albedo.z) * hitInfo.hit;
    }
    else
    {
        float a = 0.5f * (normalized(rayDir).y + 1.0f);
        return float3(1, 1, 1) * (1.0f - a) + float3(0.5f, 0.7f, 1) * a;
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
        rayDir.x += ((float)getRandom(state) / UINT32_MAX - 0.5f) * renderingInfo.pixelDeltaU.x;
        rayDir.y += ((float)getRandom(state) / UINT32_MAX - 0.5f) * renderingInfo.pixelDeltaV.y;
        float3 randomInDisk = randomInUnitDisk(state);
        float3 focusShift = (renderingInfo.defocusDiskU * randomInDisk.x) + (renderingInfo.defocusDiskV * randomInDisk.y);
        rayDir = rayDir - focusShift;

        HitInfo hitInfo;
        float3 accColor = getRayColor(rayOrigin + focusShift, rayDir, sceneInfo, state, hitInfo);
        for (int depth = 0; depth < 50; depth++)
        {
            if (!hitInfo.hit) break;
            accColor = accColor * getRayColor(hitInfo.o, hitInfo.d, sceneInfo, state, hitInfo);
        }
        color = color + accColor / renderingInfo.sampleCount;
    }

    color.x = linearToGamma(color.x) * 255;
    color.y = linearToGamma(color.y) * 255;
    color.z = linearToGamma(color.z) * 255;
    buffer[idx(x, y, renderingInfo.width)] = uchar4(color.x, color.y, color.z, 255);
}
