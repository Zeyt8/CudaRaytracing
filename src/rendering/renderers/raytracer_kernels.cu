#include "raytracer_kernels.cuh"

#include <cuda_tile.h>

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

__device__ float hit_sphere(const float3& center, float radius, const float3& rayCenter, const float3& rayDir) {
    float3 oc = center - rayCenter;
    float a = lengthSquared(rayDir);
    float h = dot(rayDir, oc);
    float c = lengthSquared(oc) - radius * radius;
    float discriminant = h * h - a * c;
    return (discriminant < 0) * -1 + (discriminant >= 0) * (h - sqrtf(discriminant)) / a;
}

struct HitInfo
{
    bool hit = false;
    float3 o;
    float3 d;
};

__constant__ float scales[6] = { 20.0f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f };

__device__ float3 getRayColor(float3 rayOrigin, float3 rayDir, const SceneInfo& sceneInfo, uint32_t& state, HitInfo& hitInfo)
{
    namespace ct = cuda::tiles;

    hitInfo.hit = false;

    int closestHitObj = -1;
    float closestP = FLT_MAX;
    for (auto i : ct::irange(0, sceneInfo.objCount))
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

        float3 hitPos = rayOrigin + rayDir * closestP;
        float3 n = normalized(hitPos - sceneInfo.objects[closestHitObj]);
        float cos_theta = min(dot(-normalized(rayDir), n), 1.0);
        float F_dielectric = reflectance(cos_theta, refraction);
        float p_spec = F_dielectric * metallic + (1 - metallic);
        p_spec = min(max(p_spec, 0.02f), 0.98f);

        float3 scatterDir;
        bool choseSpecular = (float)getRandom(state) / UINT32_MAX < p_spec;
        if (choseSpecular)
        {
            float ri = (dot(rayDir, n) > 0) ? (1 / refraction) : refraction;

            float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
            bool cannotRefract = ri * sin_theta > 1.0f;
            float reflectProb = reflectance(cos_theta, refraction);
            if (refraction == 0 ||
                cannotRefract ||
                reflectance(cos_theta, ri) > (float)getRandom(state) / UINT32_MAX ||
                (float)getRandom(state) / UINT32_MAX < reflectProb)
            {
                scatterDir = reflect(normalized(rayDir), n) + randomUnitVector(state) * roughness;
            }
            else
            {
                scatterDir = refract(normalized(rayDir), n, ri) + randomUnitVector(state) * roughness;
            }
        }
        else
        {
            scatterDir = n + randomOnHemisphere(n, state);
        }

        if (aproxZero(scatterDir))
        {
            scatterDir = n;
        }
        
        hitInfo.hit = (refraction != 0 || dot(scatterDir, n) > 0);
        hitInfo.o = hitPos + scatterDir * 0.001f;
        hitInfo.d = scatterDir;

        float3 diffusePart = albedo * metallic;
        float3 specularPart = albedo * (1 - metallic) + float3(1, 1, 1) * metallic;

        return (choseSpecular ? specularPart : diffusePart) * hitInfo.hit;
    }
    else
    {
        float a = 0.5f * (normalized(rayDir).y + 1.0f);
        return float3(1, 1, 1) * (1.0f - a) + float3(0.5f, 0.7f, 1) * a;
    }
}

__global__ void k_raytrace(float3 rayOrigin, float3* __restrict__ rayDirs, RenderingInfo renderingInfo, SceneInfo sceneInfo, uint32_t initState, uchar4* __restrict__ buffer)
{
    namespace ct = cuda::tiles;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= renderingInfo.width || y >= renderingInfo.height) return;

    float3 color = float3(0, 0, 0);
    uint32_t state = initState + x + y * renderingInfo.width;
    float3 initialRayDir = rayDirs[idx(x, y, renderingInfo.width)];

    for (auto i : ct::irange(0, renderingInfo.sampleCount))
    {
        float3 rayDir = initialRayDir;
        rayDir.x += ((float)getRandom(state) / UINT32_MAX - 0.5f) * renderingInfo.pixelDeltaU.x;
        rayDir.y += ((float)getRandom(state) / UINT32_MAX - 0.5f) * renderingInfo.pixelDeltaV.y;
        float3 randomInDisk = randomInUnitDisk(state);
        float3 focusShift = (renderingInfo.defocusDiskU * randomInDisk.x) + (renderingInfo.defocusDiskV * randomInDisk.y);
        rayDir = rayDir - focusShift;

        HitInfo hitInfo;
        float3 accColor = getRayColor(rayOrigin + focusShift, rayDir, sceneInfo, state, hitInfo);
        for(auto depth : ct::irange(0, 50))
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
