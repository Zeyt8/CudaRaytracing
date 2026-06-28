#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "utils/float3_helpers.cuh"

#define pi 3.1415926535897932385f

__host__ __device__ inline uint32_t getRandom(uint32_t& state)
{
    state ^= state >> 16;
    state *= 0x21F0AAADu;
    state ^= state >> 15;
    state *= 0xD35A2D97u;
    state ^= state >> 15;
    return state;
}

__host__ __device__ inline float3 randomUnitVector(uint32_t& state) {
    while (true) {
        float3 p = float3((float)getRandom(state) / UINT32_MAX - 0.5f, (float)getRandom(state) / UINT32_MAX - 0.5f, (float)getRandom(state) / UINT32_MAX - 0.5f);
        float lensq = lengthSquared(p);
        if (1e-160 < lensq && lensq <= 1)
        {
            return p / sqrt(lensq);
        }
    }
}

__device__ __host__ inline float3 randomOnHemisphere(const float3& normal, uint32_t& state) {
    float3 on_unit_sphere = randomUnitVector(state);
    if (dot(on_unit_sphere, normal) > 0.0)
    {
        return on_unit_sphere;
    }
    else
    {
        return -on_unit_sphere;
    }
}

template<typename T>
__device__ __host__ inline T linearToGamma(T linearComponent)
{
    return sqrt(linearComponent) * (linearComponent > 0);
}

template<typename T>
__device__ __host__ inline T reflect(const T& v, const T& n) {
    return v - n * dot(v, n) * 2;
}

template<typename T>
__device__ __host__ inline T refract(const T& uv, const T& n, float etai_over_etat) {
    float cos_theta = min(dot(-uv, n), 1.0f);
    T r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    T r_out_parallel = n * -sqrt(abs(1.0 - lengthSquared(r_out_perp)));
    return r_out_perp + r_out_parallel;
}

template<typename T>
__device__ __host__ inline T reflectance(T cosine, T refraction_index) {
    T r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ __host__ inline float degToRad(float degrees) {
    return degrees * pi / 180.0f;
}