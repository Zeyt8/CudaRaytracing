#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "utils/float3_helpers.cuh"

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