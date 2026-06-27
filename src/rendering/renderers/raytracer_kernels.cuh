#pragma once

#include <cuda_runtime.h>

__global__ void k_setPixelCenterAndDir(int width, int height, float3 pixel100Loc, float3 pixelDeltaU, float3 pixelDeltaY, float3 cameraCenter, float3* rayDirs);

struct RenderingInfo
{
	int width;
	int height;
	int sampleCount;
};

struct SceneInfo
{
	float3* objects;
	int objCount;
};

__global__ void k_raytrace(float3 rayOrigin, float3* __restrict__ rayDirs, RenderingInfo renderingInfo, SceneInfo sceneInfo, uint32_t initState, uchar4* __restrict__ buffer);