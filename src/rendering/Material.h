#pragma once

#include <cuda_runtime.h>

struct Material
{
	float4 albedo;
	float roughness;
	float metallic;
	float refractionIndex;
};