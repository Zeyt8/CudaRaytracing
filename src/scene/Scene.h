#pragma once

#include <cuda_runtime.h>

#include <vector>

class Scene
{
public:
	std::vector<float3>& GetObjects();
	void AddObject(float3 object);

private:
	std::vector<float3> _objects;
};