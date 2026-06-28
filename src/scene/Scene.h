#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "rendering/Material.h"

class Scene
{
public:
	std::vector<float3>& GetObjects();
	std::vector<int>& GetObjectMaterials();
	std::vector<Material>& GetMaterials();
	void AddObject(float3 object, int material);
	void AddMaterial(Material material);

private:
	std::vector<float3> _objects;
	std::vector<int> _objectMaterials;
	std::vector<Material> _materials;
};