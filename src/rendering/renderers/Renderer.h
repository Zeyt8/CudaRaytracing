#pragma once

#include <cuda_runtime.h>

#include "scene/Scene.h"
#include "rendering/Material.h"
#include "rendering/Camera.h"

class Renderer
{
public:
	Renderer(Scene* scene, int width, int height, Camera camera);
	virtual void Draw(uchar4* pbo) = 0;

protected:
	Scene* _scene;
	int _width;
	int _height;
	Camera _camera;
	float3* _h_Objects = nullptr;
	float3* _d_Objects = nullptr;
	int* _h_ObjectMaterials = nullptr;
	int* _d_ObjectMaterials = nullptr;
	Material* _h_Materials = nullptr;
	Material* _d_Materials = nullptr;
};