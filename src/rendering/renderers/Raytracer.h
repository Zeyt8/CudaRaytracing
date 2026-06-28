#pragma once

#include "Renderer.h"

class Raytracer : Renderer
{
public:
	Raytracer(Scene* scene, int width, int height, Camera camera);
	void Draw(uchar4* pbo);

private:
	float3* _rayDirs = nullptr;
	float3 _pixelDeltaU;
	float3 _pixelDeltaV;
};