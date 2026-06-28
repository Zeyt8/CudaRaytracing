#pragma once

#include "Renderer.h"

#include "raytracer_kernels.cuh"

class Raytracer : Renderer
{
public:
	Raytracer(Scene* scene, int width, int height, Camera camera);
	void Draw(uchar4* pbo);

private:
	float3* _rayDirs = nullptr;
	RenderingInfo _ri;
};