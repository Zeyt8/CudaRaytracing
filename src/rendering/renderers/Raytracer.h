#pragma once

#include "Renderer.h"

class Raytracer : Renderer
{
public:
	Raytracer(Scene* scene, int width, int height);
	void Draw(uchar4* pbo);

private:
	const float3 _cameraPos = float3{ 0.0f, 0.0f, 0.0f };
	const float _cameraFocalLength = 1.0f;
	float3* _rayDirs = nullptr;
	float3* _h_Objects = nullptr;
	float3* _d_Objects = nullptr;
};