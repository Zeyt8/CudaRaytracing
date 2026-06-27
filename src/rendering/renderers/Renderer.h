#pragma once

#include <cuda_runtime.h>

#include "scene/Scene.h"

class Renderer
{
public:
	Renderer(Scene* scene, int width, int height);
	virtual void Draw(uchar4* pbo) = 0;

protected:
	Scene* _scene;
	int _width;
	int _height;
};