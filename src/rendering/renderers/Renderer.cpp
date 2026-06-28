#include "Renderer.h"

Renderer::Renderer(Scene* scene, int width, int height, Camera camera)
{
	_scene = scene;
	_width = width;
	_height = height;
	_camera = camera;
}
