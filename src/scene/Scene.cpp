#include "Scene.h"

std::vector<float3>& Scene::GetObjects()
{
    return _objects;
}

void Scene::AddObject(float3 object)
{
    _objects.push_back(object);
}
