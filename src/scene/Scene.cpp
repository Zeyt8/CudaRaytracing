#include "Scene.h"

std::vector<float3>& Scene::GetObjects()
{
    return _objects;
}

std::vector<int>& Scene::GetObjectMaterials()
{
    return _objectMaterials;
}

std::vector<Material>& Scene::GetMaterials()
{
    return _materials;
}

void Scene::AddObject(float3 object, int material)
{
    _objects.push_back(object);
    _objectMaterials.push_back(material);
}

void Scene::AddMaterial(Material material)
{
    _materials.push_back(material);
}
