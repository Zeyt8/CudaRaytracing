#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;               // Prefer NVIDIA GPU
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001; // Prefer AMD GPU
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <chrono>

#include "scene/Scene.h"
#include "rendering/RenderSurface.h"
#include "rendering/renderers/Raytracer.h"

GLFWwindow* initWindow(int width, int height) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Raytracing", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewInit();
    glViewport(0, 0, width, height);
    glClearColor(0.8f, 0.8f, 1.0f, 1.0f);
    return window;
}

int main()
{
    int width = 1080;
    int height = 720;
    GLFWwindow* window = initWindow(width, height);
    cudaSetDevice(0);

    Scene scene;
    scene.AddMaterial(Material(float4(0.5f, 0.5f, 0.5f, 1), 0.8f, 0, 0));
    scene.AddMaterial(Material(float4(1, 0.1f, 0.1f, 1), 0.5f, 0, 0));
    scene.AddMaterial(Material(float4(0.1f, 1, 0.1f, 1), 0.0f, 0, 0));
    scene.AddMaterial(Material(float4(1, 1, 1, 1), 0.0f, 0, 1.33f));
    scene.AddMaterial(Material(float4(1, 0, 0, 1), 0.5f, 0.0f, 0));
    scene.AddMaterial(Material(float4(1, 1, 1, 1), 0.8f, 0, 1.1f));
    scene.AddObject(float3(0.0f, -20.2f, 1.0f), 0);
    scene.AddObject(float3(0.0f, 0, 1.0f), 1);
    scene.AddObject(float3(0.5f, 0, 1.0f), 2);
    scene.AddObject(float3(-0.2f, 0, 0.7f), 3);
    scene.AddObject(float3(-0.5f, 0, 1), 4);
    scene.AddObject(float3(0.2f, 0, 0.7f), 5);

    RenderSurface renderSurface(width, height);
    Raytracer renderer(&scene, width, height, Camera(float3(0, 0.75f, 0), float3(0, -1, 1), float3(0, 1, 1), 1.25f, 90, 2));

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "GPU: " << r << std::endl;

    int frames = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        frames++;

        float delta = std::chrono::duration<float>(currentTime - lastTime).count();
        if (delta >= 1.0f) {
            std::string title = "CUDA Raytracing - FPS: " + std::to_string(frames);
            glfwSetWindowTitle(window, title.c_str());
            frames = 0;
            lastTime = currentTime;
        }

        uchar4* devPtr = renderSurface.MapCudaResource();

        renderer.Draw(devPtr);

        renderSurface.UnmapCudaResource();
        glClear(GL_COLOR_BUFFER_BIT);
        renderSurface.Draw();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    renderSurface.Cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
