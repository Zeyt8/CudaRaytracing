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
#include <chrono>

#include "rendering/Renderer.h"
#include "raytracing/full_color.h"

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
    GLFWwindow* window = initWindow(1080, 720);
    cudaSetDevice(0);

    Renderer renderer(1080, 720);

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "Renderer: " << r << std::endl;

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

        uchar4* devPtr = renderer.MapCudaResource();

        full_color(uchar4(0, 255, 255, 255), devPtr);

        renderer.UnmapCudaResource();
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    renderer.Cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
