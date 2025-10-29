/*
 * OpenCV to OpenGL Exercise
 *
 * GOAL: Render a live video feed from a camera onto a 3D object using OpenGL.
 *
 * INSTRUCTIONS:
 * This file is partially complete. Your main task is to complete the section
 * marked "TODO" to create the initial OpenGL texture from a camera frame.
 *
 * The rendering loop has been completed for you as an example.
 *
 */

#include <glad/gl.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#define GLAD_GL_IMPLEMENTATION

#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <common/Camera.hpp>
#include <common/ColorShader.hpp>
#include <common/Object.hpp>
#include <common/Quad.hpp>
#include <common/Scene.hpp>
#include <common/Shader.hpp>
#include <common/Texture.hpp>
#include <common/TextureShader.hpp>
#include <opencv2/opencv.hpp>

#include "filters/Filters.hpp"

using namespace std;

// Helper function to initialize the window
bool initWindow(std::string windowName);

/* ------------------------------------------------------------------------- */
/* main                                                                      */
/* ------------------------------------------------------------------------- */
int main(void) {
    // Open camera
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera. Exiting." << endl;
        return -1;
    }
    cout << "Camera opened successfully." << endl;

    // Initialize OpenGL context
    if (!initWindow("Webcam")) return -1;

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        fprintf(stderr, "Failed to initialize OpenGL context (GLAD)\n");
        cap.release();
        return -1;
    }
    cout << "Loaded OpenGL " << GLAD_VERSION_MAJOR(version) << "."
         << GLAD_VERSION_MINOR(version) << "\n";

    // Basic OpenGL setup
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glClearColor(0.1f, 0.1f, 0.2f, 0.0f);  // A dark blue background
    glEnable(GL_DEPTH_TEST);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Prepare Scene, Shaders, and Objects

    // We get one frame from the camera to determine its size.
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: couldn't capture an initial frame from camera. "
                "Exiting.\n";
        cap.release();
        glfwTerminate();
        return -1;
    }

    // Create objects needed for rendering.
    TextureShader* textureShader =
        new TextureShader("videoTextureShader.vert", "videoTextureShader.frag");
    Scene* myScene = new Scene();
    Camera* renderingCamera = new Camera();
    renderingCamera->setPosition(
        glm::vec3(0, 0, -2.5));  // Move camera back to see the quad

    // Calculate aspect ratio and create a quad with the correct dimensions.
    float videoAspectRatio = (float)frame.cols / (float)frame.rows;
    Quad* myQuad = new Quad(videoAspectRatio);
    myQuad->setShader(textureShader);
    myScene->addObject(myQuad);

    // This variable will hold our OpenGL texture.
    Texture* videoTexture = nullptr;

    // Flip image on the x-axis
    cv::flip(frame, frame, 0);
    videoTexture = new Texture(frame.data, frame.cols, frame.rows, true);

    // We must tell the shader which texture to use.
    textureShader->setTexture(videoTexture);

    // Keys we watch for toggles (kept for backwards compatibility)
    const int keysToWatch[] = {GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4,
                               GLFW_KEY_G, GLFW_KEY_E, GLFW_KEY_P};
    bool prevKeyState[sizeof(keysToWatch) / sizeof(keysToWatch[0])] = {false};

    auto setDefaultShaderOnQuad = [&](void) {
        TextureShader* sh = new TextureShader("videoTextureShader.vert",
                                              "videoTextureShader.frag");
        sh->setTexture(videoTexture);
        myQuad->setShader(
            sh);  // Object takes ownership and will delete previous shader
    };

    auto setGPUShaderOnQuad = [&](const std::string& fragPath) {
        TextureShader* sh =
            new TextureShader("videoTextureShader.vert", fragPath);
        sh->setTexture(videoTexture);
        myQuad->setShader(sh);
    };

    cout << "Filter keys: 1=None, 2=CPU Gray, 3=CPU Edge, 4=CPU Pixelate, "
            "G=GPU Gray, E=GPU Edge, P=GPU Pixelate"
         << endl;

    // Make variables to track current filter
    enum class FilterMode {
        NONE,
        CPU_GRAY,
        CPU_EDGE,
        CPU_PIXELATE,
        GPU_GRAY,
        GPU_EDGE,
        GPU_PIXELATE
    };
    FilterMode currentMode = FilterMode::NONE;

    // Main Render Loop
    while (!glfwWindowShouldClose(window)) {
        // Capture a new frame
        cap >> frame;

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Check for ESC key press
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // --- Handle keyboard toggles (detect on-press events) ---
        for (size_t i = 0; i < sizeof(keysToWatch) / sizeof(keysToWatch[0]);
             ++i) {
            int k = keysToWatch[i];
            bool cur = (glfwGetKey(window, k) == GLFW_PRESS);
            if (cur && !prevKeyState[i]) {
                // Key just pressed
                switch (k) {
                    case GLFW_KEY_1:
                        setDefaultShaderOnQuad();
                        cout << "Filter: NONE\n";
                        currentMode = FilterMode::NONE;
                        break;
                    case GLFW_KEY_2:
                        Filters::applyGrayscaleCPU(frame);
                        setDefaultShaderOnQuad();
                        cout << "Filter: CPU GRAY\n";
                        currentMode = FilterMode::CPU_GRAY;
                        break;
                    case GLFW_KEY_3:
                        Filters::applyCannyCPU(frame);
                        setDefaultShaderOnQuad();
                        cout << "Filter: CPU EDGE\n";
                        currentMode = FilterMode::CPU_EDGE;
                        break;
                    case GLFW_KEY_4:
                        Filters::applyPixelateCPU(frame);
                        setDefaultShaderOnQuad();
                        cout << "Filter: CPU PIXELATE\n";
                        currentMode = FilterMode::CPU_PIXELATE;
                        break;
                    case GLFW_KEY_G:
                        setGPUShaderOnQuad(Filters::gpuFragmentPathGrayscale());
                        cout << "Filter: GPU GRAY\n";
                        currentMode = FilterMode::GPU_GRAY;
                        break;
                    case GLFW_KEY_E:
                        setGPUShaderOnQuad(Filters::gpuFragmentPathEdge());
                        cout << "Filter: GPU EDGE\n";
                        currentMode = FilterMode::GPU_EDGE;
                        break;
                    case GLFW_KEY_P:
                        setGPUShaderOnQuad(Filters::gpuFragmentPathPixelate());
                        cout << "Filter: GPU PIXELATE\n";
                        currentMode = FilterMode::GPU_PIXELATE;
                        break;
                }
            }
            prevKeyState[i] = cur;
        }

        // Update the texture with a new frame from the camera
        if (!frame.empty() && videoTexture != nullptr) {
            // Apply CPU filters if requested (modify frame before upload)
            switch (currentMode) {
                case FilterMode::CPU_GRAY:
                    Filters::applyGrayscaleCPU(frame);
                    break;
                case FilterMode::CPU_EDGE:
                    Filters::applyCannyCPU(frame);
                    break;
                case FilterMode::CPU_PIXELATE:
                    Filters::applyPixelateCPU(frame);
                    break;
                default:
                    // No CPU processing needed
                    break;
            }

            // Flip the frame vertically for OpenGL texture coordinates
            cv::flip(frame, frame, 0);

            // Upload the frame to the GPU
            videoTexture->update(frame.data, frame.cols, frame.rows, true);
        }

        // Render the scene from the camera's point of view
        myScene->render(renderingCamera);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --- Cleanup -----------------------------------------------------------
    cout << "Closing application..." << endl;
    cap.release();
    delete myScene;
    delete renderingCamera;
    delete videoTexture;

    glfwTerminate();
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Helper: initWindow (GLFW)                                                 */
/* ------------------------------------------------------------------------- */
bool initWindow(std::string windowName) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return false;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(1024, 768, windowName.c_str(), NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    return true;
}
