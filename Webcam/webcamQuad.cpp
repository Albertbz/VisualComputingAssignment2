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

#include <cmath>
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
#include "transforms/Transforms.hpp"

using namespace std;

// Helper function to initialize the window
bool initWindow(std::string windowName);

// --- Simple global transform state for mouse interaction (UV space) -----
static bool g_isDragging = false;
static double g_lastX = 0.0, g_lastY = 0.0;
static float g_translateU = 0.0f, g_translateV = 0.0f;
static float g_scale = 1.0f;
static float g_rotation = 0.0f;  // degrees, positive = CCW
// Transform toggles accessible from callbacks
static bool g_transformsEnabled = false;
static bool g_transformsUseCPU =
    false;  // when true, apply transforms on CPU (cv::Mat)
static bool g_gpuTransformActive =
    false;  // whether we have set the GPU transform shader

// GLFW callbacks (defined here so they can access the static globals)
static void scroll_callback(GLFWwindow* win, double xoffset, double yoffset) {
    // Zoom around current cursor position
    double mx, my;
    int w, h;
    glfwGetCursorPos(win, &mx, &my);
    glfwGetWindowSize(win, &w, &h);
    if (w <= 0 || h <= 0) return;
    // If GPU, invert mx and my
    if (g_gpuTransformActive) {
        mx = w - mx;
        my = h - my;
    }
    // Convert to UV (0..1). Note: window y is top-down so invert Y to get
    // UV-space where V increases upwards.
    float px = (float)(mx / (double)w);
    float py = (float)(my / (double)h);

    float oldScale = g_scale;
    // scale exponentially for smooth zooming
    // if GPU, invert zoom direction
    float dir = g_gpuTransformActive ? -1.0f : 1.0f;
    float factor = powf(1.1f, (float)yoffset * dir);
    float newScale = oldScale * factor;

    // Keep the point under cursor fixed. The shader composes scale around
    // the image center, so we must account for the center (cx,cy).
    // Derived: t_new = t_old + (s_old - s_new) * (p - c)
    float s_old = oldScale;
    float s_new = newScale;
    float cx = 0.5f, cy = 0.5f;
    g_translateU = g_translateU + (s_old - s_new) * (px - cx);
    g_translateV = g_translateV + (s_old - s_new) * (py - cy);
    g_scale = s_new;
}

static void mouse_button_callback(GLFWwindow* win, int button, int action,
                                  int mods) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        g_isDragging = true;
        glfwGetCursorPos(win, &g_lastX, &g_lastY);
    } else if (action == GLFW_RELEASE) {
        g_isDragging = false;
    }
}

static void cursor_pos_callback(GLFWwindow* win, double xpos, double ypos) {
    if (!g_isDragging) return;
    int w, h;
    glfwGetWindowSize(win, &w, &h);
    if (w <= 0 || h <= 0) return;
    double dx = xpos - g_lastX;
    double dy = ypos - g_lastY;
    // Convert pixel delta to UV delta. GLFW Y is top-down, so invert the
    // vertical delta: moving the mouse up should increase V.
    // If shift is held, interpret horizontal drag as rotation.
    int shiftLeft = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT);
    int shiftRight = glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT);
    if (shiftLeft == GLFW_PRESS || shiftRight == GLFW_PRESS) {
        // rotation sensitivity: degrees per pixel (tweakable)
        const float rotSens = 0.35f;
        g_rotation += (float)dx * rotSens;
    } else {
        float du = (float)(dx / (double)w);
        float dv = (float)(dy / (double)h);
        g_translateU += du;
        g_translateV += dv;
    }
    g_lastX = xpos;
    g_lastY = ypos;
}

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
    // Install mouse/scroll callbacks for interactive transforms
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
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
    // Add T = toggle transforms on/off, C = toggle CPU/GPU transform mode
    const int keysToWatch[] = {GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4,
                               GLFW_KEY_G, GLFW_KEY_E, GLFW_KEY_P, GLFW_KEY_T,
                               GLFW_KEY_C, GLFW_KEY_R};
    bool prevKeyState[sizeof(keysToWatch) / sizeof(keysToWatch[0])] = {false};

    // Transform toggles
    // (moved to file-level globals so callbacks can see them)

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
                    case GLFW_KEY_T:
                        g_transformsEnabled = !g_transformsEnabled;
                        cout << "Transforms "
                             << (g_transformsEnabled ? "ENABLED" : "DISABLED")
                             << "\n";
                        // If enabling GPU transforms, switch shader
                        if (g_transformsEnabled && !g_transformsUseCPU) {
                            setGPUShaderOnQuad(
                                Transforms::gpuFragmentPathTransform());
                            g_gpuTransformActive = true;
                        } else {
                            // disabling transforms or switching to CPU: restore
                            // default shader
                            if (g_gpuTransformActive) {
                                setDefaultShaderOnQuad();
                                g_gpuTransformActive = false;
                            }
                        }
                        break;
                    case GLFW_KEY_C:
                        g_transformsUseCPU = !g_transformsUseCPU;
                        cout << "Transform mode: "
                             << (g_transformsUseCPU ? "CPU" : "GPU") << "\n";
                        // If switching to GPU while transforms are enabled, set
                        // GPU shader
                        if (g_transformsEnabled && !g_transformsUseCPU) {
                            setGPUShaderOnQuad(
                                Transforms::gpuFragmentPathTransform());
                            g_gpuTransformActive = true;
                        } else {
                            if (g_gpuTransformActive) {
                                setDefaultShaderOnQuad();
                                g_gpuTransformActive = false;
                            }
                        }
                        break;
                    case GLFW_KEY_R:
                        // Reset transforms to identity
                        g_translateU = 0.0f;
                        g_translateV = 0.0f;
                        g_scale = 1.0f;
                        g_rotation = 0.0f;
                        cout << "Transforms reset to identity\n";
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

            // Apply CPU transforms if enabled and requested
            if (g_transformsEnabled && g_transformsUseCPU) {
                // Convert UV-space translate/scale to pixel-space. UV +V is up,
                // image pixel Y increases downward, so invert V when mapping
                // to pixel-space.
                float dx_pixels = -g_translateU * (float)frame.cols;
                // UV +V is up, image pixel Y increases downward, so invert V
                // when mapping to pixel-space for CPU transforms.
                float dy_pixels = g_translateV * (float)frame.rows;
                // Apply scale around center first, then translate
                if (fabs(g_scale - 1.0f) > 1e-6f) {
                    Transforms::applyScaleCPU(frame, g_scale, g_scale);
                }
                // Apply rotation around center (degrees)
                if (fabs(g_rotation) > 1e-6f) {
                    Transforms::applyRotateCPU(frame, g_rotation);
                }
                if (fabs(dx_pixels) > 0.0f || fabs(dy_pixels) > 0.0f) {
                    Transforms::applyTranslateCPU(frame, dx_pixels, dy_pixels);
                }
            }

            // Flip the frame vertically for OpenGL texture coordinates
            cv::flip(frame, frame, 0);

            // Upload the frame to the GPU
            videoTexture->update(frame.data, frame.cols, frame.rows, true);
        }

        // Render the scene from the camera's point of view
        // Bind the quad's shader and upload the UV transform if present
        myQuad->bindShaders();
        {
            GLint prog = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
            if (prog != 0) {
                GLint loc = glGetUniformLocation((GLuint)prog, "uTransform");
                if (loc >= 0) {
                    // Build a 3x3 UV transform: translate * T(center) *
                    // S(scale) * T(-center)
                    float cx = 0.5f, cy = 0.5f;
                    glm::mat3 T_neg(1.0f);
                    T_neg[2][0] = -cx;
                    T_neg[2][1] = -cy;
                    glm::mat3 S(1.0f);
                    S[0][0] = g_scale;
                    S[1][1] = g_scale;
                    glm::mat3 T_back(1.0f);
                    T_back[2][0] = cx;
                    T_back[2][1] = cy;
                    // Rotation around center (convert degrees to radians)
                    glm::mat3 R(1.0f);
                    float ang = glm::radians(g_rotation);
                    float ca = std::cos(ang);
                    float sa = std::sin(ang);
                    // column-major: set columns accordingly
                    R[0][0] = ca;
                    R[0][1] = sa;
                    R[1][0] = -sa;
                    R[1][1] = ca;
                    glm::mat3 T_translate(1.0f);
                    T_translate[2][0] = g_translateU;
                    T_translate[2][1] = g_translateV;
                    // Compensate for the quad's aspect ratio so rotations in UV
                    // space behave like pixel-space rotations. The quad is
                    // created with the video aspect ratio, so X and Y are
                    // scaled differently; to rotate without warping we scale X
                    // by aspect, rotate, then undo the scale.
                    float aspect = 1.0f;
                    if (!frame.empty() && frame.rows != 0) {
                        aspect = (float)frame.cols / (float)frame.rows;
                    }
                    glm::mat3 A(1.0f);     // scale X by aspect
                    glm::mat3 Ainv(1.0f);  // inverse: scale X by 1/aspect
                    A[0][0] = aspect;
                    Ainv[0][0] = 1.0f / aspect;

                    // Compose with aspect compensation: translate * back * Ainv
                    // * R * S * A * T_neg
                    glm::mat3 M =
                        T_translate * T_back * Ainv * R * S * A * T_neg;
                    glUniformMatrix3fv(loc, 1, GL_FALSE, &M[0][0]);
                }
                // Provide texel offset to shaders that sample neighbors
                GLint locTexel =
                    glGetUniformLocation((GLuint)prog, "texelOffset");
                if (locTexel >= 0) {
                    // frame.cols/rows are > 0 here (frame checked earlier)
                    glUniform2f(locTexel, 1.0f / (float)frame.cols,
                                1.0f / (float)frame.rows);
                }
                // Provide an edge threshold uniform used by GPU edge shader.
                // When not in GPU edge mode we set it to 0.0 to preserve
                // previous behavior (raw gradient magnitude).
                GLint locEdge =
                    glGetUniformLocation((GLuint)prog, "edgeThreshold");
                if (locEdge >= 0) {
                    float thr =
                        (currentMode == FilterMode::GPU_EDGE) ? 0.2f : 0.0f;
                    glUniform1f(locEdge, thr);
                }
            }
        }
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
