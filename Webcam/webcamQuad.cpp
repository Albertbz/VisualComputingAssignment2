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

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#define GLAD_GL_IMPLEMENTATION

#include <GLFW/glfw3.h>

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

GLFWwindow* window;

// Helper function to initialize the window
bool initWindow(std::string windowName);

// --- Simple global transform state for mouse interaction (UV space) -----
static bool g_isDragging = false;
static double g_lastX = 0.0, g_lastY = 0.0;
static float g_translateU = 0.0f, g_translateV = 0.0f;
static float g_scale = 1.0f;
static float g_rotation = 0.0f;  // degrees, positive = CCW
// Last zoom cursor position in UV (0..1). Used so CPU scaling can pivot
// about the cursor point.
static float g_zoomPivotU = 0.5f, g_zoomPivotV = 0.5f;
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

    // Record pivot for CPU scaling (in UV coordinates). For CPU path we
    // will map these to pixel coordinates before calling applyScaleCPU.
    g_zoomPivotU = px;
    g_zoomPivotV = py;

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
int main(int argc, char** argv) {
    // --- Simple CLI parsing for benchmarking -------------------------
    bool doBenchmark = false;
    std::string benchmarkOut = "benchmark.csv";
    std::string filterArg = "none";     // none, gray, edge, pixelate
    std::string backendArg = "gpu";     // cpu or gpu (filter backend)
    std::string transformsArg = "off";  // off, cpu, gpu
    // optional initial transform values (for benchmark runs)
    float presetTranslateU = 0.0f;
    float presetTranslateV = 0.0f;
    float presetScale = 1.0f;
    float presetRotation = 0.0f;            // degrees
    int targetWidth = 0, targetHeight = 0;  // 0 = native
    int benchFrames = 300;
    bool detailedBenchmark = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--benchmark")
            doBenchmark = true;
        else if (a == "--out" && i + 1 < argc)
            benchmarkOut = argv[++i];
        else if (a == "--filter" && i + 1 < argc)
            filterArg = argv[++i];
        else if (a == "--backend" && i + 1 < argc)
            backendArg = argv[++i];
        else if (a == "--transforms" && i + 1 < argc)
            transformsArg = argv[++i];
        else if (a == "--translateU" && i + 1 < argc)
            presetTranslateU = std::stof(argv[++i]);
        else if (a == "--translateV" && i + 1 < argc)
            presetTranslateV = std::stof(argv[++i]);
        else if (a == "--scale" && i + 1 < argc)
            presetScale = std::stof(argv[++i]);
        else if (a == "--rotation" && i + 1 < argc)
            presetRotation = std::stof(argv[++i]);
        else if (a == "--resolution" && i + 1 < argc) {
            std::string res = argv[++i];
            size_t x = res.find('x');
            if (x != std::string::npos) {
                targetWidth = std::stoi(res.substr(0, x));
                targetHeight = std::stoi(res.substr(x + 1));
            }
        } else if (a == "--frames" && i + 1 < argc) {
            benchFrames = std::stoi(argv[++i]);
        } else if (a == "--detailed") {
            detailedBenchmark = true;
        }
    }
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
    renderingCamera->setPosition(glm::vec3(0, 0, -2.5));

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

    // If user requested a target resolution, set capture properties now.
    if (targetWidth > 0 && targetHeight > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, targetWidth);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, targetHeight);
        cout << "Requested camera resolution: " << targetWidth << "x"
             << targetHeight << endl;
        // Re-query an initial frame at new resolution
        cap >> frame;
        if (!frame.empty()) {
            cv::flip(frame, frame, 0);
            videoTexture->update(frame.data, frame.cols, frame.rows, true);
        }
    }

    // If benchmarking was requested, configure filters/transforms accordingly
    std::ofstream csvOut;
    std::ofstream csvDetailedOut;
    std::vector<double> frameTimesMs;
    std::string buildType =
#ifdef NDEBUG
        "Release";
#else
        "Debug";
#endif

    // (Benchmark configuration that needs runtime symbols is done later
    // after shader helper lambdas and FilterMode are declared.)

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

    // If benchmarking was requested, configure filters/transforms accordingly
    if (doBenchmark) {
        cout << "Running in BENCHMARK mode -> " << benchmarkOut << "\n";
        std::string fa = filterArg;
        for (auto& c : fa) c = (char)tolower(c);
        std::string be = backendArg;
        for (auto& c : be) c = (char)tolower(c);

        if (fa == "none") {
            setDefaultShaderOnQuad();
            currentMode = FilterMode::NONE;
        } else if (fa == "gray") {
            if (be == "cpu") {
                currentMode = FilterMode::CPU_GRAY;
                setDefaultShaderOnQuad();
            } else {
                currentMode = FilterMode::GPU_GRAY;
                setGPUShaderOnQuad(Filters::gpuFragmentPathGrayscale());
            }
        } else if (fa == "edge") {
            if (be == "cpu") {
                currentMode = FilterMode::CPU_EDGE;
                setDefaultShaderOnQuad();
            } else {
                currentMode = FilterMode::GPU_EDGE;
                setGPUShaderOnQuad(Filters::gpuFragmentPathEdge());
            }
        } else if (fa == "pixelate") {
            if (be == "cpu") {
                currentMode = FilterMode::CPU_PIXELATE;
                setDefaultShaderOnQuad();
            } else {
                currentMode = FilterMode::GPU_PIXELATE;
                setGPUShaderOnQuad(Filters::gpuFragmentPathPixelate());
            }
        } else {
            cout << "Unknown filter name '" << filterArg
                 << "', defaulting to none\n";
            setDefaultShaderOnQuad();
            currentMode = FilterMode::NONE;
        }

        // Transforms argument: off, cpu, gpu
        if (transformsArg == "cpu") {
            g_transformsEnabled = true;
            g_transformsUseCPU = true;
            g_gpuTransformActive = false;
        } else if (transformsArg == "gpu") {
            g_transformsEnabled = true;
            g_transformsUseCPU = false;
            setGPUShaderOnQuad(Transforms::gpuFragmentPathTransform());
            g_gpuTransformActive = true;
        } else {
            g_transformsEnabled = false;
            g_transformsUseCPU = false;
            g_gpuTransformActive = false;
        }

        // If transforms are enabled for the benchmark, apply any preset
        // transform values provided on the CLI so the run exercises
        // non-identity transforms.
        if (g_transformsEnabled) {
            g_translateU = presetTranslateU;
            g_translateV = presetTranslateV;
            g_scale = presetScale;
            g_rotation = presetRotation;
            // keep current pivot at center unless user adjusted g_zoomPivot
        }

        // Open CSV for writing
        csvOut.open(benchmarkOut);
        if (!csvOut.is_open()) {
            cerr << "Could not open output CSV '" << benchmarkOut
                 << "' for writing. Will print to stdout instead.\n";
        } else {
            csvOut << "frame_ms,frame_index,filter,backend,resolution,"
                      "transforms,build"
                   << std::endl;
        }
        if (detailedBenchmark) {
            std::string det = benchmarkOut + ".detailed.csv";
            csvDetailedOut.open(det);
            if (csvDetailedOut.is_open()) {
                csvDetailedOut << "frame_index,total_ms,capture_ms,process_ms,"
                                  "transform_ms,upload_ms,draw_ms,filter,"
                                  "backend,resolution,transforms,build"
                               << std::endl;
            } else {
                cerr << "Could not open detailed CSV '" << det
                     << "' for writing.\n";
            }
        }
    }
    // Main Render Loop
    while (!glfwWindowShouldClose(window)) {
        // Start frame timer (include capture + processing + render)
        auto tstart = std::chrono::high_resolution_clock::now();

        // Capture a new frame (this is part of the timed region)
        auto tcap_start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto tcap_end = std::chrono::high_resolution_clock::now();

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
        double capture_ms = 0.0, proc_ms = 0.0, trans_ms = 0.0, upload_ms = 0.0;
        if (!frame.empty() && videoTexture != nullptr) {
            // Apply CPU filters if requested (modify frame before upload)
            auto tproc_start = std::chrono::high_resolution_clock::now();
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
            auto tproc_end = std::chrono::high_resolution_clock::now();
            proc_ms = std::chrono::duration_cast<
                          std::chrono::duration<double, std::milli>>(
                          tproc_end - tproc_start)
                          .count();

            // Apply CPU transforms if enabled and requested
            auto ttrans_start = std::chrono::high_resolution_clock::now();
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
                    // Convert pivot UV -> pixel coordinates (frame has origin
                    // top-left before the vertical flip applied later).
                    // Invert U because we want to map from UV to pixel space
                    double pivotX =
                        (1 - (double)g_zoomPivotU) * (double)frame.cols;
                    double pivotY = (double)g_zoomPivotV * (double)frame.rows;
                    Transforms::applyScaleCPU(frame, g_scale, g_scale, pivotX,
                                              pivotY);
                }
                // Apply rotation around center (degrees)
                if (fabs(g_rotation) > 1e-6f) {
                    Transforms::applyRotateCPU(frame, g_rotation);
                }
                if (fabs(dx_pixels) > 0.0f || fabs(dy_pixels) > 0.0f) {
                    Transforms::applyTranslateCPU(frame, dx_pixels, dy_pixels);
                }
            }
            auto ttrans_end = std::chrono::high_resolution_clock::now();
            trans_ms = std::chrono::duration_cast<
                           std::chrono::duration<double, std::milli>>(
                           ttrans_end - ttrans_start)
                           .count();

            // Flip the frame vertically for OpenGL texture coordinates
            auto tupload_start = std::chrono::high_resolution_clock::now();
            cv::flip(frame, frame, 0);

            // Upload the frame to the GPU
            videoTexture->update(frame.data, frame.cols, frame.rows, true);
            auto tupload_end = std::chrono::high_resolution_clock::now();
            upload_ms = std::chrono::duration_cast<
                            std::chrono::duration<double, std::milli>>(
                            tupload_end - tupload_start)
                            .count();

            capture_ms = std::chrono::duration_cast<
                             std::chrono::duration<double, std::milli>>(
                             tcap_end - tcap_start)
                             .count();
        }

        // Render the scene from the camera's point of view
        // Bind the quad's shader and upload the UV transform if present
        auto tdraw_start = std::chrono::high_resolution_clock::now();
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
        auto tdraw_end = std::chrono::high_resolution_clock::now();

        // End timer for this frame
        auto tend = tdraw_end;
        double ms =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(tend - tstart)
                .count();

        double draw_ms = std::chrono::duration_cast<
                             std::chrono::duration<double, std::milli>>(
                             tdraw_end - tdraw_start)
                             .count();

        if (doBenchmark) {
            // Create a resolution string
            int w = (frame.empty() ? 0 : frame.cols);
            int h = (frame.empty() ? 0 : frame.rows);
            std::ostringstream resos;
            resos << w << "x" << h;

            // Write either to CSV file or stdout
            if (csvOut.is_open()) {
                csvOut << ms << "," << frameTimesMs.size() << "," << filterArg
                       << "," << backendArg << "," << resos.str() << ","
                       << transformsArg << "," << buildType << "\n";
            } else {
                std::cout << ms << "," << frameTimesMs.size() << ","
                          << filterArg << "," << backendArg << ","
                          << resos.str() << "," << transformsArg << ","
                          << buildType << std::endl;
            }
            if (detailedBenchmark && csvDetailedOut.is_open()) {
                csvDetailedOut << frameTimesMs.size() << "," << ms << ","
                               << capture_ms << "," << proc_ms << ","
                               << trans_ms << "," << upload_ms << "," << draw_ms
                               << "," << filterArg << "," << backendArg << ","
                               << resos.str() << "," << transformsArg << ","
                               << buildType << "\n";
            }
            frameTimesMs.push_back(ms);

            if ((int)frameTimesMs.size() >= benchFrames) {
                std::cout << "Benchmark complete: captured "
                          << frameTimesMs.size() << " frames." << std::endl;
                break;
            }
        }
    }

    // If benchmarking, emit a short summary and close CSV
    if (doBenchmark) {
        double sum = 0.0;
        for (double t : frameTimesMs) sum += t;
        double mean =
            frameTimesMs.empty() ? 0.0 : sum / (double)frameTimesMs.size();
        double var = 0.0;
        for (double t : frameTimesMs) var += (t - mean) * (t - mean);
        double stddev = frameTimesMs.size() > 1
                            ? sqrt(var / (frameTimesMs.size() - 1))
                            : 0.0;
        std::cout << "Benchmark summary: frames=" << frameTimesMs.size()
                  << ", mean_ms=" << mean << ", std_ms=" << stddev << "\n";
        if (csvOut.is_open()) csvOut.close();
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
