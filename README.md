# Webcam -> OpenGL (Visual Computing Assignment 2)

This project captures a live webcam feed using OpenCV and renders it on an OpenGL textured quad. It includes CPU and GPU filter implementations (grayscale, edge, pixelate).

This README explains how to build and run the project on macOS (zsh). Adjust paths/commands for Linux or Windows as needed.

## Prerequisites

- Xcode command-line tools (for compiler and make):

  ```bash
  xcode-select --install
  ```

- Homebrew (recommended) and the following packages:

  ```bash
  brew install cmake pkg-config glfw glm opencv
  ```

Notes:
- `glad` is included in `external/glad/` in the repo. The project uses GLFW, GLM and OpenCV. If you installed OpenCV via Homebrew, CMake should find it through `pkg-config` / CMake find modules.

## Build

1. Create a build directory (the repository already contains a `build/` directory in this workspace):

   ```bash
   mkdir -p build
   cd build
   ```

2. Run CMake and build:

   ```bash
   cmake ..
   make -j$(sysctl -n hw.ncpu)
   ```

3. After a successful build the `Webcam` executable is produced in the `build/` directory. Run it with:

   ```bash
   ./Webcam
   ```

If the executable can't find shader files at runtime, ensure your working directory is `build/` (or run `./Webcam` from that directory) so relative shader paths (e.g. `Webcam/videoTextureShader.frag`) resolve correctly.

## Runtime controls

- Keys (when the GLFW window is focused):
  - `1` — None
  - `2` — CPU Grayscale
  - `3` — CPU Edge
  - `4` — CPU Pixelate
  - `G` — GPU Grayscale
  - `E` — GPU Edge
  - `P` — GPU Pixelate
  - `ESC` — Quit

## Camera selection

The example code opens `cv::VideoCapture cap(1);` by default. If you want to use the default camera device `0`, change the index in `Webcam/webcamQuad.cpp`:

```cpp
cv::VideoCapture cap(0); // or 1, 2, ... depending on your system
```

Rebuild after changing the index.