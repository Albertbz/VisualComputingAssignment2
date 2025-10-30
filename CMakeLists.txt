# CMake entry point
cmake_minimum_required(VERSION 3.10)

include(CMakePrintHelpers)
project(VC_IntroOpenGL)

cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_SOURCE_DIR)

# --- Dependencies ---
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${GLM_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "external"
    ${GLFW_INCLUDE_DIRS}
    .
)

# Use experimental glm features
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

set(ALL_LIBS
    ${OPENGL_LIBRARY}
    glfw
    ${OpenCV_LIBS}
)

add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)


# --------------------------------------------------------------------------
# Part 03 - OpenCV camera feed on textured quad
# --------------------------------------------------------------------------
add_executable(Webcam
common/Shader.cpp
    common/Shader.hpp
	common/ColorShader.cpp
    common/ColorShader.hpp
    common/Camera.cpp
    common/Camera.hpp
    common/Scene.cpp
    common/Scene.hpp
    common/Object.cpp
    common/Object.hpp
    common/Triangle.cpp
    common/Triangle.hpp
	common/Texture.cpp
    common/Texture.hpp
	common/TextureShader.cpp
    common/TextureShader.hpp
	common/Quad.cpp
    common/Quad.hpp
    filters/Filters.cpp
    filters/Filters.hpp
    transforms/Transforms.cpp
    transforms/Transforms.hpp
    Webcam/webcamQuad.cpp
)
target_link_libraries(Webcam
    ${ALL_LIBS}
)

# --------------------------------------------------------------------------
# Source grouping for IDE organization
# --------------------------------------------------------------------------
SOURCE_GROUP(common REGULAR_EXPRESSION ".*/common/.*")
SOURCE_GROUP(shaders REGULAR_EXPRESSION ".*/.*shader$")
