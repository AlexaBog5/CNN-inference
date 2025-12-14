cmake_minimum_required(VERSION 3.20)

project(lenet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lenet lenet.cpp)

target_compile_definitions(lenet PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    # how many images to process
    NUM_IMAGES=10
    # paths
    WEIGHTS_LOCAL_PATH="data/data-mnist-lenet.raw"
    MNIST_LOCAL_PATH="data/data-mnist-t10k-images-idx3-ubyte"
    # 1 - save images, 0 - do not save images
    TO_SAVE_IMAGES=1
    # where to save output images (the directory will be created if it does not exist)
    OUTPUT_LOCAL_PATH="data/output/"
)

# Warnings
if (MSVC)
    target_compile_options(lenet PRIVATE /W4)
else()
    target_compile_options(lenet PRIVATE -Wall -Wextra)
endif()
