#include <cuda_runtime.h>

#include "cudaproj.h"

using namespace std;
using uchar = unsigned char;
//matrix[x][y] == matrix[x * width + y]

__global__ void copy_nearest_pixel(uchar* source_matrix, size_t old_height, size_t old_width, uchar* result_matrix, size_t height, size_t width) {
    size_t current_x = blockIdx.x;
    size_t pixels_per_thread = (width + blockDim.x - 1) / blockDim.x;
    size_t start_y = pixels_per_thread * threadIdx.x;
    size_t end_y = start_y + pixels_per_thread;
    if (end_y > width) {
        end_y = width;
    }
    for (size_t current_y = start_y; current_y < end_y; ++current_y) {
        // i need to find nearest pixel in old matrix
        size_t old_x = current_x * old_height / height;
        size_t old_y = current_y * old_width / width;
        result_matrix[current_x * width + current_y] = source_matrix[old_x * old_width + old_y];
    }
}

void resize_matrix(uchar* matrix, size_t height, size_t width, uchar* result_matrix, size_t desired_height, size_t desired_width) {
    uchar* source_matrix_gpu;
    cudaMalloc(&source_matrix_gpu, height * width);
    cudaMemcpy(source_matrix_gpu, matrix, height * width, cudaMemcpyHostToDevice);

    uchar* result_matrix_gpu;
    cudaMalloc(&result_matrix_gpu, desired_height * desired_width);

    size_t threads_per_block = min((size_t)128, desired_width);
    copy_nearest_pixel<<<desired_height, threads_per_block>>>(source_matrix_gpu, height, width, result_matrix_gpu, desired_height, desired_width);
    cudaMemcpy(result_matrix, result_matrix_gpu, desired_height * desired_width, cudaMemcpyDeviceToHost);

    cudaFree(source_matrix_gpu);
    cudaFree(result_matrix_gpu);
}