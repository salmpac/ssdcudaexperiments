#include "cudaproj.h"

//matrix[x][y] == matrix[x * width + y]

__global__ void copy_nearest_pixel(utype* source_matrix, size_t old_height, size_t old_width, utype* result_matrix, size_t height, size_t width) {
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

void resize_matrix_nearest(utype* matrix, size_t height, size_t width, utype* result_matrix, size_t desired_height, size_t desired_width) {
    chrono::milliseconds start_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    utype* source_matrix_gpu;
    cudaMalloc(&source_matrix_gpu, height * width * sizeof(utype));
    cudaMemcpy(source_matrix_gpu, matrix, height * width * sizeof(utype), cudaMemcpyHostToDevice);

    utype* result_matrix_gpu;
    cudaMalloc(&result_matrix_gpu, desired_height * desired_width * sizeof(utype));

    chrono::milliseconds start_time_gpu = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    size_t threads_per_block = min((size_t)128, desired_width);
    copy_nearest_pixel<<<desired_height, threads_per_block>>>(source_matrix_gpu, height, width, result_matrix_gpu, desired_height, desired_width);
    chrono::milliseconds end_time_gpu = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    cout << "gpu part nearest value interpolation time: " << (end_time_gpu - start_time_gpu).count() << '\n';

    cudaMemcpy(result_matrix, result_matrix_gpu, desired_height * desired_width * sizeof(utype), cudaMemcpyDeviceToHost);

    cudaFree(source_matrix_gpu);
    cudaFree(result_matrix_gpu);
    chrono::milliseconds end_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    cout << "whole nearest value interpolation time: " << (end_time - start_time).count() << '\n';
}

// let us now function at 4 points
// f(x1, y1) = q11
// f(x2, y1) = q21
// f(x1, y2) = q12
// f(x2, y2) = q22
// x1 < x < x2 and y1 < y < y2
// and we want to compute f(x, y)
// f(x, y1) = (x2 - x)/(x2 - x1) * q11 + (x - x1)/(x2 - x1) * q21
// f(x, y2) = (x2 - x)/(x2 - x1) * q12 + (x - x1)/(x2 - x1) * q22
// f(x, y) = (y2 - y)/(y2 - y1) * f(x, y1) + (y - y1)/(y2 - y1) * f(x, y2)

__global__ void bilinear(utype* source_matrix, size_t old_height, size_t old_width, utype* result_matrix, size_t height, size_t width) {
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
        if (old_x + 1 == old_height || old_y + 1 == old_width) {
            // edge point
            // result_matrix[current_x * width + current_y] = source_matrix[old_x * old_width + old_y];
            result_matrix[current_x * width + current_y] = 0;
            continue;
        }
        double x1 = old_x;
        double y1 = old_y;
        double x2 = x1 + 1;
        double y2 = y1 + 1;
        double x = (double)current_x * old_height / (double)height;
        double y = (double)current_y * old_width / (double)width;
        double q11 = source_matrix[old_x * old_width + old_y];
        double q12 = source_matrix[old_x * old_width + old_y + 1];
        double q21 = source_matrix[(old_x + 1) * old_width + old_y];
        double q22 = source_matrix[(old_x + 1) * old_width + old_y + 1];
        double f_x_y1 = (x2 - x)/(x2 - x1) * q11 + (x - x1)/(x2 - x1) * q21;
        double f_x_y2 = (x2 - x)/(x2 - x1) * q12 + (x - x1)/(x2 - x1) * q22;
        result_matrix[current_x * width + current_y] = (y2 - y)/(y2 - y1) * f_x_y1 + (y - y1)/(y2 - y1) * f_x_y2;
    }
}

void resize_matrix_bilinear(utype* matrix, size_t height, size_t width, utype* result_matrix, size_t desired_height, size_t desired_width) {
    chrono::milliseconds start_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    utype* source_matrix_gpu;
    cudaMalloc(&source_matrix_gpu, height * width * sizeof(utype));
    cudaMemcpy(source_matrix_gpu, matrix, height * width * sizeof(utype), cudaMemcpyHostToDevice);

    utype* result_matrix_gpu;
    cudaMalloc(&result_matrix_gpu, desired_height * desired_width * sizeof(utype));

    chrono::milliseconds start_time_gpu = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    size_t threads_per_block = min((size_t)128, desired_width);
    bilinear<<<desired_height, threads_per_block>>>(source_matrix_gpu, height, width, result_matrix_gpu, desired_height, desired_width);
    chrono::milliseconds end_time_gpu = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    cout << "gpu part bilinear interpolation time: " << (end_time_gpu - start_time_gpu).count() << '\n';

    cudaMemcpy(result_matrix, result_matrix_gpu, desired_height * desired_width * sizeof(utype), cudaMemcpyDeviceToHost);

    cudaFree(source_matrix_gpu);
    cudaFree(result_matrix_gpu);
    chrono::milliseconds end_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now().time_since_epoch());
    cout << "whole bilinear interpolation time: " << (end_time - start_time).count() << '\n';
}
