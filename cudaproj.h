#pragma once

#include <gdal_priv.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;
using utype = unsigned char;
#define GDT_Type GDT_Byte

const bool debug_info = false;

struct image_info {
    vector<utype*> bands;
    size_t width;
    size_t height;
};

void create_raster(const char* filename, image_info image);
image_info get_raster(const char* filename);

void resize_matrix_nearest(utype* matrix, size_t height, size_t width, utype* result_matrix, size_t desired_height, size_t desired_width);
void resize_matrix_bilinear(utype* matrix, size_t height, size_t width, utype* result_matrix, size_t desired_height, size_t desired_width);
