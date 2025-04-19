#include <gdal_priv.h>
#include <vector>

using namespace std;
using uchar = unsigned char;

struct image_info {
    vector<uchar*> bands;
    size_t width;
    size_t height;
};


void create_raster(const char* filename, image_info image);
image_info get_raster(const char* filename);

void resize_matrix(uchar* matrix, size_t height, size_t width, uchar* result_matrix, size_t desired_height, size_t desired_width);