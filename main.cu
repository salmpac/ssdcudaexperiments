#include <iostream>
#include <vector>
#include <string>

#include "cudaproj.h"

using namespace std;
using uchar = unsigned char;

int main() {
    GDALAllRegister();

    string source_filename = "image.tiff";
    string result_filename = "result.tiff";
    image_info img = get_raster(source_filename.c_str());
    image_info img_new = {{}, 700, 400};

    for (size_t i = 0; i < img.bands.size(); ++i) {
        img_new.bands.push_back(new uchar[img_new.width * img_new.height]);
        resize_matrix(img.bands[i], img.height, img.width, img_new.bands[i], img_new.height, img_new.width);
    }
    create_raster(result_filename.c_str(), img_new);

    for (int i = 0; i < img.bands.size(); ++i) {
        delete[] img.bands[i];
        delete[] img_new.bands[i];
    }
}
