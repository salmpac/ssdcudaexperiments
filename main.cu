#include "cudaproj.h"

void resize_some_images() {
    string source_filename = "image.tiff";
    string result_filename = "result.tiff";
    image_info img = get_raster(source_filename.c_str());
    image_info img_new = {{}, 500, 500};

    for (size_t i = 0; i < img.bands.size(); ++i) {
        img_new.bands.push_back(new utype[img_new.width * img_new.height]);
        resize_matrix_bilinear(img.bands[i], img.height, img.width, img_new.bands[i], img_new.height, img_new.width);
    }
    create_raster(result_filename.c_str(), img_new);

    for (int i = 0; i < img.bands.size(); ++i) {
        delete[] img.bands[i];
        delete[] img_new.bands[i];
    }
}

int main() {
    GDALAllRegister();
    resize_some_images();
    return 0;
}
