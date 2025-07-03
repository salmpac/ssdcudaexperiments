#include "cudaproj.h"

void create_raster(const char* filename, image_info image) {
    vector<utype*> bands = image.bands;
    size_t width = image.width;
    size_t height = image.height;
    size_t bands_count = bands.size();
    GDALDatasetH dataset = GDALCreate(GDALGetDriverByName("GTiff"), filename, width, height, bands_count, GDT_Type, NULL);
    if (dataset == NULL) {
        cout << "Error while creating file\n";
        return;
    }
    // bands are really numbered from 1
    // pretty dumb
    for (size_t i = 1; i <= bands_count; ++i) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, i);
        GDALRasterIO(band, GF_Write, 0, 0, width, height, image.bands[i - 1], width, height, GDT_Type, 0, 0);
    }
    GDALClose(dataset);
}

void print_dataset_info(GDALDataset* dataset) {
    if (dataset == nullptr) {
        std::cerr << "Error: dataset pointer is nullptr." << std::endl;
        return;
    }

    // Get the number of raster bands
    int bandCount = dataset->GetRasterCount();
    std::cout << "Number of bands: " << bandCount << std::endl;

    // Get the size of the image
    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    std::cout << "Image size: " << width << " x " << height << std::endl;

    // Get information about each band
    for (int i = 1; i <= bandCount; ++i) {
        GDALRasterBand* band = dataset->GetRasterBand(i);
        if (band) {
            std::cout << "Band " << i << ":" << std::endl;
            std::cout << "  Data type: " << GDALGetDataTypeName(band->GetRasterDataType()) << std::endl;
            std::cout << "  Minimum value: " << band->GetMinimum(nullptr) << std::endl;
            std::cout << "  Maximum value: " << band->GetMaximum(nullptr) << std::endl;
        }
    }

    // Get geotransform information
    double geoTransform[6];
    if (dataset->GetGeoTransform(geoTransform) == CE_None) {
        std::cout << "Geotransform:" << std::endl;
        std::cout << "  X: " << geoTransform[0] << std::endl;
        std::cout << "  Y: " << geoTransform[3] << std::endl;
        std::cout << "  Pixel width: " << geoTransform[1] << std::endl;
        std::cout << "  Pixel height: " << geoTransform[5] << std::endl;
    } else {
        std::cout << "Geotransform not available." << std::endl;
    }
}

image_info get_raster(const char* filename) {
    GDALDataset* dataset = (GDALDataset*) GDALOpen(filename, GA_ReadOnly);
    if (dataset == nullptr) {
        cout << "Error while creating file\n";
        return {};
    }
    print_dataset_info(dataset);

    image_info imgInfo;
    imgInfo.width = dataset->GetRasterXSize();
    imgInfo.height = dataset->GetRasterYSize();
    int bandCount = dataset->GetRasterCount();

    imgInfo.bands.resize(bandCount);
    for (int i = 0; i < bandCount; ++i) {
        GDALRasterBand* band = dataset->GetRasterBand(i + 1);
        imgInfo.bands[i] = new utype[imgInfo.width * imgInfo.height * sizeof(utype)];
        band->RasterIO(GF_Read, 0, 0, imgInfo.width, imgInfo.height, imgInfo.bands[i], imgInfo.width, imgInfo.height, GDT_Type, 0, 0);
    }

    GDALClose(dataset);
    return imgInfo;
}
