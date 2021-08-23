#ifndef JBU_UPSAMPLING_H
#define JBU_UPSAMPLING_H

#include "io_utils.h"
#include "cuda_helper.h"

bool JointBilateralUpsampling(const std::string& path_lr, const std::string& path_hr, const std::string& out_folder);

struct Parameters {
    int height_lr, width_lr;
    int height_hr, width_hr;
    int radius;
};

struct TextureObj {
    cudaTextureObject_t data[2];
};

class JBU {

    Parameters params;
    
    // Host data
    float* data_host;
    TextureObj tex_host;

    // Device data
    float* data_dev;
    cudaArray* array_dev[2];
    TextureObj* tex_dev;

public:

    JBU(const cv::Mat& data_lr, const cv::Mat& data_hr);
    ~JBU();

    cv::Mat_<float> GetResult();
    void RunCuda();
};

#endif