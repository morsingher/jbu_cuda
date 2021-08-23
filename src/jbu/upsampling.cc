#include "upsampling.h"

bool JointBilateralUpsampling(const std::string& path_lr, const std::string& path_hr, const std::string& out_folder) {
    
    // Read reference depth

    cv::Mat_<float> data_lr;
    if (!ReadColmapMat(path_lr, data_lr)) {
        std::cout << "Failed to load low resolution input: " << path_lr << std::endl;
        return false;
    }
    SaveImage(data_lr, out_folder + "input_viz.jpg");

    // Read reference image

    cv::Mat_<float> data_hr;
    if (!GetFloatImage(path_hr, data_hr)) {
        std::cout << "Failed to load high resolution input: " << path_hr << std::endl;
    }

    // Run upsampling

    JBU jbu(data_lr, data_hr);

    auto start = std::chrono::high_resolution_clock::now();

    jbu.RunCuda();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CUDA elapsed time: " << duration.count() / 1e9 << " s" << std::endl; 

    // Save results

    const cv::Mat_<float> res = jbu.GetResult();
    WriteColmapMat(out_folder + "output.dmb", res);
    SaveImage(res, out_folder + "output_viz.jpg");

    // OpenCV comparison

    cv::Mat_<float> opencv_res;

    start = std::chrono::high_resolution_clock::now();

    cv::resize(data_lr, opencv_res, cv::Size(data_hr.cols, data_hr.rows), cv::INTER_LINEAR);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "OpenCV elapsed time: " << duration.count() / 1e9 << " s" << std::endl; 

    SaveImage(opencv_res, out_folder + "output_opencv.jpg");

    return true;
}

JBU::JBU(const cv::Mat& data_lr, const cv::Mat& data_hr) {
    
    // Set JBU parameters

    params.height_lr = data_lr.rows;
    params.width_lr = data_lr.cols;
    params.height_hr = data_hr.rows;
    params.width_hr = data_hr.cols;
    params.radius = std::max(data_hr.rows / data_lr.rows, data_hr.cols / data_hr.cols);

    // Move data to texture memory

    std::vector<cv::Mat_<float>> data_vec = { data_hr, data_lr };

    for (int i = 0; i < 2; i++) 
    {
        const int rows = data_vec[i].rows;
        const int cols = data_vec[i].cols;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0,  0, cudaChannelFormatKindFloat);
        checkCudaErrors(cudaMallocArray(&array_dev[i], &channelDesc, cols, rows));
        checkCudaErrors(cudaMemcpy2DToArray(array_dev[i], 0, 0, data_vec[i].ptr<float>(), data_vec[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array_dev[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        checkCudaErrors(cudaCreateTextureObject(&(tex_host.data[i]), &resDesc, &texDesc, NULL));
    }

    // Prepare CUDA memory

    data_host = reinterpret_cast<float*>(malloc(sizeof(float) * data_hr.rows * data_hr.cols));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&data_dev), sizeof(float) * data_hr.rows * data_hr.cols));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&tex_dev), sizeof(TextureObj)));
    checkCudaErrors(cudaMemcpy(tex_dev, &tex_host, sizeof(TextureObj), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
}

JBU::~JBU() {

    free(data_host);
    checkCudaErrors(cudaFree(data_dev));
    checkCudaErrors(cudaFree(tex_dev));

    for (int i = 0; i < 2; i++) {
        checkCudaErrors(cudaDestroyTextureObject(tex_host.data[i]));
        checkCudaErrors(cudaFreeArray(array_dev[i]));
    }
}

cv::Mat_<float> JBU::GetResult() {

    cv::Mat_<float> res = cv::Mat::zeros(params.height_hr, params.width_hr, CV_32FC1);
    
    for (int i = 0; i < res.cols; i++) {
        for (int j = 0; j < res.rows; j++) {
            res(j, i) = data_host[j * res.cols + i];
        }
    }
    
    return res;
}
