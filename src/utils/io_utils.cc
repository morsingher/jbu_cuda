#include "io_utils.h"

void SaveImage(const cv::Mat_<float>& data, const std::string& filename) {

    double min, max;
    cv::minMaxLoc(data, &min, &max);
    cv::Mat_<float> data_viz = cv::Mat::zeros(data.rows, data.cols, CV_32FC1);

    for (int col = 0; col < data.cols; ++col) {
        for (int row = 0; row < data.rows; ++row) {
            data_viz(row, col) = 255.0 * (data(row, col) - min) / (max - min);
        }
    }

    cv::imwrite(filename, data_viz);
}

bool ReadColmapMat(const std::string& filename, cv::Mat& mat) {

    std::fstream text_file(filename, std::ios::in);
    if (!text_file) {
        std::cout << "Failed to open text: " << filename << std::endl;
        return false;
    }

    int rows, cols, channels;
    char dummy;
    text_file >> cols >> dummy >> rows >> dummy >> channels >> dummy;
    std::streampos pos = text_file.tellg();
    text_file.close();

    if (channels == 1) {
        mat = cv::Mat::zeros(rows, cols, CV_32FC1);
    } else {
        mat = cv::Mat::zeros(rows, cols, CV_32FC3);
    }

    std::fstream bin_file(filename, std::ios::in | std::ios::binary);
    if (!bin_file) {
        std::cout << "Failed to open bin: " << filename << std::endl;
        return false;
    }
    bin_file.seekg(pos);
    bin_file.read(reinterpret_cast<char*>(mat.data), sizeof(float) * channels * rows * cols);

    return true;
}

bool WriteColmapMat(const std::string& filename, const cv::Mat& mat) {

    std::fstream text_file(filename, std::ios::out);
    if (!text_file) {
        std::cout << "Failed to open text: " << filename << std::endl;
        return false;
    }

    text_file << mat.cols << "&" << mat.rows << "&" << mat.channels() << "&";
    text_file.close();

    std::fstream bin_file(filename, std::ios::out | std::ios::binary | std::ios::app);
    if (!bin_file) {
        std::cout << "Failed to open bin: " << filename << std::endl;
        return false;
    }

    bin_file.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    bin_file.close();

    return true;
}

bool GetFloatImage(const std::string& filename, cv::Mat& mat) {
    
    cv::Mat_<uint8_t> image_uint = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image_uint.empty()) {
        std::cout << "Failed to read: " << filename << std::endl;
        return false;
    }
    image_uint.convertTo(mat, CV_32FC1);
    return true;
}
