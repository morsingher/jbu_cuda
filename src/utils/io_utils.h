#ifndef JBU_IO_UTILS_H
#define JBU_IO_UTILS_H

#include "common.h"

void SaveImage(const cv::Mat_<float>& data, const std::string& filename);
bool ReadColmapMat(const std::string& filename, cv::Mat& mat);
bool WriteColmapMat(const std::string& filename, const cv::Mat& mat);
bool GetFloatImage(const std::string& filename, cv::Mat& mat);

#endif // JBU_IO_UTILS_H