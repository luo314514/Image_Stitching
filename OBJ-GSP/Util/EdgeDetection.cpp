#include "EdgeDetection.h"

void edgeDetection(cv::Mat& _img, cv::Mat& _edge, double _threshold) {
    // 【核心修复】：无论如何，先给后续流程一张合法的灰度图
    if (_img.channels() == 3) {
        cv::cvtColor(_img, _edge, cv::COLOR_BGR2GRAY);
    } else {
        _img.copyTo(_edge);
    }
    // 此时 _edge 已经变成 CV_8UC1 类型，满足所有 OpenCV 断言
    return; 
}