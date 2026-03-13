#pragma once
#include "../Configure.h"
#include <opencv2/imgproc/imgproc_c.h>
#include "../Feature/ImageData.h"


void thin(Mat srcImage, Mat& dst, double kernalSizeTimes);
void thinTest(Mat srcImage, Mat& dst, double kernalSizeTimes);
