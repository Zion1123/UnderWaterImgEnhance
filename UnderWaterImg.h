#pragma once
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include<vector>
#include <opencv2\opencv.hpp>


class ImageProcess{
	public:
		cv::Mat DehazedImg(cv::Mat input);
		cv::Mat SPDF(cv::Mat src);
};