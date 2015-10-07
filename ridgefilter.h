//Header file for final enhancement operation

#ifndef RIDGEFILTER_H
#define RIDGEFILTER_H

#include "commonfiles.h"

class RidgeFilter
{
public:
	cv::Mat run(cv::Mat inputImage, cv::Mat orientationImage, cv::Mat frequency, double kx, double ky);

private:
	int angleInc;
	double pi = 3.14159265;
	void meshgrid(int sze);
	cv::Mat reffilter;
	cv::Mat enhancedImage;
	cv::Mat meshX, meshY;
	cv::vector<cv::Mat> filter;
	cv::Mat gaborFiltering(cv::Mat inputImage, double frequency, cv::Mat orientationImage, cv::Mat enhancedImage);
};

#endif
