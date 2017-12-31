// Header file for enhancement method

#ifndef _FPENHANCEMENT_H
#define _FPENHANCEMENT_H

#include "commonfiles.h"
#include "checksize.h"
#include "normalizer.h"
#include "ridgeorient.h"
#include "ridgefilter.h"

class FPEnhancement
{
public:
	FPEnhancement();
	cv::Mat run(cv::Mat& inputImage);
	cv::Mat postProcessingFilter(cv::Mat &inputImage);
	cv::Mat getNormalizedImage();
	cv::Mat getOrientationImage();

private:
	CheckSize sizeChecker; // for checking the input image size
	Normalizer normalizer; //for calculating normalized image
	RidgeOrient ridgeOrient; //for calculating orientation field
	RidgeFilter ridgeFilter; //for filtering ridges

	cv::Mat enhancedImage;
	cv::Mat normalizedImage;
	cv::Mat orientationImage;

	int windowSize;
	double kx, ky;
	double thresh;
	double blocksigma;
	double gradientsigma;
	double orientsmoothsigma;
};


#endif
