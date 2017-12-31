// Function for image normalization using the method.
// The method is based on Anil Jain's algorithm.
//
// Author: Ekberjan Derman
// Contributor : Baptiste Amato, Julien Jerphanion
// Emails:
//    ekberjanderman@gmail.com
//    baptiste.amato@gmail.com
//    julien.jerphanion@protonmail.com
//
// Last update : 12.2017

#include "normalizer.h"

cv::Mat Normalizer::run(cv::Mat& im, double reqmean, double reqvar)
{
	im.convertTo(im, CV_32FC1);
	cv::minMaxLoc(im, &min, &max, &min_loc, &max_loc);

	mean = cv::mean(im);
	normalizedImage = im - mean[0];

	cv::Scalar normMean = cv::mean(normalizedImage);
	stdNorm = deviation(normalizedImage, normMean[0]);
	normalizedImage = normalizedImage / stdNorm;
	normalizedImage = reqmean + normalizedImage * cv::sqrt(reqvar);

	cv::minMaxLoc(normalizedImage, &minNorm, &maxNorm);
	return normalizedImage;
}

// Calculate standard deviation of the image
float Normalizer::deviation(cv::Mat& im, float ave)
{
	float sdev = 0.0;
	float var = 0.0;
	float sd = 0.0;

	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			float pixel = im.at<float>(i, j);
			float dev = (pixel - ave)*(pixel - ave);
			sdev = sdev + dev;
		}
	}

	int sze = im.rows * im.cols;
	var = sdev / (sze - 1);
	sd = std::sqrt(var);

	return sd;
}
