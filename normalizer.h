// Header file for normalization operation

#ifndef _NORMALIZER_H
#define _NORMALIZER_H

#include "commonfiles.h"

class Normalizer {
public:
	cv::Mat run(cv::Mat& im, double reqmean, double reqvar);

private:
	double min;
	double max;
	double minNorm;
	double maxNorm;
	double stdNorm;

	cv::Mat normalizedImage;
	cv::Scalar mean, meanNorm;
	cv::Point min_loc, max_loc;

	float deviation(cv::Mat& im, float ave);

};

#endif
