#ifndef RIDGEORIENT_H
#define RIDGEORIENT_H

#include "commonfiles.h"

class RidgeOrient {
public:
	RidgeOrient();
	cv::Mat run(cv::Mat im, double gradientsigma, double blocksigma, double orientsmoothsigma);

private:
	// params for Sobel filter
	int scale;
	int delta;
	int ddepth;

	double pi;
	double M, VAR, M0, VAR0;

	cv::Mat orientim;
	cv::Mat normalizedImage;
	cv::Mat grad_x, grad_y;
	cv::Mat grad_x_abs, grad_y_abs;
	cv::Mat denom;
	cv::Mat sin2theta;
	cv::Mat cos2theta;

	void gradient(cv::Mat image, cv::Mat xGradient, cv::Mat yGradient);


};

#endif
