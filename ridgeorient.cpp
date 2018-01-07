// Main function for estimating orientation field of fingerprint ridges.
//
// Author: Ekberjan Derman
// Contributor : Baptiste Amato, Julien Jerphanion
// Emails:
//    ekberjanderman@gmail.com
//    baptiste.amato@psycle.io
//    julien.jerphanion@protonmail.com
//
// Last update : 12.2017

#include "ridgeorient.h"

RidgeOrient::RidgeOrient() {
	// params for Sobel opeation
	scale = 1;
	delta = 0;
	ddepth = CV_32FC1;
	pi = M_PI; // 3.14159265;
}

cv::Mat
RidgeOrient::run(cv::Mat im, double gradientsigma, double blocksigma, double orientsmoothsigma) {
	int sze = 6 * round(gradientsigma);

	if (sze % 2 == 0) {
		sze++;
	}

	// Define Gaussian kernel
	cv::Mat gaussKernelX = cv::getGaussianKernel(sze, gradientsigma, CV_32FC1);
	cv::Mat gaussKernelY = cv::getGaussianKernel(sze, gradientsigma, CV_32FC1);
	cv::Mat gaussKernel = gaussKernelX * gaussKernelY.t();

	// Peform Gaussian filtering
	cv::Mat fx, fy;
	cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
	cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
	cv::filter2D(gaussKernel, fx, -1, kernelx);
	cv::filter2D(gaussKernel, fy, -1, kernely);

	gradient(gaussKernel, fx, fy); // Gradient of Gaussian

	grad_x.convertTo(grad_x, CV_32FC1);
	grad_y.convertTo(grad_y, CV_32FC1);

	cv::filter2D(im, grad_x, -1, fx, cv::Point(-1, -1), 0,
				 cv::BORDER_DEFAULT); // Gradient of the image in x
	cv::filter2D(im, grad_y, -1, fy, cv::Point(-1, -1), 0,
				 cv::BORDER_DEFAULT); // Gradient of the image in y

	cv::Mat grad_xx, grad_xy, grad_yy;
	cv::multiply(grad_x, grad_x, grad_xx);
	cv::multiply(grad_x, grad_y, grad_xy);
	cv::multiply(grad_y, grad_y, grad_yy);

	// Now smooth the covariance data to perform a weighted summation of the data
	int sze2 = 6 * round(blocksigma);

	if (sze2 % 2 == 0) {
		sze2++;
	}

	cv::Mat gaussKernelX2 = cv::getGaussianKernel(sze2, blocksigma, CV_32FC1);
	cv::Mat gaussKernelY2 = cv::getGaussianKernel(sze2, blocksigma, CV_32FC1);
	cv::Mat gaussKernel2 = gaussKernelX2 * gaussKernelY2.t();

	cv::filter2D(grad_xx, grad_xx, -1, gaussKernel2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(grad_xy, grad_xy, -1, gaussKernel2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(grad_yy, grad_yy, -1, gaussKernel2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	grad_xy *= 2;

	// Analytic solution of principal direction
	cv::Mat G1, G2, G3;
	cv::multiply(grad_xy, grad_xy, G1);
	G2 = grad_xx - grad_yy;
	cv::multiply(G2, G2, G2);

	G3 = G1 + G2;
	cv::sqrt(G3, denom);

	cv::divide(grad_xy, denom, sin2theta);
	cv::divide(grad_xx - grad_yy, denom, cos2theta);


	int sze3 = 6 * round(orientsmoothsigma);

	if (sze3 % 2 == 0) {
		sze3 += 1;
	}

	cv::Mat gaussKernelX3 = cv::getGaussianKernel(sze3, orientsmoothsigma, CV_32FC1);
	cv::Mat gaussKernelY3 = cv::getGaussianKernel(sze3, orientsmoothsigma, CV_32FC1);
	cv::Mat gaussKernel3 = gaussKernelX3 * gaussKernelY3.t();

	cv::filter2D(cos2theta, cos2theta, -1, gaussKernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	cv::filter2D(sin2theta, sin2theta, -1, gaussKernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

	sin2theta.convertTo(sin2theta, ddepth);
	cos2theta.convertTo(cos2theta, ddepth);
	orientim = cv::Mat::zeros(sin2theta.rows, sin2theta.cols, ddepth);

	// Pointer access more effective than Mat.at<T>()
	for (int i = 0; i < sin2theta.rows; i++) {
		const float *sin2theta_i = sin2theta.ptr<float>(i);
		const float *cos2theta_i = cos2theta.ptr<float>(i);
		float *orientim_i = orientim.ptr<float>(i);
		for (int j = 0; j < sin2theta.cols; j++) {
			orientim_i[j] = (pi + std::atan2(sin2theta_i[j], cos2theta_i[j])) / 2;
		}
	}

	return orientim;
}

//Calculate gradient in x- and y-direction of the image
void RidgeOrient::gradient(cv::Mat image, cv::Mat xGradient, cv::Mat yGradient) {
	xGradient = cv::Mat::zeros(image.rows, image.cols, ddepth);
	yGradient = cv::Mat::zeros(image.rows, image.cols, ddepth);

	// Pointer access more effective than Mat.at<T>()
	for (int i = 1; i < image.rows - 1; i++) {
		const float *image_i = image.ptr<float>(i);
		float *xGradient_i = xGradient.ptr<float>(i);
		float *yGradient_i = yGradient.ptr<float>(i);
		for (int j = 1; j < image.cols - 1; j++) {
			float xPixel1 = image_i[j - 1];
			float xPixel2 = image_i[j + 1];

			float yPixel1 = image.at<float>(i - 1, j);
			float yPixel2 = image.at<float>(i + 1, j);

			float xGrad;
			float yGrad;

			if (j == 0) {
				xPixel1 = image_i[j];
				xGrad = xPixel2 - xPixel1;
			} else if (j == image.cols - 1) {
				xPixel2 = image_i[j];
				xGrad = xPixel2 - xPixel1;
			} else {
				xGrad = 0.5 * (xPixel2 - xPixel1);
			}

			if (i == 0) {
				yPixel1 = image_i[j];
				yGrad = yPixel2 - yPixel1;
			} else if (i == image.rows - 1) {
				yPixel2 = image_i[j];
				yGrad = yPixel2 - yPixel1;
			} else {
				yGrad = 0.5 * (yPixel2 - yPixel1);
			}

			xGradient_i[j] = xGrad;
			yGradient_i[j] = yGrad;
		}
	}
}
