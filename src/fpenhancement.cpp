// Author: Ekberjan Derman
// Contributor : Baptiste Amato, Julien Jerphanion
// Emails:
//    ekberjanderman@gmail.com
//    baptiste.amato@psycle.io
//    git@jjerphan.xyz

#include "fpenhancement.h"

// see : https://docs.opencv.org/3.4/df/d4e/group__imgproc__c.html
#define CV_RGB2GRAY 7

using namespace cv;


/*
	Perform Gabor filter based image enhancement using orientation field and frequency
	detailed information can be found in Anil Jain's paper
*/
Mat FPEnhancement::extractFingerPrints(Mat &inputImage) {
	// Perform median blurring to smooth the image
	Mat blurredImage;
	medianBlur(inputImage, blurredImage, 3);

	// Check whether the input image is grayscale.
	// If not, convert it to grayscale.
	if (blurredImage.channels() != 1) {
		cvtColor(blurredImage, blurredImage, CV_RGB2GRAY);
	}

	std::cout << "Rows: " << blurredImage.rows << " / Cols: " << blurredImage.cols << std::endl;

	// Perform normalization using the method provided in the paper
	cv::Mat normalizedImage = this->normalize_image(blurredImage, 0, 1);

	std::cout << "Normalization done" << std::endl;

	// Calculate ridge orientation field
	cv::Mat orientationImage = this->orient_ridge(normalizedImage);

	std::cout << "Orientation done" << std::endl;
	
	Mat freq = Mat::ones(normalizedImage.rows, normalizedImage.cols, normalizedImage.type()) * freqValue;

	// Get the final enhanced image and return it as result
	cv::Mat enhancedImage = ridgeFilter.run(normalizedImage, orientationImage, freq, kx, ky);

	std::cout << "Done" << std::endl;

	return enhancedImage;
}

// Normalization function of Anil Jain's algorithm.
cv::Mat FPEnhancement::normalize_image(cv::Mat& im, double reqmean, double reqvar) {
	cv::Mat normalizedImage;
	
	im.convertTo(im, CV_32FC1);

	cv::Scalar mean = cv::mean(im);
	normalizedImage = im - mean[0];

	cv::Scalar normMean = cv::mean(normalizedImage);
	float stdNorm = deviation(normalizedImage, normMean[0]);
	normalizedImage = normalizedImage / stdNorm;
	normalizedImage = reqmean + normalizedImage * cv::sqrt(reqvar);

	return normalizedImage;
}


// Calculate gradient in x- and y-direction of the image
void FPEnhancement::gradient(cv::Mat image, cv::Mat xGradient, cv::Mat yGradient) {
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


// Main function for estimating orientation field of fingerprint ridges.
cv::Mat FPEnhancement::orient_ridge(cv::Mat& im) {

	cv::Mat orientim;
	cv::Mat normalizedImage;
	cv::Mat grad_x, grad_y;
	cv::Mat grad_x_abs, grad_y_abs;
	cv::Mat sin2theta;
	cv::Mat cos2theta;

	int sze = 6 * round(gradientSigma);

	if (sze % 2 == 0) {
		sze++;
	}

	// Define Gaussian kernel
	cv::Mat gaussKernelX = cv::getGaussianKernel(sze, gradientSigma, CV_32FC1);
	cv::Mat gaussKernelY = cv::getGaussianKernel(sze, gradientSigma, CV_32FC1);
	cv::Mat gaussKernel = gaussKernelX * gaussKernelY.t();

	// Peform Gaussian filtering
	cv::Mat fx, fy;
	cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
	cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
	cv::filter2D(gaussKernel, fx, -1, kernelx);
	cv::filter2D(gaussKernel, fy, -1, kernely);

	// Gradient of Gaussian
	gradient(gaussKernel, fx, fy);

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
	int sze2 = 6 * round(blockSigma);

	if (sze2 % 2 == 0) {
		sze2++;
	}

	cv::Mat gaussKernelX2 = cv::getGaussianKernel(sze2, blockSigma, CV_32FC1);
	cv::Mat gaussKernelY2 = cv::getGaussianKernel(sze2, blockSigma, CV_32FC1);
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

	cv::Mat denom;
	G3 = G1 + G2;
	cv::sqrt(G3, denom);

	cv::divide(grad_xy, denom, sin2theta);
	cv::divide(grad_xx - grad_yy, denom, cos2theta);

	int sze3 = 6 * round(orientSmoothSigma);

	if (sze3 % 2 == 0) {
		sze3 += 1;
	}

	cv::Mat gaussKernelX3 = cv::getGaussianKernel(sze3, orientSmoothSigma, CV_32FC1);
	cv::Mat gaussKernelY3 = cv::getGaussianKernel(sze3, orientSmoothSigma, CV_32FC1);
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
			orientim_i[j] = (M_PI + std::atan2(sin2theta_i[j], cos2theta_i[j])) / 2;
		}
	}

	return orientim;
}


// Calculate standard deviation of the image
float FPEnhancement::deviation(cv::Mat& im, float average) {
	float sdev = 0.0;
	float var = 0.0;
	float sd = 0.0;

	for (int i = 0; i < im.rows; i++) {
		for (int j = 0; j < im.cols; j++) {
			float pixel = im.at<float>(i, j);
			float dev = (pixel - average)*(pixel - average);
			sdev = sdev + dev;
		}
	}

	int sze = im.rows * im.cols;
	var = sdev / (sze - 1);
	sd = std::sqrt(var);

	return sd;
}


Mat FPEnhancement::postProcessingFilter(Mat &inputImage) {
	Mat inputImageGrey;
	Mat processedImage;
	Mat filter;

	if (inputImage.channels() != 1) {
		cvtColor(inputImage, inputImageGrey, CV_RGB2GRAY);
	}  else {
		inputImageGrey = inputImage.clone();
	}

	// Blurring the image several times with a kernel 3x3
	// to have smooth surfaces
	for (int j = 0; j < blurringTimes; j++) {
		blur(inputImageGrey, inputImageGrey, Size(3, 3));
	}

	// Canny detector to catch the edges
	Canny(inputImageGrey, filter, cannyLowThreshold, cannyLowThreshold * cannyRatio, kernelSize);

	// Use Canny's output as a mask
	processedImage = Scalar::all(0);
	inputImageGrey.copyTo(processedImage, filter);

	Mat element = getStructuringElement(dilationType,
										Size(2 * dilationSize + 1, 2 * dilationSize + 1),
										Point(dilationSize, dilationSize));

	// Dilate the image to get the contour of the finger
	dilate(processedImage, processedImage, element);

	// Fill the image from the middle to the edge.
	floodFill(processedImage, cv::Point(filter.cols / 2, filter.rows / 2), Scalar(255));

	return processedImage;
}