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

// Constructor
FPEnhancement::FPEnhancement():
 	windowSize(38),
	thresh(0.000000001),
	orientSmoothSigma(5.0),
	blockSigma(5.0),
	gradientSigma(1.0),
	kx(0.8),
	ky(0.8),
	// The frequency is set to 0.11 experimentally. You can change this value
	// or use a dynamic calculation instead.
	freqValue(0.11)
{}

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
	normalizedImage = this->normalize_image(blurredImage, 0, 1);

	// Calculate ridge orientation field
	orientationImage = ridgeOrient.run(normalizedImage, gradientSigma, blockSigma,
										 orientSmoothSigma);
	
	Mat freq = Mat::ones(normalizedImage.rows, normalizedImage.cols, normalizedImage.type()) * freqValue;

	// Get the final enhanced image and return it as result
	enhancedImage = ridgeFilter.run(normalizedImage, orientationImage, freq, kx, ky);

	return enhancedImage;
}

// Normalization function of Anil Jain's algorithm.
cv::Mat FPEnhancement::normalize_image(cv::Mat& im, double reqmean, double reqvar) {
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
	int cannyLowThreshold = 10;
	int cannyRatio = 3;
	int kernelSize = 3;
	int blurringTimes = 30;
	int dilationSize = 10;
	int dilationType = 1;
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