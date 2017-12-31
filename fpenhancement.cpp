// Fingerprint enhancement main function.
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

#include "fpenhancement.h"


using namespace cv;

//Constructor
//The relevant params are set here
FPEnhancement::FPEnhancement() {
	windowSize = 38;
	thresh = 0.000000001;
	orientsmoothsigma = 5.0;
	blocksigma = 5.0;
	gradientsigma = 1.0;
	kx = 0.8;
	ky = 0.8;
}

// Peform Gabor filter based image enhancement
// using orientation field and frequency
// Detained information can be found in Anil Jain's
// paper
Mat FPEnhancement::run(Mat &inputImage) {
	//Check whether the input image size is at least of 300x300
	// Mat paddedImage = sizeChecker.check(inputImage);

	//Perform median blurring to smooth the image
	Mat blurredImage;
	// medianBlur(paddedImage, blurredImage, 3);
	medianBlur(inputImage, blurredImage, 3);

	// Check whether the input image is grayscale.
	//I f not, convert it to grayscale.
	if (blurredImage.channels() != 1) {
		cvtColor(blurredImage, blurredImage, CV_RGB2GRAY);
	}

	std::cout << "Rows: " << blurredImage.rows << " / Cols: " << blurredImage.cols << std::endl;

	// Perform normalization using the method provided in the paper
	normalizedImage = normalizer.run(blurredImage, 0, 1);
	// Calculate ridge orientation field
	orientationImage = ridgeOrient.run(normalizedImage, gradientsigma, blocksigma,
										 orientsmoothsigma);
	// The frequency is set to 0.11 experimentally. You can change this value
	// or use a dynamic calculation instead.
	Mat freq = Mat::ones(normalizedImage.rows, normalizedImage.cols, normalizedImage.type());
	freq *= 0.11;

	// Get the final enhanced image and return it as result
	enhancedImage = ridgeFilter.run(normalizedImage, orientationImage, freq, kx, ky);

	return enhancedImage;
}

Mat FPEnhancement::postProcessingFilter(Mat &inputImage) {
	int lowThreshold = 10;
	int ratio = 3;
	int kernel_size = 3;
	Mat inputImageGrey;
	Mat processedImage;
	Mat filter;

	if (inputImage.channels() != 1) {
		cvtColor(inputImage, inputImageGrey, CV_RGB2GRAY);
	}  else {
		inputImageGrey = inputImage.clone();
	}

	/// Blurring the image several times with a kernel 3x3
	/// to have smooth surfaces
	for (int j = 0; j < 30; j++) {
		blur(inputImageGrey, inputImageGrey, Size(3, 3));
	}

	/// Canny detector to catch the edges
	Canny(inputImageGrey, filter, lowThreshold, lowThreshold * ratio, kernel_size);

	/// Using Canny's output as a mask
	processedImage = Scalar::all(0);
	inputImageGrey.copyTo(processedImage, filter);

	int dilationSize = 10;
	int dilationType = 1;
	Mat element = getStructuringElement(dilationType,
										Size(2 * dilationSize + 1, 2 * dilationSize + 1),
										Point(dilationSize, dilationSize));

	/// Dilating the image to get the contour of the finger
	dilate(processedImage, processedImage, element);

	// Filling the image from the middle to the edge.
	floodFill(processedImage, cv::Point(filter.cols / 2, filter.rows / 2), Scalar(255));

	return processedImage;
}

Mat FPEnhancement::getNormalizedImage() {
	return normalizedImage;
}

Mat FPEnhancement::getOrientationImage() {
	return orientationImage;
}
