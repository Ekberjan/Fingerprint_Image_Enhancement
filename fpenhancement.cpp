//Fingerprint enhancement main function
//The method is based on Anil Jain's algorithm
//Author: Ekberjan Derman
//email: ekberjanderman@gmail.com
//08.2015

#include "stdafx.h"
#include "fpenhancement.h"

//Constructor
//The relevant params are set here
FPEnhancement::FPEnhancement()
{
	windowSize = 30;
	thresh = 0.1;
	orientsmoothsigma = 5.0;
	blocksigma = 5.0;
	gradientsigma = 1.0;
	kx = 0.4;
	ky = 0.4;
}

//Peform Gabor filter based image enhancement
//using orientation field and frequency
//Detained information can be found in Anil Jain's 
//paper
cv::Mat FPEnhancement::run(cv::Mat& inputImage)
{
	//Make sure the input image is valid
	if (!inputImage.data)
	{
		std::cerr << "The provided input image is invalid. Please check it again. " << std::endl;
		return enhancedImage;
	}

	//Check whether the input image size is at least of 300x300	
	cv::Mat paddedImage = sizeChecker.check(inputImage);
	
	//Perform median blurring to smooth the image
	cv::Mat blurredImage;
	cv::medianBlur(paddedImage, blurredImage, 3);
	
	//Check whether the input image is grayscale. 
	//If not, convert it to grayscale.
	if (blurredImage.channels() != 1)
	{
		cv::cvtColor(blurredImage, blurredImage, CV_RGB2GRAY);
	}

	//Perform normalization using the method provided in the paper
	normalizedImage = normalizer.run(blurredImage, 0, 1);	
	//Calculate ridge orientation field
	orientationImage = ridgeOrient.run(normalizedImage, gradientsigma, blocksigma, orientsmoothsigma);
	//The frequency is set to 0.11 experimentally. You can change this value
	//or use a dynamic calculation instead.
	cv::Mat freq = cv::Mat::ones(normalizedImage.rows, normalizedImage.cols, normalizedImage.type());
	freq *= 0.11;

	//Get the final enhanced image and return it as result
	enhancedImage = ridgeFilter.run(normalizedImage, orientationImage, freq, kx, ky);

	return enhancedImage;
}

cv::Mat FPEnhancement::getNormalizedImage()
{
	return normalizedImage;
}

cv::Mat FPEnhancement::getOrientationImage()
{
	return orientationImage;
}
