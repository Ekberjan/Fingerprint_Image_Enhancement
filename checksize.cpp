#include "checksize.h"

cv::Mat CheckSize::check(cv::Mat& inputImage)
{
	expectedRow = 300;
	expectedCol = 300;

	int inputRow = inputImage.rows;
	int inputCol = inputImage.cols;

	if (inputRow >= expectedRow && inputCol >= expectedCol)
	{
		return inputImage;
	}

	// Pad row with zeros
	if (inputRow < expectedRow)
	{
		cv::copyMakeBorder(inputImage, result, 0, expectedRow - inputRow, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	}

	// Pad col with zeros
	if (inputCol < expectedCol)
	{
		cv::copyMakeBorder(inputImage, result, 0, 0, 0, expectedCol - inputCol, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	}

	return result;
}
