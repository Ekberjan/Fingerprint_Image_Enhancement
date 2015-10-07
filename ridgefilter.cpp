//Main function for performing Gabor filtering
//for enhancement using previously calculated 
//orientation image and frequency. The output
//is final enhanced image. 
//Refer to the paper for detailed description. 
//Author: Ekberjan Derman
//email: ekberjanderman@gmail.com
//08.2015

#include "stdafx.h"
#include "ridgefilter.h"

cv::Mat RidgeFilter::run(cv::Mat inputImage, cv::Mat orientationImage, cv::Mat frequency, double kx, double ky)
{
	angleInc = 3; // fixed angle increment between filter orientations  in degrees.

	inputImage.convertTo(inputImage, CV_32FC1);
	int rows = inputImage.rows;
	int cols = inputImage.cols;

	orientationImage.convertTo(orientationImage, CV_32FC1);

	enhancedImage = cv::Mat::zeros(rows, cols, CV_32FC1);
	cv::vector<int> validr;
	cv::vector<int> validc;

	int ind = 1;
	double freq = frequency.at<float>(1, 1);
	double unfreq = freq;

	cv::Mat freqindex = cv::Mat::ones(100, 1, CV_32FC1);

	double sigmax = (1 / freq) * kx;
	double sigmay = (1 / freq) * ky;

	int szek = round(3 * (std::max(sigmax, sigmay)));
	meshgrid(szek);

	reffilter = cv::Mat::zeros(meshX.rows, meshX.cols, CV_32FC1);

	meshX.convertTo(meshX, CV_32FC1);
	meshY.convertTo(meshY, CV_32FC1);

	for (int i = 0; i < meshX.rows; i++)
	{
		for (int j = 0; j < meshX.cols; j++)
		{
			float pixVal2 = -0.5*(meshX.at<float>(i, j)*meshX.at<float>(i, j) / (sigmax*sigmax) +
				meshY.at<float>(i, j)*meshY.at<float>(i, j) / (sigmay*sigmay));
			float pixVal = std::exp(pixVal2);
			float cosVal = 2 * pi * unfreq * meshX.at<float>(i, j);
			reffilter.at<float>(i, j) = pixVal * cos(cosVal);
		}
	}

	for (int m = 0; m < 180 / angleInc; m++)
	{
		double angle = -(m*angleInc + 90);
		cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point((float)(reffilter.rows / 2.0F), (float)(reffilter.cols / 2.0F)), angle, 1.0);
		cv::Mat rotResult;
		cv::warpAffine(reffilter, rotResult, rot_mat, reffilter.size());
		filter.push_back(rotResult);
	}

	// Find indices of matrix points greater than maxsze from the image boundary
	int maxsze = szek;
	// Convert orientation matrix values from radians to an index value that corresponds
	// to round(degrees/angleInc)
	int maxorientindex = std::round(180 / angleInc);

	cv::Mat orientindex(rows, cols, CV_32FC1);

	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			if (x > maxsze && x < rows - maxsze && y > maxsze && y < cols - maxsze)
			{
				validr.push_back(y);
				validc.push_back(x);
			}

			int orientpix = static_cast<int>(std::round(orientationImage.at<float>(y, x) / pi * 180 / angleInc));

			if (orientpix < 0)
			{
				orientpix += maxorientindex;
			}
			if (orientpix >= maxorientindex)
			{
				orientpix -= maxorientindex;
			}

			orientindex.at<float>(y, x) = orientpix;
		}
	}

	// Finally, do the filtering
	for (int k = 0; k < validr.size(); k++)
	{
		int s = szek;
		int r = validr[k];
		int c = validc[k];
		int filterindex = freqindex.at<float>(std::round(frequency.at<float>(r, c) * 100));

		cv::Rect roi(c - s - 1, r - s - 1, meshX.cols, meshX.rows);
		cv::Mat subim(meshX.rows, meshX.cols, CV_32FC1);

		for (int m = r - s; m < r + s; m++)
		{
			for (int n = c - s; n < c + s; n++)
			{
				int tmpRow = m - r + s;
				int tmpCol = n - c + s;

				if (tmpRow < subim.rows && tmpCol < subim.cols && m < inputImage.rows && n < inputImage.cols)
				{
					subim.at<float>(m - r + s, n - c + s) = inputImage.at<float>(m, n);
				}
			}
		}

		cv::Mat subFilter = filter.at(orientindex.at<float>(r, c));
		cv::Mat mulResult;
		cv::multiply(subim, subFilter, mulResult);

		cv::Scalar resultSum = cv::sum(mulResult);
		float value = resultSum[0];
		if (value > 0)
		{
			enhancedImage.at<float>(r, c) = 255;
		}
	}

	cv::Mat aux = enhancedImage.rowRange(0, rows).colRange(0, szek + 1);
	aux.setTo(255);

	aux = enhancedImage.rowRange(0, szek + 1).colRange(0, cols);
	aux.setTo(255);

	aux = enhancedImage.rowRange(rows - szek, rows).colRange(0, cols);
	aux.setTo(255);

	aux = enhancedImage.rowRange(0, rows).colRange(cols - 2 * (szek + 1) - 1, cols);
	aux.setTo(255);

	return enhancedImage;
}

//This is equivalent to Matlab's 'meshgrid' function
void RidgeFilter::meshgrid(int sze)
{
	cv::Range xr = cv::Range(-sze, sze);
	cv::Range yr = cv::Range(-sze, sze);

	std::vector<int>t_x, t_y;

	for (int i = xr.start; i <= xr.end; i++)
	{
		t_x.push_back(i);
	}

	for (int j = yr.start; j <= yr.end; j++)
	{
		t_y.push_back(j);
	}

	cv::Mat xgv = cv::Mat(t_x);
	cv::Mat ygv = cv::Mat(t_y);

	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, meshX);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), meshY);
}

//Perform Gabor filtering to final enhancement
//Further description can be found in the paper
cv::Mat RidgeFilter::gaborFiltering(cv::Mat inputImage, double frequency, cv::Mat orientationImage, cv::Mat enhancedImage)
{
	int rows = inputImage.rows;
	int cols = inputImage.cols;

	cv::Mat gaussKernelX = cv::getGaussianKernel(5, 1, CV_32FC1);
	cv::Mat gaussKernelY = cv::getGaussianKernel(5, 1, CV_32FC1);
	cv::Mat gaussKernel = gaussKernelX * gaussKernelY.t();

	cv::Mat freqMat = cv::Mat::ones(rows, cols, CV_32FC1);
	freqMat *= frequency;
	cv::Mat frequencyImage = cv::Mat::zeros(rows, cols, CV_32FC1);

	cv::Ptr<cv::FilterEngine> fe = cv::createLinearFilter(CV_32FC1, CV_32FC1, gaussKernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT,
		cv::BORDER_CONSTANT, cv::Scalar(0));
	fe->apply(freqMat, frequencyImage);

	double sigmaX = 4.0;
	double sigmaY = 1.0;

	double sigmaU = 0.5 * pi * sigmaX;
	double sigmaV = 0.5 * pi * sigmaY;

	double Wg = 7;
	int halfW = round(Wg / 2);

	for (int i = halfW + 1; i < rows - halfW; i++)
	{
		for (int j = halfW + 1; j < cols - halfW; j++)
		{
			float theta = orientationImage.at<float>(i, j) + pi / 2;
			float f = frequency;
			int sum = 0;

			for (int u = -halfW; u <= halfW; u++)
			{
				for (int v = -halfW; v <= halfW; v++)
				{
					float uTheta = u * cos(theta) + v * sin(theta);
					float vTheta = -u * sin(theta) + v * cos(theta);
					float pixel = inputImage.at<float>(i - u, j - v);
					float H = exp(-0.5 * ((uTheta*uTheta) / (sigmaU * sigmaU)) + ((vTheta*vTheta) / (sigmaV*sigmaV))) * cos(2 * pi*f*uTheta);
					sum += H*pixel;
				}
			}
			enhancedImage.at<float>(i, j) = sum;
		}
	}

	return enhancedImage;
}
