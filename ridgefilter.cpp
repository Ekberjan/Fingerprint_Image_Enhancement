// Main function for performing Gabor filtering
// for enhancement using previously calculated
// orientation image and frequency. The output
// is final enhanced image.
// Refer to the paper for detailed description.
//
// Author: Ekberjan Derman
// Contributor : Baptiste Amato, Julien Jerphanion
// Emails:
//    ekberjanderman@gmail.com
//    baptiste.amato@psycle.io
//    julien.jerphanion@psycle.io
// Last update : 12.2017

#include "ridgefilter.h"

namespace cv {
	using std::vector;
}

// Perform Gabor filtering to final enhancement
// Further description can be found in the paper
cv::Mat RidgeFilter::run(cv::Mat inputImage, cv::Mat orientationImage, cv::Mat frequency, double kx,
						 double ky) {
	angleInc = 3; // fixed angle increment between filter orientations  in degrees.

	inputImage.convertTo(inputImage, CV_32FC1);
	int rows = inputImage.rows;
	int cols = inputImage.cols;

	orientationImage.convertTo(orientationImage, CV_32FC1);

	enhancedImage = cv::Mat::zeros(rows, cols, CV_32FC1);
	cv::vector<int> validr;
	cv::vector<int> validc;

	double unfreq = frequency.at<float>(1, 1);

	cv::Mat freqindex = cv::Mat::ones(100, 1, CV_32FC1);

	double sigmax = (1 / unfreq) * kx;
	double sigmax_squared = sigmax * sigmax;
	double sigmay = (1 / unfreq) * ky;
	double sigmay_squared = sigmay * sigmay;

	int szek = round(3 * (std::max(sigmax, sigmay)));
	meshgrid(szek);

	reffilter = cv::Mat::zeros(meshX.rows, meshX.cols, CV_32FC1);

	meshX.convertTo(meshX, CV_32FC1);
	meshY.convertTo(meshY, CV_32FC1);

	double pi_by_unfreq_by_2 = 2 * pi * unfreq;

	for (int i = 0; i < meshX.rows; i++) {
		const float *meshX_i = meshX.ptr<float>(i);
		const float *meshY_i = meshY.ptr<float>(i);
		float *reffilter_i = reffilter.ptr<float>(i);
		for (int j = 0; j < meshX.cols; j++) {
			float meshX_i_j = meshX_i[j];
			float meshY_i_j = meshY_i[j];
			float pixVal2 = -0.5 * (meshX_i_j * meshX_i_j / sigmax_squared +
									meshY_i_j * meshY_i_j / sigmay_squared);
			float pixVal = std::exp(pixVal2);
			float cosVal = pi_by_unfreq_by_2 * meshX_i_j;
			reffilter_i[j] = pixVal * cos(cosVal);
		}
	}

	for (int m = 0; m < 180 / angleInc; m++) {
		double angle = -(m * angleInc + 90);
		cv::Mat rot_mat = cv::getRotationMatrix2D(
				cv::Point((float) (reffilter.rows / 2.0F), (float) (reffilter.cols / 2.0F)), angle,
				1.0);
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

	int rows_maxsze = rows - maxsze;
	int cols_maxsze = cols - maxsze;

	for (int y = 0; y < rows; y++) {
		const float *orientationImage_y = orientationImage.ptr<float>(y);
		float *orientindex_y = orientindex.ptr<float>(y);
		for (int x = 0; x < cols; x++) {
			if (x > maxsze && x < cols_maxsze && y > maxsze && y < rows_maxsze) {
				validr.push_back(y);
				validc.push_back(x);
			}

			int orientpix = static_cast<int>(std::round(
					orientationImage_y[x] / pi * 180 / angleInc));

			if (orientpix < 0) {
				orientpix += maxorientindex;
			}
			if (orientpix >= maxorientindex) {
				orientpix -= maxorientindex;
			}

			orientindex_y[x] = orientpix;
		}
	}

	// Finally, do the filtering
	for (int k = 0; k < validr.size(); k++) {
		int r = validr[k];
		int c = validc[k];

		cv::Rect roi(c - szek - 1, r - szek - 1, meshX.cols, meshX.rows);
		cv::Mat subim(inputImage(roi));

		cv::Mat subFilter = filter.at(orientindex.at<float>(r, c));
		cv::Mat mulResult;
		cv::multiply(subim, subFilter, mulResult);

		// float value = cv::sum(mulResult)[0];
		if (cv::sum(mulResult)[0] > 0) {
			enhancedImage.at<float>(r, c) = 255;
		}

	}

	// Add a border.
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

// This is equivalent to Matlab's 'meshgrid' function
void RidgeFilter::meshgrid(int sze) {
	std::vector<int> t;

	for (int i = -sze; i < sze; i++) {
		t.push_back(i);
	}

	cv::Mat gv = cv::Mat(t);
	int total = gv.total();
	gv = gv.reshape(1, 1);

	cv::repeat(gv, total, 1, meshX);
	cv::repeat(gv.t(), 1, total, meshY);
}
