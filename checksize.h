#ifndef _CHECK_SIZE_H
#define _CHEKC_SIZE_H

#include "commonfiles.h"

class CheckSize {
public:
	cv::Mat check(cv::Mat& inputImage);

private:
	int expectedRow;
	int expectedCol;
	cv::Mat result;
};

#endif
