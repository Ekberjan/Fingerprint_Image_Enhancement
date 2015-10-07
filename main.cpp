//Main function for running finpgerprint image enhancement algorithm
//Author: Ekberjan Derman
//email: ekberjanderman@gmail.com
//08.2015
//Enhancement method is based on Anil Jain's paper:
//'Fingerprint Image Enhancement: Algorithm and Performance Evaluation', 
//IEEE Transactions on Pattern Analysis and Machine Intelligence, 
// vol. 20, No. 8, August, 1998


#include "stdafx.h"
#include "fpenhancement.h"

int _tmain(int argc, _TCHAR* argv[])
{
	cv::Mat input = cv::imread("inputImage.tif"); // change this line to load your actual input file

	//Run the enhancement algorithm
	FPEnhancement fpEnhancement;
	cv::Mat enhancedImage = fpEnhancement.run(input);

	//Display the result for check
	cv::imshow("enhancedImage", enhancedImage);
	std::cout << "Press any key to continue... " << std::endl;
	cv::waitKey();

	return 0;
}

