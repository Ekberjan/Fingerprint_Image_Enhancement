// Main function for running finpgerprint image enhancement algorithm
//
// This enhancement method is based on Anil Jain's paper:
// 'Fingerprint Image Enhancement: Algorithm and Performance Evaluation',
//  IEEE Transactions on Pattern Analysis and Machine Intelligence,
//  vol. 20, No. 8, August, 1998
//
// Author: Ekberjan Derman
// Contributor : Baptiste Amato, Julien Jerphanion
// Emails:
//    ekberjanderman@gmail.com
//    baptiste.amato@psycle.io
//    git@jjerphan.xyz
//
// Last update : 12.2017

#include "fpenhancement.h"
#include "common.h"
#include "cxxopts.hpp"

using namespace cv;


std::string getImageType(int number) {
	// find type
	int imgTypeInt = number % 8;
	std::string imgTypeString;

	switch (imgTypeInt) {
		case 0:
			imgTypeString = "8U";
			break;
		case 1:
			imgTypeString = "8S";
			break;
		case 2:
			imgTypeString = "16U";
			break;
		case 3:
			imgTypeString = "16S";
			break;
		case 4:
			imgTypeString = "32S";
			break;
		case 5:
			imgTypeString = "32F";
			break;
		case 6:
			imgTypeString = "64F";
			break;
		default:
			break;
	}

	// Find channel
	int channel = (number / 8) + 1;

	std::stringstream type;
	type << "CV_" << imgTypeString << "C" << channel;

	return type.str();
}

int main(int argc, char * argv[]) {

	// CLI management

	cxxopts::Options options("fingerprint",
	"Extract fingerprints from an image");

	options.add_options()
		("i,input_image", "Input image", cxxopts::value<std::string>())
		("o,output_image", "Output image", cxxopts::value<std::string>()->default_value("out.png"))
		("s,show", "Show the result of the algorithm", cxxopts::value<bool>()->default_value("false"))
		("d,downsize", "Downsize the image", cxxopts::value<bool>()->default_value("true"))
		("n,no_save", "Don't save the image", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print usage")
		// ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
		// ("d,debug", "Enable debugging") // a bool parameter
		;

	auto result = options.parse(argc, argv);

	if (result.count("help")) {
		exit(0);
	}

	if (result.count("input_image") + result.count("i") == 0) {
		std::cout << "Bad usage: the input image has to be specified" << std::endl;
		std::cout << options.help() << std::endl;
		exit(1);
	}

	// CLI parameters
	const std::string& input_image = result["input_image"].as<std::string>();
	const std::string& output_image = result["output_image"].as<std::string>();

	bool show_result = result["s"].as<bool>();
	bool downsize = result["d"].as<bool>();
	bool save_image = !(result["no_save"].as<bool>());

	/// 
	
	cv::Mat input = cv::imread(input_image);

	// Make sure the input image is valid
	if (!input.data) {
		std::cerr << "The provided input image is invalid. Please check it again. " << std::endl;
		exit(1);
	}

	if (downsize) {
		while (input.rows > 1000 || input.cols > 1000) {
			const float fact = 0.9;
			std::cout << "Downsizing from (" << input.rows << ", " << input.cols <<
			") to (" << (int) (input.rows * fact) << ", " << (int) (input.cols * fact) << ")" << std::endl;  
			cv::resize(input, input, cv::Size(), fact, fact, cv::INTER_CUBIC);
		}
	}

	// Run the enhancement algorithm
	FPEnhancement fpEnhancement;
	cv::Mat enhancedImage = fpEnhancement.run(input);

	// Doing the postProcessing
	cv::Mat filter = fpEnhancement.postProcessingFilter(input);

	std::cout << "Type of the image  : " << getImageType(enhancedImage.type()) << std::endl;
	std::cout << "Type of the filter : " << getImageType(filter.type()) << std::endl;

	// Finally applying the filter to get the end result
	Mat end_result;

	end_result = Scalar::all(0);
	enhancedImage.copyTo(end_result, filter);

	if (show_result) {
		imshow("End result", end_result);
		std::cout << "Press any key to continue... " << std::endl;
		cv::waitKey();
	}

	if (save_image) {
		imwrite(output_image, end_result);
	}

	return 0;
}
