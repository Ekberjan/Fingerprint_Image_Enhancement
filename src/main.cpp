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

#include "common.h"
#include "cxxopts.hpp"
#include "fpenhancement.h"

std::string getImageType(int number) {
  // Find type
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

int main(int argc, char *argv[]) {

  // CLI management

  cxxopts::Options options("fingerprint", "Extract fingerprints from an image");

  options.add_options()("i,input_image", "Input image",
                        cxxopts::value<std::string>())(
      "o,output_image", "Output image",
      cxxopts::value<std::string>()->default_value("out.png"))(
      "s,show", "Show the result of the algorithm",
      cxxopts::value<bool>()->default_value("false"))(
      "d,downsize", "Downsize the image",
      cxxopts::value<bool>()->default_value("false"))(
      "b,border", "Add border to the image",
      cxxopts::value<bool>()->default_value("false"))(
      "n,no_save", "Don't save the image",
      cxxopts::value<bool>()->default_value("false"))(
      "min_rows", "Minimum number of rows",
      cxxopts::value<int>()->default_value("1000"))(
      "min_cols", "Minimum number of columns",
      cxxopts::value<int>()->default_value("1000"))

      ("h,help", "Print usage")("v,verbose", "Verbose output",
                                cxxopts::value<bool>()->default_value("false"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    exit(0);
  }

  if (result.count("input_image") + result.count("i") == 0) {
    std::cerr << "Bad usage: the input image has to be specified" << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
  }

  // CLI parameters
  const auto &input_image = result["input_image"].as<std::string>();
  const auto &output_image = result["output_image"].as<std::string>();

  bool show_result = result["s"].as<bool>();
  bool downsize = result["d"].as<bool>();
  bool save_image = !(result["n"].as<bool>());
  bool verbose = result["v"].as<bool>();

  int min_rows = result["min_rows"].as<int>();
  int min_cols = result["min_cols"].as<int>();

  ///

  cv::Mat input = cv::imread(input_image);

  // Make sure the input image is valid
  if (!input.data) {
    std::cerr << "The provided input image is invalid. Please check it again. "
              << std::endl;
    exit(1);
  }

  if (downsize) {
    while (input.rows > min_rows || input.cols > min_cols) {
      const float fact = 0.9;
      if (verbose) {
        std::cout << "Downsizing from (" << input.rows << ", " << input.cols
                  << ") to (" << (int)(input.rows * fact) << ", "
                  << (int)(input.cols * fact) << ")" << std::endl;
      }
      cv::resize(input, input, cv::Size(), fact, fact, cv::INTER_CUBIC);
    }
  }

  // Run the enhancement algorithm
  FPEnhancement fpEnhancement;
  cv::Mat enhancedImage = fpEnhancement.extractFingerPrints(input);

  // Doing the postProcessing
  cv::Mat filter = fpEnhancement.postProcessingFilter(input);

  if (verbose) {
    std::cout << "Type of the image  : " << getImageType(enhancedImage.type())
              << std::endl;
    std::cout << "Type of the filter : " << getImageType(filter.type())
              << std::endl;
  }

  // Finally applying the filter to get the end result
  cv::Mat endResult(cv::Scalar::all(0));
  enhancedImage.copyTo(endResult, filter);

  if (show_result) {
    cv::imshow("End result", endResult);
    std::cout << "Press any key to continue... " << std::endl;
    cv::waitKey();
  }

  if (save_image) {
    cv::imwrite(output_image, endResult);
  }

  return 0;
}
