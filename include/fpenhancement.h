#ifndef _FPENHANCEMENT_H
#define _FPENHANCEMENT_H

#include "common.h"

class FPEnhancement {
public:
    FPEnhancement(double kx = 0.8,
                  double ky = 0.8,
                  double blockSigma = 5.0,
                  double gradientSigma = 1.0,
                  double orientSmoothSigma = 5.0,
                  // The frequency is set to 0.11 experimentally. You can change this value
                  // or use a dynamic calculation instead.
                  double freqValue = 0.11,
                  int ddepth = CV_32FC1,
                  bool addBorder = false,
                  int cannyLowThreshold = 10,
                  int cannyRatio = 3,
                  int kernelSize = 3,
                  int blurringTimes = 30,
                  int dilationSize = 10,
                  int dilationType = 1,
                  bool verbose = false) : kx(kx),
                                          ky(ky),
                                          blockSigma(blockSigma),
                                          gradientSigma(gradientSigma),
                                          orientSmoothSigma(orientSmoothSigma),
                                          freqValue(freqValue),
                                          ddepth(ddepth),
                                          addBorder(addBorder),
                                          cannyLowThreshold(cannyLowThreshold),
                                          cannyRatio(cannyRatio),
                                          kernelSize(kernelSize),
                                          blurringTimes(blurringTimes),
                                          dilationSize(dilationSize),
                                          dilationType(dilationType),
                                          verbose(verbose){};

    cv::Mat extractFingerPrints(const cv::Mat &inputImage);

    cv::Mat postProcessingFilter(const cv::Mat &inputImage) const;

private:
    const bool verbose;

    // Image normalization
    static cv::Mat normalize_image(const cv::Mat &im, double reqMean, double reqVar);
    static float deviation(const cv::Mat &im, float ave);

    // For calculating orientation field
    const int ddepth;
    void gradient(const cv::Mat &image, cv::Mat &xGradient, cv::Mat &yGradient) const;
    cv::Mat orient_ridge(const cv::Mat &im);

    // For filtering ridges
    const bool addBorder;
    static void meshgrid(int kernelSize, cv::Mat &meshX, cv::Mat &meshY);
    cv::Mat filter_ridge(const cv::Mat &inputImage, const cv::Mat &orientationImage, const cv::Mat &frequency) const;

    const double kx, ky;
    const double blockSigma;
    const double gradientSigma;
    const double orientSmoothSigma;
    const double freqValue;

    // Post processingFiltering
    const int cannyLowThreshold;
    const int cannyRatio;
    const int kernelSize;
    const int blurringTimes;
    const int dilationSize;
    const int dilationType;
};


#endif
