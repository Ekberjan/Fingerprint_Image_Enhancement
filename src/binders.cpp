#include <pybind11/pybind11.h>
#include <string.h>
#include "fpenhancement.h"
#include "common.h"
#include "ndarray_converter.h"

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(fingerprint, m) {
    NDArrayConverter::init_numpy();

    m.doc() = "Finger print extraction";

    py::class_<FPEnhancement>(m, "Extractor")
        .def(py::init<double, // kx
                      double, // ky
                      double, // blockSigma
                      double, // gradientSigma
                      double, // orientSmoothSigma
                      double, // freqValue
                      int,    // ddepth
                      bool,   // addBorder
                      int,    // cannyLowThreshold
                      int,    // cannyRatio
                      int,    // kernelSize
                      int,    // blurringTimes
                      int,    // dilationSize
                      int,    // dilationType
                      bool    // verbose
                      >(),
                      "Constructor",
                      "kx"_a = 0.8,
                      "ky"_a = 0.8,
                      "block_sigma"_a = 5.0,
                      "gradient_sigma"_a = 1.0,
                      "orient_smooth_sigma"_a = 5.0,
                      "freq_value"_a = 0.11,
                      "ddepth"_a = CV_32FC1,
                      "add_border"_a = false,
                      "canny_low_threshold"_a = 10,
                      "canny_ratio"_a = 3,
                      "kernel_size"_a = 3,
                      "blurring_times"_a = 30,
                      "dilation_size"_a = 10,
                      "dilation_type"_a = 1,
                      "verbose"_a = false
                      )
        .def("extract_fingerprints", &FPEnhancement::extractFingerPrints)
        .def("post_processing", &FPEnhancement::postProcessingFilter);
}


