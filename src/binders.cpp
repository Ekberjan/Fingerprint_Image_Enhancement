#include <pybind11/pybind11.h>
#include <string.h>
#include "fpenhancement.h"
namespace py = pybind11;


PYBIND11_MODULE(finger, m) {
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
                      >());
}


