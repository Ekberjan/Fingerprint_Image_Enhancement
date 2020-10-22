# Fingerprint Image Enhancement Using OpenCV

[![Build Status](https://travis-ci.org/jjerphan/Fingerprint_Image_Enhancement.svg?branch=master)](https://travis-ci.org/jjerphan/Fingerprint_Image_Enhancement)

A C++ implementation of the enhancement method based on Anil Jain's paper:

*Fingerprint Image Enhancement: Algorithm and Performance Evaluation*, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 20, No. 8, August, 1998. This paper is readable [here](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwjZgPWX5bTYAhWHblAKHTteCPUQFghFMAI&url=http%3A%2F%2Fwww.math.tau.ac.il%2F~turkel%2Fimagepapers%2Ffingerprint.pdf&usg=AOvVaw35b-7mvIizEjNnV54_rrRq).

## Overview
This program is using OpenCV 2.4.10 and program performs respectively :
 - the image normalization,
 - the image orientation calculation
 - a Gabor filter based image enhancement
 - a cropping to only get the fingerprints

## Installation

Install OpenCV on your machine. Then clone the repo:

```bash
git clone git@github.com:jjerphan/Fingerprint_Image_Enhancement.git
cd Fingerprint_Image_Enhancement
```

### CLI

You can build the CLI tool as follows.
```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

You can then execute the script on a picture :
```bash
.bin/fingerPrint -i input.png
```

To have an overview of options, just use:
```
./bin/fingerPrint -h
```


### API

Install the package using:
```bash
python setup.py install
```

## Example of result
Here is an input image :

![Input image : picture of an index finger](doc/input.png)

and the results obtained by running the command above:

![Results : Fingerprint extracted](doc/result.png)

You need to manually save the image shown in the OpenCV window at the end.

## License

This project is distributed under the Apache License 2.0.

It features code from [`edmBernard/pybind11_opencv_numpy`](https://github.com/edmBernard/pybind11_opencv_numpy)
published under the same license.
