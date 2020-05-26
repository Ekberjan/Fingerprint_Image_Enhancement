cmake_minimum_required(VERSION 3.2)
project(fingerprint-src)

find_package( OpenCV REQUIRED )

set(SOURCE_FILES checksize.cpp  fpenhancement.cpp  main.cpp  normalizer.cpp  ridgefilter.cpp  ridgeorient.cpp)

add_executable( fingerPrint ${SOURCE_FILES})
target_link_libraries( fingerPrint ${OpenCV_LIBS} )
