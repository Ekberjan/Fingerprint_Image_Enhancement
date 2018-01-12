set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( fingerPrint )
find_package( OpenCV REQUIRED )
add_executable( fingerPrint checksize.cpp  fpenhancement.cpp  main.cpp  normalizer.cpp  ridgefilter.cpp  ridgeorient.cpp
 )
target_link_libraries( fingerPrint ${OpenCV_LIBS} )
