cmake_minimum_required(VERSION 3.2)

project(Fingerprint_Image_Enhancement)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(FINGERPRINT_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(FINGERPRINT_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(FINGERPRINT_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src)

# Python Binders
find_package( OpenCV REQUIRED )
add_subdirectory(pybind11)
set(BINDERS_FILES  src/fpenhancement.cpp  src/binders.cpp
src/ndarray_converter.cpp)
pybind11_add_module(fingerprint ${BINDERS_FILES})
target_link_libraries( fingerprint PRIVATE ${OpenCV_LIBS} )
