cmake_minimum_required(VERSION 3.2)

project(fingerPrint)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

include_directories(include)

add_subdirectory(src)

# Python Binders
find_package( OpenCV REQUIRED )
add_subdirectory(pybind11)
set(BINDERS_FILES  src/fpenhancement.cpp  src/binders.cpp )
pybind11_add_module(fingerprint ${BINDERS_FILES})
target_link_libraries( fingerprint PRIVATE ${OpenCV_LIBS} )

