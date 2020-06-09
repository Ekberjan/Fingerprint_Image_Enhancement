cmake_minimum_required(VERSION 3.2)
project(fpenhancement_tests)

find_package( OpenCV REQUIRED )

add_subdirectory(lib/googletest)

include_directories(${FINGERPRINT_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/fingerprint_tests.cpp)

add_executable(fingerprint_tests ${SOURCE_FILES})
target_link_libraries(fingerprint_tests fpenhancement ${OpenCV_LIBS} gtest)
install(TARGETS fingerprint_tests DESTINATION bin)