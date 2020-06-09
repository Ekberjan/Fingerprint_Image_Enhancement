cmake_minimum_required(VERSION 3.2)
project(fpenhancement)

find_package( OpenCV REQUIRED )

set(SOURCE_FILES  fpenhancement.cpp  main.cpp)

add_executable( fingerprint ${SOURCE_FILES})
target_link_libraries( fingerprint ${OpenCV_LIBS} )

add_library(fpenhancement SHARED STATIC ${SOURCE_FILES})
install(TARGETS fingerprint DESTINATION ${FINGERPRINT_INSTALL_BIN_DIR})
install(FILES fpenhancement.h DESTINATION ${FINGERPRINT_INSTALL_INCLUDE_DIR})