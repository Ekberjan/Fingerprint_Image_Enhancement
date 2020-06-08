cmake_minimum_required(VERSION 3.2)
project(fingerprint)

find_package( OpenCV REQUIRED )

set(SOURCE_FILES  fpenhancement.cpp  main.cpp)

add_executable( fingerprint ${SOURCE_FILES})
target_link_libraries( fingerprint ${OpenCV_LIBS} )
install(TARGETS fingerprint DESTINATION ${FINGERPRINT_INSTALL_BIN_DIR})