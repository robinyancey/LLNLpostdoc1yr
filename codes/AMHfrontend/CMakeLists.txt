cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main-app main-app.cpp)
target_link_libraries(main-app "${TORCH_LIBRARIES}")

target_link_libraries(main-app ${OpenCV_LIBS} )

set_property(TARGET main-app PROPERTY CXX_STANDARD 14)
