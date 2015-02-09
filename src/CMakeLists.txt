#set the opencv directories
set(OpenCV_DIR "/usr/local/lib")
find_package(OpenCV REQUIRED core imgproc highgui features2d legacy flann nonfree)
find_package(Boost REQUIRED filesystem system date_time thread program_options)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(surf-detector main.cpp)
target_link_libraries(surf-detector ${OpenCV_LIBS} ${Boost_PROGRAM_OPTIONS_LIBRARY})
