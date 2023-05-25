project(ipm)
cmake_minimum_required(VERSION 3.25)


find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(ipm
    src/main.cpp
)

target_link_libraries(ipm
    ${OpenCV_LIBS}
)

add_executable(ipm2
    src/secondmethod.cpp
)

target_link_libraries(ipm2
    ${OpenCV_LIBS}
)

add_executable(base_ipm
    src/cameraipm.cpp
)

target_link_libraries(base_ipm
    ${OpenCV_LIBS}
)