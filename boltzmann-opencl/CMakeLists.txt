cmake_minimum_required(VERSION 3.14.5)
project(boltzmann-opencl)

option(DEBUG "Enable debug mode" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(OpenCL REQUIRED)

add_executable(boltzmann-opencl device_info.c err_code.c wtime.c d2q9-bgk.c d2q9-bgk.cl input.params)

if (APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework OpenCL")
elseif (WIN32)
    target_link_libraries(boltzmann-opencl ${OPENCL_LIBRARIES} winmm.lib)
endif (APPLE)

if (DEBUG)
    message("-- Debug mode enabled")
    add_definitions(-DDEBUG)
else (DEBUG)
    message("-- Debug mode disabled")
endif (DEBUG)

add_definitions(-DCL_KHR_FP32)

configure_file(d2q9-bgk.cl ${CMAKE_CURRENT_BINARY_DIR}/d2q9-bgk.cl COPYONLY)
configure_file(input.params ${CMAKE_CURRENT_BINARY_DIR}/input.params COPYONLY)
configure_file(obstacles.dat ${CMAKE_CURRENT_BINARY_DIR}/obstacles.dat COPYONLY)
