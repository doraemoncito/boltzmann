cmake_minimum_required(VERSION 3.14.5)
project(boltzmann-openmp)

set(CMAKE_CXX_STANDARD 17)

add_executable(boltzmann-openmp d2q9-bgk.c)

include(FindOpenMP)
if (OPENMP_FOUND)
    message("OpenMP found; support for OpenMP has been enabled")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else (OPENMP_FOUND)
    message("OpenMP could not be found; support for OpenMP has been disabled")
endif (OPENMP_FOUND)
