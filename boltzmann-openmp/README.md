# Boltzmann Fluid Simulation (OpenMP version)

Copyright (c) 2012-2019 Jose Hernandez

## Pre-requisites

On MacOS, please install libomp before building the simulation program. I.e.

```shell script
brew install libomp
```

## Buiding and running the application

On MacOS generate the makefile with the following cmake comnand:

```shell script
cmake -DOpenMP_C_FLAGS="-fopenmp=lomp" -DOpenMP_CXX_FLAGS="-fopenmp=lomp" -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -I/usr/local/Cellar/libomp/8.0.0/include" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY=/usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp /usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -I/usr/local/Cellar/libomp/8.0.0/include" .
```

On other platforms where CMake can detect OpenMP, the following command may suffice:

Run the following commands to build the application:

```shell script
cmake .
```

After generating the makefile, compile the code using make. I.e.

```shell script
make
```

Once built, the code can be run using this command:

```shell script
./boltzmann-openmp
```

## XCode project generation

You can generate XCode projects using CMake. Please note that you may need to delete *CMakeCache.txt* if the project has been built before using a different gerenerator.

```shell script
rm CMakeCache.txt
cmake -G "XCode" -DOpenMP_C_FLAGS="-fopenmp=lomp" -DOpenMP_CXX_FLAGS="-fopenmp=lomp" -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp /usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -I/usr/local/Cellar/libomp/8.0.0/include" -DOpenMP_CXX_LIB_NAMES="libomp" -DOpenMP_libomp_LIBRARY=/usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp /usr/local/Cellar/libomp/8.0.0/lib/libomp.dylib -I/usr/local/Cellar/libomp/8.0.0/include" .
```

## References

- [LLVM OpenMP®: Support for the OpenMP language](https://openmp.llvm.org)
