# Boltzmann Fluid Simulations for HPC

## Introduction

This project contains a number of implementations in C/C++ of a Boltzmann fluid simulation optimised for different environments / platforms in addition to the original serial implementation of the d2q9-bgk lattice Boltzmann scheme.

Each subsequent implementation of the code was arrived at by making changes to the preceeding version, starting with the original serial sub-optimal version, in the order listed in this table:

|Executable Name|Description|
|-|-|
|boltzmann-original|Original serial version of the code|
|boltzmann-serial|Optimised (still serial) version of the original serial code|
|boltzmann-openmp|Pararelised version of the code using [OpenMP](https://www.openmp.org)|
|boltzmann-mpi|Distributed version of the code using [MPI](https://www.open-mpi.org)|
|boltzmann-opencl|Graphics Processing Unit version of the code implemented using [OpenCL](https://www.khronos.org/opencl/)|

## Building and Running the Code

Each of the projects can be built and run individually using [CMake](https://cmake.org/), e.g.

```shell script
cd boltzmann-mpi
cmake .
make
```

Please note that unlike the code, which has been run on a number of operating systems and GPU devices, the CMake build files have only been tested on MacOS.

Some of the sub-projects have pre-requisites and specific build instructions wich are described in separate documentation files:

+ [Building and running the d2q9-bgk lattice Boltzmann scheme for MPI](boltzmann-mpi/README.md)
+ [Building and running the d2q9-bgk lattice Boltzmann scheme for OpenMP](boltzmann-openmp/README.md)

## Documentation

This repository also includes a number of short papers describing how the optimisations, paralisation and distribution of the original across a number of nodes was achieved.

1. [Parallelising the Lattice Boltzman scheme](documentation/boltzmann-openmp.pdf)
1. [Distributed Lattice Boltzman using MPI](documentation/boltzmann-mpi.pdf)
1. [Lattice Boltzman simulation in OpenCL](documentation/boltzmann-opencl.pdf)
