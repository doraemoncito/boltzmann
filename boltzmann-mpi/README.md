# Boltzmann Fluid Simulation (MPI version)

Copyright (c) 2012-2019 Jose Hernandez

## Pre-requisites

On MacOS, please install libomp before building the simulation program. I.e.

```shell script
brew install openmpi
```

## Buiding and running the application

Generate the makefile with the following cmake comnand:

```shell script
cmake .
```

After generating the makefile, compile the code using make. I.e.

```shell script
make
```

Once built, the code can be run using this command:

```shell script
mpirun boltzmann-mpi
```

## XCode project generation

You can generate XCode projects using CMake. Please note that you may need to delete *CMakeCache.txt* if the project has been built before using a different gerenerator.

```shell script
rm CMakeCache.txt
cmake  .
```
