cmake_minimum_required (VERSION 2.6)

if ($ENV{NERSC_HOST} STREQUAL hopper)

    set(CMAKE_C_COMPILER cc)
    set(CMAKE_CXX_COMPILER CC)

else ($ENV{NERSC_HOST} STREQUAL carver)

    set(CMAKE_C_COMPILER mpicc)
    include_directories($ENV{HDF5_PAR_DIR}/include $ENV{OMPI_DIR}/include)
    link_directories ($ENV{OMPI_DIR}/lib)

endif ($ENV{NERSC_HOST} STREQUAL hopper)

project(hdf5-bench-proj C)

set(CMAKE_C_FLAGS $ENV{HDF5})
set(CMAKE_EXE_LINKER_FLAGS $ENV{HDF5})

add_library(read_h5 read_h5.c)
target_link_libraries (read_h5 hdf5)

add_executable (h5_bench h5_bench.c) 
target_link_libraries (h5_bench read_h5)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
