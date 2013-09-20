cmake_minimum_required (VERSION 2.6)

set(CMAKE_C_COMPILER mpicc)

if ($ENV{NERSC_HOST} STREQUAL hopper)

    set(CMAKE_C_COMPILER cc)
    set(CMAKE_CXX_COMPILER CC)

endif ($ENV{NERSC_HOST} STREQUAL hopper)

if ($ENV{NERSC_HOST} STREQUAL carver)

    include_directories($ENV{HDF5_PAR_DIR}/include $ENV{OMPI_DIR}/include)
    link_directories ($ENV{OMPI_DIR}/lib)

endif ($ENV{NERSC_HOST} STREQUAL carver)

if ($ENV{NERSC_HOST} STREQUAL gordon)

    include_directories($ENV{HDF5HOME}/include $ENV{MPIHOME}/include)
    link_directories($ENV{HDF5HOME}/lib $ENV{MPIHOME}/lib)
    set(EXTRALINK mpich)

endif ($ENV{NERSC_HOST} STREQUAL gordon)

project(hdf5-bench-proj C)

set(CMAKE_C_FLAGS $ENV{HDF5})
set(CMAKE_EXE_LINKER_FLAGS $ENV{HDF5})

add_library(read_h5 read_h5.c)
target_link_libraries (read_h5 hdf5 ${EXTRALINK})

add_executable (h5_bench h5_bench.c) 
target_link_libraries (h5_bench read_h5 ${EXTRALINK})

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
