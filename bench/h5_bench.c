#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include "read_h5.h"
#include <hdf5.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int mpi_namelen, mpi_rank, mpi_size;
    char mpi_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);

	printf("%d\n", mpi_rank);

    long samples_per_process = 805306368/5; // 6GB

    thin_data_struct * data = (thin_data_struct *) malloc (samples_per_process * sizeof (thin_data_struct));
    
    read_hdf5_thin(mpi_rank, mpi_rank * samples_per_process, samples_per_process, data, "../testdata/s32/thin2.h5");
    //read_hdf5_thin(mpi_rank, mpi_rank * samples_per_process, samples_per_process, data, "../testdata/thin.h5");

    MPI_Finalize();
}
