#include <assert.h>
#include "hdf5.h"
#include <string.h>
#include <stdlib.h>
#include "read_h5.c"

int main(int argc, char **argv)
{
    int mpi_namelen;
    char mpi_name[MPI_MAX_PROCESSOR_NAME];
    int i, n;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);

	printf("%d\n", mpi_rank);

    char ** ChannelsPointer;
    //int ChNodeNumElements = 1255733622/480; //first survey
    //long ChNodeNumElements = 3742210598/2732; //3 surveys
    long ChNodeNumElements = 24; //3 surveys
    long FirstElement = mpi_rank * ChNodeNumElements;
    
    float ** DataPointer =  (float **) malloc (12 * sizeof (float *));
    for (i=0; i<12; i++) 
        DataPointer[i] =  (float *) malloc (ChNodeNumElements * sizeof (float));
    
    float ** buffer = (float **) malloc (ChNodeNumElements * sizeof (float *));
    buffer[0] = (float *) malloc (12 * ChNodeNumElements * sizeof (float));
    for (i=1; i<ChNodeNumElements; i++)
        buffer[i] = buffer[0] + i * 12;
    //float buffer[12][24];
    
    for (i=0; i<5; i++) 
        printf("%d %.2f %.2f\n", i, buffer[0][i], buffer[1][i]);

    read_h5_data(mpi_rank, ChannelsPointer, FirstElement, ChNodeNumElements, buffer, argv[1]);

    int j;
    for (i=0; i<24; i++) { 
        printf("\n%d", i);
        for (j=0;j<12; j++)
            printf(" %.5f", buffer[i][j]);
    }


    MPI_Finalize();
}
