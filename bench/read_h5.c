#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "read_h5.h"
#include <hdf5.h>
#include <mpi.h>

void read_hdf5_thin(int pid, long first_elem, int num_elements, thin_data_struct *data, const char * filename)
{
    hsize_t start[1]; hsize_t count[1]; hsize_t stride[1]; hsize_t dims[1]; 
    int i, n;
    bool val;

    stride[0] = 1; count[0] = num_elements;
    start[0] = first_elem;

	printf("%d read_hdf5_thin ", pid);
    printf(" %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    ///* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    ///* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    hid_t fid1 = H5Fopen(filename, H5F_ACC_RDONLY, acc_tpl1);

    /* Release file-access template */
    ret = H5Pclose(acc_tpl1); assert(ret != FAIL);
    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

    /* create a compound hdf5 memory type */
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof(thin_data_struct));
    H5Tinsert (memtype, "L" , HOFFSET (thin_data_struct, L) ,  H5T_NATIVE_LONG);
    H5Tinsert (memtype, "D0", HOFFSET (thin_data_struct, D0),  H5T_NATIVE_DOUBLE);
    H5Tinsert (memtype, "D1", HOFFSET (thin_data_struct, D1),  H5T_NATIVE_DOUBLE);
    H5Tinsert (memtype, "D2", HOFFSET (thin_data_struct, D2),  H5T_NATIVE_DOUBLE);
    H5Tinsert (memtype, "D3", HOFFSET (thin_data_struct, D3),  H5T_NATIVE_DOUBLE);

    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL); assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER); assert(xfer_plist != FAIL);
    //ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT); assert(ret != FAIL);

    //printf("Hyperslab\n");
    /* select hyperslab */
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, NULL); assert(ret != FAIL);
    start[0] = 0;
    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, NULL); assert(ret != FAIL);

    /* read data collectively */
    ret = H5Dread(dataset, memtype, mem_dataspace, file_dataspace, xfer_plist, data);
    assert(ret != FAIL);

    for (i=0; i<5; i++)
        printf("%i: %i, %f, %f, %f\n", pid, data[i].L, data[i].D0, data[i].D1, data[i].D2);

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); 
    H5Fclose(fid1);
}
