#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "read_h5.h"
#include "chealpix.h"
#define CPTR(VAR,CONST) ((VAR)=(CONST),&(VAR))

int c_bin_baselines(double * baselines, int * pix, int * baseline_lengths, double * tmap, int num_bas, int num_pix) {
    unsigned int i_bas, start, i;
    for (i=0; i<num_pix;i++)
        tmap[i] = 0.;
    start = 0;
    for (i_bas=0; i_bas < num_bas; i_bas++) 
    {
        
        for (i=start; i < start+baseline_lengths[i_bas]; i++)
            tmap[pix[i]] += baselines[i_bas];
        start = i;
    }
    return 0;
}

int c_bin_baselines_mp(double * baselines, int * pix, int * baseline_offsets, double * tmaps, int num_bas, int num_pix) {
    #pragma omp parallel
    {
        unsigned int i_bas, i, n;
        unsigned int thread_off = omp_get_thread_num() * num_pix;
        unsigned int num_threads = omp_get_num_threads();

        for (i=thread_off; i<thread_off+num_pix;i++)
        {
            tmaps[i] = 0.;
        }

        #pragma omp for schedule(static)
        for (i_bas=0; i_bas < num_bas; i_bas++) 
        {
            for (i=baseline_offsets[i_bas]; i < baseline_offsets[i_bas+1]; i++)
                tmaps[thread_off+pix[i]] += baselines[i_bas];
        }

        #pragma omp for schedule(static)
        for (i=0; i<num_pix;i++) {
            for (n=1; n<num_threads; n++)
                tmaps[i] += tmaps[n*num_pix+i];
        }
    }
    return 0;
}

int c_sigremove_mp(double * baselines, double * output, int * pix, int * baseline_offsets, int * baseline_lenghts, double * tmap, int num_bas) {
    #pragma omp parallel
    {
        unsigned int i_bas, i;

        #pragma omp for schedule(static)
        for (i_bas=0; i_bas < num_bas; i_bas++) 
        {
            output[i_bas] = baseline_lenghts[i_bas] * baselines[i_bas];
            for (i=baseline_offsets[i_bas]; i < baseline_offsets[i_bas+1]; i++)
                output[i_bas] -= tmap[pix[i]];
        }
    }
    return 0;
}

int c_scanmap_mp(double * map, int * pix, double * out, int numelements) {
    #pragma omp parallel
    {
        unsigned int i = 0;
        #pragma omp for schedule(static)
        for (i=0; i<numelements;i++) {
            out[i] = map[pix[i]];
        }
    }
    return 0;
}

int c_bin_cal_mp(double * baselines, double * data, double * dipole, double * gain, int * pix, int * baseline_offsets, double * tmaps, int num_bas, int num_pix) {
    #pragma omp parallel
    {
        unsigned int i_bas, i, n;
        unsigned int thread_off = omp_get_thread_num() * num_pix;
        unsigned int num_threads = omp_get_num_threads();

        for (i=thread_off; i<thread_off+num_pix;i++)
        {
            tmaps[i] = 0.;
        }

        #pragma omp for schedule(static)
        for (i_bas=0; i_bas < num_bas; i_bas++) 
        {
            for (i=baseline_offsets[i_bas]; i < baseline_offsets[i_bas+1]; i++)
                tmaps[thread_off+pix[i]] += data[i] * gain[i_bas] - dipole[i] - baselines[i_bas];
        }

        #pragma omp for schedule(static)
        for (i=0; i<num_pix;i++) {
            for (n=1; n<num_threads; n++)
                tmaps[i] += tmaps[n*num_pix+i];
        }
    }
    return 0;
}

    

int get_num_threads() {
    int num_threads;
    #pragma omp parallel shared(num_threads)
    {
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}

int compute_weights(InputScalar psi, float * qw, float *uw) {
    Scalar dpsi, spsi, cpsi, cf;
    dpsi = (Scalar) psi;
    spsi = sin(dpsi);
    cpsi = cos(dpsi);
    cf = 1./(cpsi*cpsi + spsi*spsi);
    *qw = (float) (cpsi*cpsi -   spsi*spsi)*cf;
    *uw = (float) 2*cpsi*spsi*cf;
}

int read_h5_pids(long ** datap, int* NumPIDs, const char *filename)
{

    hsize_t     dims[1];
    int ndims;
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,H5P_DEFAULT);
    int status;

    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT);
    assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset);
    assert(file_dataspace != FAIL);

    ndims = H5Sget_simple_extent_dims (file_dataspace, dims, NULL);

    long * data = (long *) malloc (dims[0] * sizeof (long));
    *datap = data;
    NumPIDs[0] = dims[0];

    int ret = H5Dread(dataset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    assert(ret != FAIL);

    //int i;
    //for (i=0; i<10; i++)
    //    //printf("%ld, %ld, %ld, %ld \n", data[i].start_30,  data[i].start_44,  data[i].start_70, data[i].pointID_unique );
    //    printf("%ld \n", data[i]);

    H5Sclose(file_dataspace);

    ret=H5Dclose(dataset);
    assert(ret != FAIL);

    H5Fclose(fid1);
}

int read_h5_vec(const char *filename, long total_length, long firstelem, int numelements, Scalar *buffer)
{
    int i;
    hsize_t start[1]; hsize_t start_mem[1]; hsize_t count[1]; hsize_t stride[1]; 
    stride[0] = 1; start[0] = firstelem; start_mem[0] = 0;

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    //fid1=H5Fcreate(mapfilename, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl1);
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1); assert(ret != FAIL);

    //file_dataspace = H5Screate_simple (1, count, NULL); assert(file_dataspace != FAIL);

    /* open the dataset1 collectively */
    //dataset = H5Dcreate2(fid1, "data", H5T_NATIVE_SCALAR, file_dataspace, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);


    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER); assert(xfer_plist != FAIL);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT); assert(ret != FAIL);

    /* select hyperslab */
    count[0] = numelements;
    printf("read_h5_vec: %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL); assert (mem_dataspace != FAIL);
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL, count, NULL); assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, NULL, count, NULL); assert(ret != FAIL);

    /* write data collectively */
    //ret = H5Dwrite(dataset, H5T_NATIVE_SCALAR, mem_dataspace, file_dataspace, xfer_plist, buffer);
    ret = H5Dread(dataset, H5T_NATIVE_SCALAR, mem_dataspace, file_dataspace, xfer_plist, buffer); assert(ret != FAIL);

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); H5Fclose(fid1);
}

int read_h5_data(int MyPID, long firstelem, int numelements, InputScalar **data, const char *filename, int NumChannels)
{
    hsize_t start[2]; hsize_t start_mem[2]; hsize_t count[2]; hsize_t stride[2]; hsize_t dims[2]; 
    int i, n;

    stride[0] = 1; stride[1] = 1; count[0] = numelements; count[1] = NumChannels; 
    start[0] = firstelem; start[1] = 0; start_mem[0] = 0; start_mem[1] = 0;

	printf("%d read_h5_data ", MyPID);
    printf(" %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    //printf("Open\n");
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1); assert(ret != FAIL);
    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (2, count, NULL); assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER); assert(xfer_plist != FAIL);
    //ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT); assert(ret != FAIL);

    //printf("Hyperslab\n");
    /* select hyperslab */
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, NULL); assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, stride, count, NULL); assert(ret != FAIL);

    /* read data collectively */
    ret = H5Dread(dataset, H5T_NATIVE_INPUT, mem_dataspace, file_dataspace, xfer_plist, data[0]);
    assert(ret != FAIL);

    for (i=0; i<2; i++)
        printf("%f, %f, %f, %f \n", data[i][0],   data[i][1],  data[i][2],  data[i][3] );

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); 
    H5Fclose(fid1);
}

int read_h5_data_col(int MyPID, long firstelem, int numelements, InputScalar *data, const char *filename, int NumChannel)
{
    hsize_t start[2]; hsize_t count[2]; hsize_t stride[2]; hsize_t dims[2]; 
    int i, n;

    stride[0] = 1; stride[1] = 1; count[0] = numelements; count[1] = 1; 
    start[0] = firstelem; start[1] = NumChannel;

	printf("%d read_h5_data ", MyPID);
    printf(" %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    //printf("Open\n");
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1); assert(ret != FAIL);
    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

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
    start[1] = 0;
    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start, stride, count, NULL); assert(ret != FAIL);

    /* read data collectively */
    ret = H5Dread(dataset, H5T_NATIVE_INPUT, mem_dataspace, file_dataspace, xfer_plist, data);
    assert(ret != FAIL);

    for (i=0; i<2; i++)
        printf("%f, \n", data[i]);

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); 
    H5Fclose(fid1);
}

int write_h5_M(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX)
{
    
    int i;
    hsize_t start[1]; 
    hsize_t start_mem[1]; 
    hsize_t count[1]; 
    hsize_t stride[1]; 
    MDATATYPE *buffer = (MDATATYPE *) malloc(numelements * sizeof (MDATATYPE));
    for (i=0; i<numelements; i++) {
        buffer[i].II = data[0][i];
        buffer[i].IQ = data[1][i];
        buffer[i].IU = data[2][i];
        buffer[i].QQ = data[3][i];
        buffer[i].QU = data[4][i];
        buffer[i].UU = data[5][i];
    }

    stride[0] = 1;	/* for hyperslab setting */
    start[0] = firstelem;			/* for hyperslab setting */
    start_mem[0] = 0;

    /*
     * Create the compound datatype for memory.
     */
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof(MDATATYPE));
    int ret = H5Tinsert (memtype, "II", HOFFSET  (MDATATYPE, II),  H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "IQ" , HOFFSET (MDATATYPE, IQ ), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "IU" , HOFFSET (MDATATYPE, IU ), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "QQ" , HOFFSET (MDATATYPE, QQ ), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "QU" , HOFFSET (MDATATYPE, QU ), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "UU" , HOFFSET (MDATATYPE, UU ), H5T_NATIVE_SCALAR);

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS);
    assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != FAIL);

    /* open the file collectively */
    hid_t fid1=H5Fcreate(mapfilename, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1);
    assert(ret != FAIL);

    count[0] = NPIX;
    hid_t file_dataspace = H5Screate_simple (1, count, NULL);
    assert(file_dataspace != FAIL);

    /* open the dataset1 collectively */
    hid_t dataset = H5Dcreate2(fid1, "data", memtype, file_dataspace, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    assert(dataset != FAIL);


    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL);
    assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
    assert(xfer_plist != FAIL);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);
    assert(ret != FAIL);

    /* select hyperslab */
    count[0] = numelements;
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL,
	    count, NULL);
    assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, NULL,
	    count, NULL);
    assert(ret != FAIL);

    /* write data collectively */
    ret = H5Dwrite(dataset, memtype, mem_dataspace, file_dataspace, xfer_plist, buffer);
    assert(ret != FAIL);

    free(buffer);

    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    ret=H5Dclose(dataset);
    assert(ret != FAIL);

    H5Fclose(fid1);
}

int write_h5_vec(const char *mapfilename, long total_length, long firstelem, int numelements, Scalar *buffer)
{
    //remove the UNSEEN pixel
    if ((firstelem + numelements) > total_length)
        numelements--;
    int i;
    hsize_t start[1]; 
    hsize_t start_mem[1]; 
    hsize_t count[1]; 
    hsize_t stride[1]; 

    stride[0] = 1;	/* for hyperslab setting */
    start[0] = firstelem;			/* for hyperslab setting */
    start_mem[0] = 0;

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS);
    assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != FAIL);

    /* open the file collectively */
    hid_t fid1=H5Fcreate(mapfilename, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1);
    assert(ret != FAIL);

    count[0] = total_length;
    hid_t file_dataspace = H5Screate_simple (1, count, NULL);
    assert(file_dataspace != FAIL);

    /* open the dataset1 collectively */
    hid_t dataset = H5Dcreate2(fid1, "data", H5T_NATIVE_SCALAR, file_dataspace, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    assert(dataset != FAIL);


    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL);
    assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
    assert(xfer_plist != FAIL);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);
    assert(ret != FAIL);

    /* select hyperslab */
    count[0] = numelements;
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL,
	    count, NULL);
    assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, NULL,
	    count, NULL);
    assert(ret != FAIL);

    /* write data collectively */
    ret = H5Dwrite(dataset, H5T_NATIVE_SCALAR, mem_dataspace, file_dataspace, xfer_plist, buffer);
    assert(ret != FAIL);

    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    ret=H5Dclose(dataset);
    assert(ret != FAIL);

    H5Fclose(fid1);
}

int write_h5_map(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX)
{
    int i;
    hsize_t start[1]; 
    hsize_t start_mem[1]; 
    hsize_t count[1]; 
    hsize_t stride[1]; 
    MAPDATATYPE *buffer = (MAPDATATYPE *) malloc(numelements * sizeof (MAPDATATYPE));
    for (i=0; i<numelements; i++) {
        buffer[i].I = data[0][i];
        buffer[i].Q = data[1][i];
        buffer[i].U = data[2][i];
    }

    stride[0] = 1;	/* for hyperslab setting */
    start[0] = firstelem;			/* for hyperslab setting */
    start_mem[0] = 0;

    /*
     * Create the compound datatype for memory.
     */
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (MAPDATATYPE));
    int ret = H5Tinsert (memtype, "I",  HOFFSET (MAPDATATYPE, I), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "Q" , HOFFSET (MAPDATATYPE, Q), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "U" , HOFFSET (MAPDATATYPE, U), H5T_NATIVE_SCALAR);

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS);
    assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != FAIL);

    /* open the file */
    hid_t fid1=H5Fcreate(mapfilename, H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1);
    assert(ret != FAIL);

    count[0] = NPIX;
    hid_t file_dataspace = H5Screate_simple (1, count, NULL);
    assert(file_dataspace != FAIL);

    /* open the dataset1 */
    hid_t dataset = H5Dcreate2(fid1, "data", memtype, file_dataspace, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    assert(dataset != FAIL);


    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL);
    assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
    assert(xfer_plist != FAIL);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);
    assert(ret != FAIL);

    /* select hyperslab */
    count[0] = numelements;
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL,
	    count, NULL);
    assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, NULL,
	    count, NULL);
    assert(ret != FAIL);

    /* write data collectively */
    ret = H5Dwrite(dataset, memtype, mem_dataspace, file_dataspace, xfer_plist, buffer);
    assert(ret != FAIL);

    free(buffer);

    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    ret=H5Dclose(dataset);
    assert(ret != FAIL);

    H5Fclose(fid1);
}

int read_h5_map(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX)
{
    int i;
    hsize_t start[1]; 
    hsize_t start_mem[1]; 
    hsize_t count[1]; 
    hsize_t stride[1]; 
    MAPDATATYPE *buffer = (MAPDATATYPE *) malloc(numelements * sizeof (MAPDATATYPE));

    stride[0] = 1;	/* for hyperslab setting */
    start[0] = firstelem;			/* for hyperslab setting */
    start_mem[0] = 0;

    /*
     * Create the compound datatype for memory.
     */
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (MAPDATATYPE));
    int ret = H5Tinsert (memtype, "I",  HOFFSET (MAPDATATYPE, I), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "Q" , HOFFSET (MAPDATATYPE, Q), H5T_NATIVE_SCALAR);
    ret = H5Tinsert (memtype, "U" , HOFFSET (MAPDATATYPE, U), H5T_NATIVE_SCALAR);

    /* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS);
    assert(acc_tpl1 != FAIL);
    /* set Parallel access with communicator */
    ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != FAIL);

    /* open the file */
    hid_t fid1=H5Fopen(mapfilename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1);
    assert(ret != FAIL);

    count[0] = NPIX;

    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

    /* create a memory dataspace independently */
    hid_t mem_dataspace = H5Screate_simple (1, count, NULL);
    assert (mem_dataspace != FAIL);

    /* set up the collective transfer properties list */
    hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
    assert(xfer_plist != FAIL);
    ret=H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);
    assert(ret != FAIL);

    /* select hyperslab */
    count[0] = numelements;
    ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, NULL,
	    count, NULL);
    assert(ret != FAIL);

    ret=H5Sselect_hyperslab(mem_dataspace, H5S_SELECT_SET, start_mem, NULL,
	    count, NULL);
    assert(ret != FAIL);

    /* write data collectively */
    ret = H5Dread(dataset, memtype, mem_dataspace, file_dataspace, xfer_plist, buffer); assert(ret != FAIL);

    for (i=0; i<numelements; i++) {
         data[0][i] = buffer[i].I;
         data[1][i] = buffer[i].Q;
         data[2][i] = buffer[i].U;
    }

    free(buffer);

    H5Sclose(file_dataspace);
    H5Sclose(mem_dataspace);
    H5Pclose(xfer_plist);

    ret=H5Dclose(dataset);
    assert(ret != FAIL);

    H5Fclose(fid1);
}

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

typedef struct {
    double THETA, PHI, PSI;
    bool FLAG;
} pointing_t;

typedef struct {
    double V;
} channel_t;

#pragma pack(pop)  /* push current alignment to stack */

void read_pointing(int MyPID, long firstelem, int numelements, pointing_t *data, const char * filename, int NumHorn)
{
    hsize_t start[1]; hsize_t count[1]; hsize_t stride[1]; hsize_t dims[1]; 
    int i, n;
    bool val;

    //printf("SIZE pointing_t:%d \n", sizeof(pointing_t));
    //printf("SIZE bool:%d \n", sizeof(bool));
    //printf("SIZE _Bool:%d \n", sizeof(_Bool));

    stride[0] = 1; count[0] = numelements;
    start[0] = firstelem;

	printf("%d read_h5_data ", MyPID);
    printf(" %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    int status;
    hid_t boolenumtype = H5Tcreate(H5T_ENUM, sizeof(bool));
    status = H5Tenum_insert(boolenumtype, "FALSE",   CPTR(val, false ));
    status = H5Tenum_insert(boolenumtype, "TRUE",   CPTR(val, true ));

    ///* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    ///* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    //printf("Open\n");
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1); assert(ret != FAIL);
    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof(pointing_t));
    char * s = NULL;
    asprintf(&s, "%s%d", "THETA", NumHorn);
    H5Tinsert (memtype, s, HOFFSET (pointing_t, THETA),  H5T_NATIVE_DOUBLE);
    s = NULL; asprintf(&s, "%s%d", "PHI", NumHorn);
    H5Tinsert (memtype, s, HOFFSET (pointing_t, PHI),  H5T_NATIVE_DOUBLE);
    s = NULL; asprintf(&s, "%s%d", "PSI", NumHorn);
    H5Tinsert (memtype, s, HOFFSET (pointing_t, PSI),  H5T_NATIVE_DOUBLE);
    s = NULL; asprintf(&s, "%s%d", "FLAG", NumHorn);
    H5Tinsert (memtype, s, HOFFSET (pointing_t, FLAG),  boolenumtype);

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
        printf("%f, %f, %f, %d\n", data[i].THETA, data[i].PHI, data[i].PSI, data[i].FLAG);

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); 
    H5Fclose(fid1);
}

void read_data(int MyPID, long firstelem, int numelements, double * data, const char * filename, char * chtag)
{
    hsize_t start[1]; hsize_t count[1]; hsize_t stride[1]; hsize_t dims[1]; 
    int i, n;
    bool val;

    //printf("SIZE pointing_t:%d \n", sizeof(pointing_t));
    //printf("SIZE bool:%d \n", sizeof(bool));
    //printf("SIZE _Bool:%d \n", sizeof(_Bool));

    stride[0] = 1; count[0] = numelements;
    start[0] = firstelem;

	printf("%d read_h5_data ", MyPID);
    printf(" %ld -" , (long)start[0]);
	printf(" %ld \n", (long)count[0]);

    ///* setup file access template */
    hid_t acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS); assert(acc_tpl1 != FAIL);
    ///* set Parallel access with communicator */
    int ret = H5Pset_fapl_mpio(acc_tpl1, MPI_COMM_WORLD, MPI_INFO_NULL); assert(ret != FAIL);

    /* open the file collectively */
    //printf("Open\n");
    hid_t fid1=H5Fopen(filename,H5F_ACC_RDONLY,acc_tpl1);

    /* Release file-access template */
    ret=H5Pclose(acc_tpl1); assert(ret != FAIL);
    /* open the dataset1 collectively */
    hid_t dataset = H5Dopen2(fid1, "data", H5P_DEFAULT); assert(dataset != FAIL);

    hid_t file_dataspace = H5Dget_space (dataset); assert(file_dataspace != FAIL);

    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof(channel_t));
    H5Tinsert (memtype, chtag, HOFFSET (channel_t, V),  H5T_NATIVE_DOUBLE);

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
        printf("%f\n", data[i]);

    H5Sclose(file_dataspace); H5Sclose(mem_dataspace); H5Pclose(xfer_plist); ret=H5Dclose(dataset); assert(ret != FAIL); 
    H5Fclose(fid1);
}
