#ifndef READ_H5_H
#define READ_H5_H

#define FAIL -1

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

typedef struct {
    long L;
    double D0, D1, D2, D3;
} thin_data_struct;

#pragma pack(pop)  /* push current alignment to stack */
void read_hdf5_thin(int pid, long first_elem, int num_elements, thin_data_struct *data, const char * filename);
void write_hdf5_thin(int pid, long first_elem, int num_elements, thin_data_struct *data, const char * filename);

#endif
