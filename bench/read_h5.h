#ifndef READ_H5_H
#define READ_H5_H

#define FAIL -1

#include "ctypedef.h"

typedef struct {
    Scalar  I;
    Scalar  Q;
    Scalar  U;
} MAPDATATYPE;                                 /* Compound type */

typedef struct {
    Scalar  II;
    Scalar  IQ;
    Scalar  IU;
    Scalar  QQ;
    Scalar  QU;
    Scalar  UU;
} MDATATYPE;

int compute_weights(InputScalar psi, float * qw, float *uw) ;

int read_h5_pids(long ** datap, int* NumPIDs, const char *filename);

int read_h5_vec(const char *filename, long total_length, long firstelem, int numelements, Scalar *buffer);

int read_h5_data(int MyPID, long firstelem, int numelements, InputScalar **data, const char *filename, int NumChannels);


int read_h5_data_col(int MyPID, long firstelem, int numelements, InputScalar *data, const char *filename, int NumChannel);

int write_h5_M(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX);

int write_h5_vec(const char *mapfilename, long total_length, long firstelem, int numelements, Scalar *buffer);

int write_h5_map(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX);

int read_h5_map(const char *mapfilename, long firstelem, int numelements, Scalar **data, int NPIX);

int c_bin_baselines(double * baselines, int * pix, int * baseline_lengths, double * tmap, int num_bas, int num_pix);

int c_bin_baselines_mp(double * baselines, int * pix, int * baseline_offsets, double * tmaps, int num_bas, int num_pix);

int get_num_threads();

int c_sigremove_mp(double * baselines, double * output, int * pix, int * baseline_offsets, int * baseline_lenghts, double * tmap, int num_bas);

int c_bin_cal_mp(double * baselines, double * data, double * dipole, double * gain, int * pix, int * baseline_offsets, double * tmaps, int num_bas, int num_pix);

int c_scanmap_mp(double * map, int * pix, double * out, int numelements);

#endif
