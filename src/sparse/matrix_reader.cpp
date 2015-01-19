#include <hdf5.h>
#include <cassert>
#include <hdf5_hl.h>
#include "./matrix_reader.h"
#include "./mmio.h"

namespace jusha {

  void Hdf5Matrix::read_matrix(const char *file_name) {
    // TODO
    assert(0);

    // hid_t       fid;
    // hsize_t dims[1];
    // //  int RANK=1;
    // herr_t status;
    // int nrows, ncols;
    // fid = H5Fopen (file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    // status = H5Fclose (fid);
  }


  void MatrixMarket::read_matrix(const char *file_name) {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if ((f = fopen(file_name, "r")) == NULL)
      assert(0);
  

    if (mm_read_banner(f, &matcode) != 0)
      {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
      }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode) )
      {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
      }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
      {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
      }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
      fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
  }
  
}
