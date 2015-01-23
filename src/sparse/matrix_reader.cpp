
#include <cassert>
#include <map>
//#include <hdf5.h>
//#include <hdf5_hl.h>
#include "./matrix_reader.h"
#include "./mmio.h"

namespace jusha {
  
  void coo_to_csr_rows(const std::vector<int> coo_rows, const int &num_rows, std::vector<int> &row_ptrs)
  {
    // not the most efficient implementation but clear
    const int invalid_flag = -1;
    row_ptrs.resize(num_rows + 1);
    std::fill(row_ptrs.begin(), row_ptrs.end(), invalid_flag);
    
    size_t nnz = coo_rows.size();
    int last_row = -1;
    for (size_t i = 0; i != nnz; i++) {
      int cur_row = coo_rows[i];
      if (cur_row  != last_row) 
        row_ptrs[cur_row] = static_cast<int>(i);
      last_row = cur_row;
    }
    row_ptrs[num_rows] = nnz;

    // handle zero rows
    for (int i = num_rows; i >= 0; --i) {
      if (row_ptrs[i] == invalid_flag)
        row_ptrs[i] = row_ptrs[i+1];
      assert(row_ptrs[i] != invalid_flag);
    }
  }
  
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
     int i;//, *I(NULL); //, *J;
    //    double *val;

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
    int _num_rows, _num_cols, _num_nnzs;
    if ((ret_code = mm_read_mtx_crd_size(f, &_num_rows, &_num_cols, &_num_nnzs)) !=0)
      exit(1);

    // convert to 64 bit
    m_num_rows = _num_rows;
    m_num_cols = _num_cols;
    m_num_nnzs = _num_nnzs;
    /* reseve memory for matrices */

    //    I = (int *) malloc(nz * sizeof(int));
    //    J = (int *) malloc(nz * sizeof(int));
    //    val = (double *) malloc(nz * sizeof(double));
    m_rows.resize(m_num_nnzs);
    m_cols.resize(m_num_nnzs);
    m_coefs.resize(m_num_nnzs);


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    std::vector<std::map<int64_t,double> > elements;
    elements.resize(m_num_rows);
    
    for (i=0; i<m_num_nnzs; i++)
      {
        fscanf(f, "%d %zd %lg\n", &(m_rows[i]), &(m_cols[i]), &(m_coefs[i]));
        m_rows[i]--;  /* adjust from 1-based to 0-based */
        m_cols[i]--;

        elements[m_rows[i]][m_cols[i]] += m_coefs[i];
      }

    // now sort by rows
    std::vector<int> m_rows2;
    std::vector<int64_t> m_cols2;
    std::vector<double> m_coefs2;

    m_rows2.resize(m_num_nnzs);
    m_cols2.resize(m_num_nnzs);
    m_coefs2.resize(m_num_nnzs);
    int index=0;
    
    for (int row=0; row<m_num_rows; row++) {
      int nc = (int)elements[row].size();
      assert (nc > 0);

      // Make sure diagonal element is first
      if (elements[row].find(row) != elements[row].end()) {
        m_cols2[index]  = row;
        m_rows2[index] = row;        
        m_coefs2[index] = elements[row][row];
        assert (m_coefs2[index] != 0.0);
        index++;
      }
      else
        assert(0);

      for (auto iter = elements[row].begin(); iter != elements[row].end(); iter++) {
        int col = iter->first;
        if (col != row) {
          double C = iter->second;
          assert(col < m_num_cols);
          m_rows2[index] = row;
          m_cols2[index] = col;
          m_coefs2[index] = C;
          index++;
        }
      }

    }
    
    if (f !=stdin) fclose(f);

    m_rows.swap(m_rows2);
    m_cols.swap(m_cols2);
    m_coefs.swap(m_coefs2);    


    /* convert coo to csr row ptrs */
    coo_to_csr_rows(m_rows, m_num_rows, m_row_ptrs);

    // /************************/
    // /* now write out matrix */
    // /************************/

    //    mm_write_banner(stdout, matcode);
    //    mm_write_mtx_crd_size(stdout, m_num_rows, m_num_cols, m_num_nnzs);
    // for (i=0; i<nz; i++)
    //   fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
    //    free(I);
  }
  
}
