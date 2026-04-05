//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/kron_demo.c: Kronkecker product
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reads two graphs from two files and computes their Kronecker product,
// C = kron (A,B), writing the result to a file.
//
//  kron_demo A.tsv B.tsv C.tsv
//
// Where A.tsv and B.tsv are two tab- or space-delimited triplet files with
// 1-based indices.  Each line has the form:
//
//  i j x
//
// where A(i,j)=x is performed by GrB_Matrix_build, to construct the matrix.
// The dimensions of A and B are assumed to be the largest row and column
// indices that appear in the files.  The file C.tsv is the filename of the
// output file for C=kron(A,B), also with 1-based indices.

#include "graphblas_demos.h"

//------------------------------------------------------------------------------
// include read_matrix.c after temporarily disabling our FREE_ALL macro
// (so that read_matrix.c can define its own FREE_ALL internally)
//------------------------------------------------------------------------------

#undef FREE_ALL
#include "read_matrix_kron.c"

GrB_Info read_matrix_kron
(
    GrB_Matrix *A_output,
    FILE *f,
    bool make_symmetric,
    bool no_self_edges,
    bool one_based,
    bool boolean,
    bool pr
);
// re-define FREE_ALL for this demo
#define FREE_ALL                            \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&B) ;                  \
    GrB_Matrix_free (&C) ;                  \
    if (Afile != NULL) fclose (Afile) ;     \
    if (Bfile != NULL) fclose (Bfile) ;     \
    if (Cfile != NULL) fclose (Cfile) ;     \
    if (I != NULL) free (I) ;               \
    if (J != NULL) free (J) ;               \
    if (X != NULL) free (X) ;               \
    GrB_finalize ( ) ;

static int need_size = 0;
static uint64_t n_mask_64, t_mask_64 ;
static uint8_t n_mask8, t_mask8;
static uint16_t n_mask16, t_mask16;
static uint32_t n_mask32, t_mask32;

void set_kron_params(int n_bits, int t_bits) 
{   
    uint64_t n_mask = ((1ULL << n_bits) - 1) << t_bits;
    uint64_t t_mask = ~n_mask;
    int total_bits = n_bits + t_bits;
    switch (total_bits) 
    {
        case 8:  
        {
            need_size = 8; 
            n_mask8 = (uint8_t)n_mask;
            t_mask8 = (uint8_t)t_mask ;
            break;
        }
        case 16: 
        {
            need_size = 16; 
            n_mask16 = (uint16_t)n_mask;
            t_mask16 = (uint16_t)t_mask;
            break;

        }
        case 32: 
        {
            need_size = 32; 
            n_mask32 = (uint32_t)n_mask;
            t_mask32 = (uint32_t)t_mask;
            break;
        }
        default: 
        {
            need_size = 64; 
            n_mask_64 = n_mask;
            t_mask_64 = t_mask;
            break;
        }
    }
}

void my_fun(void *z, const void *x, const void *y) 
{
    double a_double = *(const double*)x;
    double b_double = *(const double*)y;
    bool result = true ;

    switch (need_size)
    {
    case 8: 
    {
        uint8_t a = (uint8_t)a_double;
        uint8_t b = (uint8_t)b_double;
        uint8_t a_term = a & t_mask8;
        uint8_t b_term = b & t_mask8;
        result = ((a & b & n_mask8) != 0) || 
                 ((a_term == b_term) && (a_term != 0));
        break;
    }      
    case 16: 
    {
        uint16_t a = (uint16_t)a_double;
        uint16_t b = (uint16_t)b_double;
        uint16_t a_term = a & t_mask16;
        uint16_t b_term = b & t_mask16;
        result = ((a & b & n_mask16) != 0) || 
                 ((a_term == b_term) && (a_term != 0));
        break;
    }
    case 32: 
    {
        uint32_t a = (uint32_t)a_double;
        uint32_t b = (uint32_t)b_double;
        uint32_t a_term = a & t_mask32;
        uint32_t b_term = b & t_mask32;
        result = ((a & b & n_mask32) != 0) || 
                 ((a_term == b_term) && (a_term != 0));
        break;
    }
    default: 
    {
        uint64_t a = (uint64_t)a_double;
        uint64_t b = (uint64_t)b_double;
        uint64_t a_term = a & t_mask_64;
        uint64_t b_term = b & t_mask_64;
        result = ((a & b & n_mask_64) != 0) || 
                 ((a_term == b_term) && (a_term != 0));
        break;
    }
    }
    
    *(bool*)z = result;
}

int main (int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Index *I = NULL, *J = NULL ;
    FILE *Afile = NULL, *Bfile = NULL, *Cfile = NULL, *Dfile =NULL;
    bool *X = NULL ;
    GrB_Info info ;

    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "kron demo: nthreads %d\n", nthreads) ;

    if (argc != 5)
    {
        FREE_ALL ;
        fprintf (stderr, "usage: kron_demo A.csv B.csv types.txt C.csv\n") ;
        exit (1) ;
    }

    Afile = fopen (argv [1], "r") ;
    Bfile = fopen (argv [2], "r") ;
    Dfile = fopen (argv [3], "r") ;
    Cfile = fopen (argv [4], "w") ;
    
    if (Afile == NULL || Bfile == NULL || Cfile == NULL)
    {
        FREE_ALL ;
        fprintf (stderr, "unable to read input files or create output file\n") ;
        exit (1) ;
    }

    //--------------------------------------------------------------------------
    // get A and B from input files
    //--------------------------------------------------------------------------

    // this would be faster and take less memory if GraphBLAS had a built-in
    // read-from-file operation
    OK (read_matrix_kron (&A, Afile, false, false, false, false, false)) ;
    OK (read_matrix_kron (&B, Bfile, false, false, false, false, false)) ;
    int n_bits, t_bits ;
    fscanf(Dfile, "%d %d", &n_bits, &t_bits) ;
    set_kron_params(n_bits, t_bits) ;

    fclose (Afile) ;
    fclose (Bfile) ;
    fclose (Dfile) ;
    Afile = NULL ;
    Bfile = NULL ;
    Dfile = NULL ;

    GrB_Index anrows, ancols, bnrows, bncols, anvals, bnvals ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    OK (GrB_Matrix_nvals (&bnvals, B)) ;

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_BOOL, anrows * bnrows, ancols * bncols)) ;

    GxB_binary_function n_mask_ch = (GxB_binary_function)my_fun;
    GrB_BinaryOp grb_mask_check;
    GrB_BinaryOp_new (&grb_mask_check, n_mask_ch, GrB_BOOL, GrB_FP64, GrB_FP64);

    OK (GrB_Matrix_kronecker_BinaryOp (C, NULL, NULL,
        grb_mask_check, A, B, NULL)) ;

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&B)) ;

    //--------------------------------------------------------------------------
    // report results
    //--------------------------------------------------------------------------

    GrB_Index cnrows, cncols, cnvals ;
    OK (GrB_Matrix_nrows (&cnrows, C)) ;
    OK (GrB_Matrix_ncols (&cncols, C)) ;
    OK (GrB_Matrix_nvals (&cnvals, C)) ;

    // note that integers of type GrB_Index should be printed with the
    // %PRIu64 format.

    fprintf (stderr, "GraphBLAS GrB_kronecker:\n"
    "A: %" PRIu64 "-by-%" PRIu64 ", %" PRIu64 " entries.\n"
    "B: %" PRIu64 "-by-%" PRIu64 ", %" PRIu64 " entries.\n"
    "C: %" PRIu64 "-by-%" PRIu64 ", %" PRIu64 " entries.\n",
    anrows, ancols, anvals,
    bnrows, bncols, bnvals,
    cnrows, cncols, cnvals) ;

    //--------------------------------------------------------------------------
    // write C to the output file
    //--------------------------------------------------------------------------

    // this would be faster and take less memory if GraphBLAS had a built-in
    // write-to-file operation

    I = (GrB_Index *) malloc ((cnvals+1) * sizeof (GrB_Index)) ;
    J = (GrB_Index *) malloc ((cnvals+1) * sizeof (GrB_Index)) ;
    X = (bool    *) malloc ((cnvals+1) * sizeof (bool   )) ;
    if (I == NULL || J == NULL || X == NULL)
    {
        fprintf (stderr, "out of memory\n") ;
        FREE_ALL ;
        exit (1) ;
    }

    OK (GrB_Matrix_extractTuples_BOOL (I, J, X, &cnvals, C)) ;

    for (int64_t k = 0 ; k < cnvals ; k++)
    {
        fprintf (Cfile, "%" PRIu64 ",%" PRIu64 "\n",
            I [k], J [k]) ;
    }

    GrB_BinaryOp_free(&grb_mask_check) ;
    FREE_ALL ;
    return (0) ;
}
