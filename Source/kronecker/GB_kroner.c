//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// C is the ztype of the operator.  C is hypersparse if either A or B are
// hypersparse, full if both A and B are full, or sparse otherwise.  C is never
// constructed as bitmap.

#define GB_FREE_WORKSPACE       \
{                               \
    GB_Matrix_free (&Awork) ;   \
    GB_Matrix_free (&Bwork) ;   \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
}

#include "kronecker/GB_kron.h"
#include "emult/GB_emult.h"
#include "slice/include/GB_search_for_vector.h"
#include "jitifyer/GB_stringify.h"
#include "memory/include/GB_memory_macros.h"

GrB_Info GB_kroner                  // C = kron (A,B)
(
    GrB_Matrix C,                   // output matrix
    const bool C_is_csc,            // desired format of C
    const GrB_BinaryOp op,          // multiply operator
    const bool flipij,              // if true, i and j are flipped: z=(x,y,j,i)
    const GrB_Matrix A_in,          // input matrix
    bool A_is_pattern,              // true if values of A are not used
    const GrB_Matrix B_in,          // input matrix
    bool B_is_pattern,              // true if values of B are not used
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;

    struct GB_Matrix_opaque Awork_header, Bwork_header ;
    GrB_Matrix Awork = NULL, Bwork = NULL ;

    ASSERT_MATRIX_OK (A_in, "A_in for kron (A,B)", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B_in for kron (A,B)", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for kron (A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A_in) ;
    GB_MATRIX_WAIT (B_in) ;

    //--------------------------------------------------------------------------
    // bitmap case: create sparse copies of A and B if they are bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix A = A_in ;
    if (GB_IS_BITMAP (A))
    { 
        GBURBLE ("A:") ;
        GB_CLEAR_MATRIX_HEADER (Awork, &Awork_header) ;
        GB_OK (GB_dup_worker (&Awork, A->iso, A, true, NULL)) ;
        ASSERT_MATRIX_OK (Awork, "dup Awork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Awork, Werk)) ;
        ASSERT_MATRIX_OK (Awork, "to sparse, Awork for kron (A,B)", GB0) ;
        A = Awork ;
    }

    GrB_Matrix B = B_in ;
    if (GB_IS_BITMAP (B))
    { 
        GBURBLE ("B:") ;
        GB_CLEAR_MATRIX_HEADER (Bwork, &Bwork_header) ;
        GB_OK (GB_dup_worker (&Bwork, B->iso, B, true, NULL)) ;
        ASSERT_MATRIX_OK (Bwork, "dup Bwork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Bwork, Werk)) ;
        ASSERT_MATRIX_OK (Bwork, "to sparse, Bwork for kron (A,B)", GB0) ;
        B = Bwork ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;

    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;

    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnvec = B->nvec ;
    const int64_t bnz = GB_nnz (B) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    double work = ((double) anz) * ((double) bnz)
                + (((double) anvec) * ((double) bnvec)) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // compute info of the output matrix C
    //--------------------------------------------------------------------------

    // C has the same type as z for the multiply operator, z=op(x,y)

    uint64_t cvlen, cvdim, cnzmax, cnvec ;
    bool ok = GB_int64_multiply (&cvlen, avlen, bvlen) ;
    ok = ok & GB_int64_multiply (&cvdim, avdim, bvdim) ;
    ok = ok & GB_int64_multiply (&cnzmax, anz, bnz) ;
    ok = ok & GB_int64_multiply (&cnvec, anvec, bnvec) ;
    ASSERT (ok) ;

    if (C_iso)
    { 
        // the values of A and B are no longer needed if C is iso
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    // C is hypersparse if either A or B are hypersparse.  It is never bitmap.
    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;

    //--------------------------------------------------------------------------
    // get operator
    //--------------------------------------------------------------------------

    GxB_binary_function fmult = op->binop_function ;
    GxB_index_binary_function fmult_idx = op->idxbinop_function ;
    const void *theta = op->theta ;
    GB_cast_function cast_A = NULL, cast_B = NULL ;
    if (!A_is_pattern)
    { 
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
    }
    if (!B_is_pattern)
    { 
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;
    }

    //--------------------------------------------------------------------------
    // count nonzero elements in result
    //--------------------------------------------------------------------------

    bool C_is_full = GB_as_if_full (A) && GB_as_if_full (B) ;
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;

    int64_t kC ;
    GrB_Index cnz = 0 ;
    int64_t nvec_nonempty = 0 ;
    size_t p_size ;
    int64_t *restrict p = GB_MALLOC_MEMORY (cnvec + 1, sizeof(int64_t), &(p_size)) ;
    ASSERT (p_size == GB_Global_memtable_size (p)) ;
    GB_memset (p, 0, p_size, nthreads) ;
    bool p_is_32 = (ctype != GrB_INT64) ;
    #define P_IS_32 p_is_32

    size_t h_size = 0, hp_size = 0 ;
    int64_t *restrict h = NULL ;
    int64_t *restrict hp = NULL ;
    
    struct GB_Matrix_opaque stub_header ;
    GrB_Matrix C_stub = &stub_header ; 
    C_stub->magic = GB_MAGIC ;
    C_stub->type = ctype ;              
    C_stub->vlen = cvlen ;               
    C_stub->vdim = cvdim ;               
    C_stub->nvec = cnvec ;
    C_stub->plen = cnzmax ;    
    C_stub->nvals = 0 ;    
    C_stub->nvec_nonempty = 0 ;     
    C_stub->is_csc = C_is_csc ;          
    C_stub->sparsity_control = C_is_hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;
    C_stub->p = p ;                      
    C_stub->h = h ;  
    // На этапе подсчета jit параметр C_stub->i не используется
    // Мы временно меняем его, чтобы передать указатель hp jit         
    C_stub->i = hp ; 
    C_stub->x = NULL ;        
    C_stub->iso = false ;  
    C_stub->jumbled = (void*)(fmult) == NULL ;

        // via the JIT kernel
    info = GB_kroner_sel_jit (C_stub, op, flipij, A, B, nthreads) ;
    
    if (info == GrB_SUCCESS) 
    { 
        cnz = C_stub->nvals;
        nvec_nonempty = C_stub->nvec_nonempty;
        hp = (int64_t *) C_stub->i;
    }
    
    fprintf(stderr, "[DEBUG] JIT counting: info=%d, GrB_NO_VALUE=%d, cnz=%ld\n", 
        info, (int)GrB_NO_VALUE, (long)cnz) ;
    if (info == GrB_NO_VALUE)
    { 
        fprintf(stderr, "[DEBUG] Using GENERIC kernel\n");
        // via the generic kernel
        #define GB_A_TYPE GB_void
        #define GB_B_TYPE GB_void
        #define GB_C_TYPE GB_void
        #define GB_P_TEMP p
        #define GB_H_TEMP h
        #define GB_HP_TEMP hp
        #define GB_A_ISO A_iso
        #define GB_B_ISO B_iso
        #define GB_C_ISO C_iso
        #define GB_NVEC_NONEMPTY_PTR &nvec_nonempty
        const bool A_iso = A->iso ;
        const bool B_iso = B->iso ;
        const int64_t asize = A->type->size ;
        const int64_t bsize = B->type->size ;

        #define GB_C_IS_FULL C_is_full
        #define GB_C_IS_HYPER C_is_hyper
        #define OP_IS_POSITIONAL ((void*)(fmult) == NULL)

        #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
        #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

        #define GB_GETA(a,Ax,p,iso)                         \
        {                                                   \
            if (!A_is_pattern)                              \
            {                                               \
                cast_A (a, Ax + (p)*asize, asize) ;         \
            }                                               \
        }

        #define GB_GETB(b,Bx,p,iso)                         \
        {                                                   \
            if (!B_is_pattern)                              \
            {                                               \
                cast_B (b, Bx + (p)*bsize, bsize) ;         \
            }                                               \
        }

        #define GB_KRONECKER_COUNTER(c,a,b)                 \
        {                                                   \
                /* standard binary operator */              \
                fmult (c, a, b) ;                           \
                for (size_t i = 0 ; i < csize ; ++i)        \
                {                                           \
                    if (*(c +  i))                          \
                    {                                       \
                        CNZ_INC() ;                         \
                        p [kC]++ ;                          \
                        break ;                             \
                    }                                       \
                }                                           \
        }

        #define GB_GENERIC
        #include "ewise/include/GB_ewise_shared_definitions.h"
        #include "kronecker/template/GB_kroner_sel_template.c"
        info = GrB_SUCCESS ;
    } 
    else 
    {
        fprintf(stderr, "[DEBUG] Using JIT kernel (info=%d)\n", info);
    }

    fprintf(stderr, "[DEBUG] op->ztype->code=%d, fmult=%p, idxbinop=%p\n",
        op->ztype->code, (void*)op->binop_function, (void*)op->idxbinop_function);
    //--------------------------------------------------------------------------
    // quick return if C is empty
    //--------------------------------------------------------------------------

    if (cnz == 0)
    { 
        GB_FREE_MEMORY(&p, p_size) ;
        if (h != NULL) GB_FREE_MEMORY(&h, h_size) ;
        if (hp != NULL) GB_FREE_MEMORY(&hp, hp_size) ;
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int C_sparsity = C_is_full ? GxB_FULL :
        ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, cnz, (int64_t) cvlen, (int64_t) cvdim, Werk) ;

    if (C_is_hyper)
    {
        GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_ph_malloc, C_is_csc,
        C_sparsity, true, B->hyper_switch, nvec_nonempty, cnz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;
    }
    else
    {
        GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_ph_malloc, C_is_csc,
        C_sparsity, true, B->hyper_switch, cnvec, cnz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;
    }

    //--------------------------------------------------------------------------
    // compute the column counts of C: Cp and Ch if C is hypersparse
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    #define GB_Cp_IS_32 Cp_is_32

    if (!C_is_full)
    {
        if (C_is_hyper)
        { 
            C->nvec = nvec_nonempty ;
            GB_nvec_nonempty_set (C, nvec_nonempty) ;

            for (int64_t i = 0; i < nvec_nonempty; i++) 
            {
                GB_ISET (Ch, i, h[i]); 
            }
            GB_FREE_MEMORY (&h, h_size) ;

            for (int64_t i = 0; i <= nvec_nonempty; i++) 
            {
                GB_ISET (Cp, i, hp[i]) ;
            }
            C->nvals = GB_IGET (Cp, nvec_nonempty) ;
            GB_FREE_MEMORY (&hp, hp_size) ;
        }
        else
        { 
            for (int64_t i = 0; i <= cnvec; i++) 
            {
                GB_ISET (Cp, i, p[i]) ;
            }
            C->nvals = GB_IGET (Cp, cnvec) ;
        }
    }
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and/or full full
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        // C->x [0] = cscalar = op (A,B)
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        { 
            // no more work to do if C is iso and full
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_WORKSPACE ;
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    // via the JIT kernel
    info = GB_kroner_jit (C, op, flipij, A, B, nthreads) ;

    fprintf(stderr, "[DEBUG] JIT fill: info=%d\n", info);
    if (info == GrB_NO_VALUE)
    { 
        // via the generic kernel
        #define GB_A_TYPE GB_void
        #define GB_B_TYPE GB_void
        #define GB_C_TYPE GB_void
        #define GB_A_ISO A_iso
        #define GB_B_ISO B_iso
        #define GB_C_ISO C_iso
        const bool A_iso = A->iso ;
        const bool B_iso = B->iso ;
        const int64_t asize = A->type->size ;
        const int64_t bsize = B->type->size ;

        #define GB_C_IS_FULL C_is_full

        #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
        #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

        #define GB_GETA(a,Ax,p,iso)                         \
        {                                                   \
            if (!A_is_pattern)                              \
            {                                               \
                cast_A (a, Ax + (p)*asize, asize) ;         \
            }                                               \
        }

        #define GB_GETB(b,Bx,p,iso)                         \
        {                                                   \
            if (!B_is_pattern)                              \
            {                                               \
                cast_B (b, Bx + (p)*bsize, bsize) ;         \
            }                                               \
        }

        #define GB_KRONECKER_OP(Cx,pC,a,ix,jx,b,iy,jy)      \
        {                                                   \
            GB_C_TYPE cwork[GB_VLA(csize)] ;                \
            bool is_nonzero = false ;                       \
            if (fmult != NULL)                              \
            {                                               \
                /* standard binary operator */              \
                fmult (cwork, a, b) ;                       \
                for (size_t i = 0 ; i < csize ; ++i)        \
                {                                           \
                    if (*(cwork + i))                       \
                    {                                       \
                        is_nonzero = true ;                 \
                        break ;                             \
                    }                                       \
                }                                           \
            }                                               \
            else                                            \
            {                                               \
                /* index binary operator */                 \
                if (flipij)                                 \
                {                                           \
                    fmult_idx (cwork,                       \
                        a, jx, ix, b, jy, iy, theta) ;      \
                }                                           \
                else                                        \
                {                                           \
                    fmult_idx (cwork,                       \
                        a, ix, jx, b, iy, jy, theta) ;      \
                }                                           \
                is_nonzero = true ;                         \
            }                                               \
            if (is_nonzero)                                 \
            {                                               \
                memcpy(Cx +(pC)*csize, cwork, csize) ;      \
                pC++ ;                                      \
            }                                               \
        }

        #define GB_GENERIC
        #include "ewise/include/GB_ewise_shared_definitions.h"
        #include "kronecker/template/GB_kroner_template.c"
        info = GrB_SUCCESS ;
    }

    GB_FREE_MEMORY (&p, p_size) ;

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        GB_OK (GB_hyper_prune (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (info) ;
}

