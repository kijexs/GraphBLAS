//------------------------------------------------------------------------------
// GB_kroner_mask_template: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// C is the ztype of the operator.  C is hypersparse if either A or B are
// hypersparse, full if both A and B are full, or sparse otherwise.  C is never
// constructed as bitmap.  This template does not need access to C->h if C
// is hypersparse, so it works the same if C is sparse or hypersparse.

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    const int64_t avlen = A->vlen ;
    const int64_t anvec = A->nvec ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    const int64_t bvlen = B->vlen ;
    const int64_t bnvec = B->nvec ;
    const int64_t bvdim = B->vdim ;

    GB_Cp_DECLARE (Cp,      ) ; GB_Cp_PTR (Cp, C) ;
    GB_C_NVALS (cnz) ;
    const int64_t cnvec = C->nvec ;
    const int64_t cvlen = C->vlen ;
    #define P_PTR (C->p)
    #define P_IS_32 (C->p_is_32)
    const int64_t csize = C->type->size ;
    #define CX_PTR(Cx, pC) ((void *)((Cx) + (pC)))
    #else
    #define CX_PTR(Cx, pC) ((void *)((Cx) + (pC)*csize))
    #endif

    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    GB_Ci_DECLARE (Ci,      ) ; GB_Ci_PTR (Ci, C) ;

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    #define GB_GETVECTOR(A, row, col) GrB_Index vector_ = (A)->is_csc ? (col) : (row) 

    #define GB_GETCOORD(A, row, col) GrB_Index coord_ = (A)->is_csc ? (row) : (col)

    #define GB_LOOKUP_XOFFSET_BITMAP_FULL(offset, A)    \
    {                                                   \
        GrB_Index offset_ = vector_ * (A)->vlen +       \
                                coord_ ;                   \
        if ((A)->b == NULL ||                           \
            (((int8_t *) (A)->b)[offset_]))             \
        {                                               \
            offset = (A)->iso ? 0 : offset_ ;           \
        }                                               \
        else                                            \
        {                                               \
            offset = -1 ;                               \
        }                                               \
    }

    #define GB_LOOKUP_XOFFSET_SPARSE(offset, A)         \
    {                                                   \
        start_ = (A)->p_is_32 ?                         \
                    ((int32_t *)(A)->p) [vector_] :       \
                    ((int64_t *)(A)->p) [vector_] ;       \
        end_ = (A)->p_is_32 ?                           \
                ((int32_t *)(A)->p) [vector_ + 1] :     \
                ((int64_t *)(A)->p) [vector_ + 1] ;     \
        end_-- ;                                        \
        if (start_ > end_)                              \
        {                                               \
            offset = -1 ;                               \
        }                                               \
        else                                            \
        {                                               \
            if (GB_binary_search (coord_, (A)->i,       \
                (A)->i_is_32, &start_, &end_))          \
            {                                           \
                offset = (A)->iso ? 0 : start_ ;        \
            }                                           \
            else                                        \
            {                                           \
                offset = - 1 ;                          \
            }                                           \
        }                                               \
    }

    #define GB_LOOKUP_XOFFSET_HYPER(offset, A)          \
    {                                                   \
        start_ = 0 ;                                    \
        end_ = (A)->plen - 1 ;                          \
        if (!GB_binary_search (vector_, (A)->h,         \
            (A)->j_is_32, &start_, &end_))              \
        {                                               \
            offset = - 1;                               \
        }                                               \
        else                                            \
        {                                               \
            int64_t k_ = start_ ;                       \
            start_ = (A)->p_is_32 ?                     \
                        ((int32_t *)(A)->p) [k_] :        \
                        ((int64_t *)(A)->p) [k_] ;        \
            end_ = (A)->p_is_32 ?                       \
                    ((int32_t *)(A)->p) [k_ + 1] :      \
                    ((int64_t *)(A)->p) [k_ + 1] ;      \
            end_-- ;                                    \
            if (start_ > end_)                          \
            {                                           \
                offset = -1 ;                           \
            }                                           \
            else                                        \
            {                                           \
                if (GB_binary_search (coord_, (A)->i,   \
                    (A)->i_is_32, &start_, &end_))      \
                {                                       \
                    offset = (A)->iso ? 0 : start_ ;    \
                }                                       \
                else                                    \
                {                                       \
                    offset = -1 ;                       \
                }                                       \
            }                                           \
        }                                               \
    }

    #define GB_LOOKUP_XOFFSET(offset, A, row, col)      \
    {                                                   \
        GB_GETVECTOR(A, row, col) ;                     \
        GB_GETCOORD(A, row, col) ;                      \
        if ((A)->p == NULL)                             \
        {                                               \
            GB_LOOKUP_XOFFSET_BITMAP_FULL (offset, A) ; \
        }                                               \
        else                                            \
        {                                               \
            int64_t start_, end_ ;                      \
            if ((A)->h == NULL)                         \
            {                                           \
                GB_LOOKUP_XOFFSET_SPARSE(offset, A) ;   \
            }                                           \
            else                                        \
            {                                           \
                GB_LOOKUP_XOFFSET_HYPER(offset, A) ;    \
            }                                           \
        }                                               \
    }

    #define GB_NROWS(A) ((A)->is_csc ? (A)->vlen : (A)->vdim)
    #define GB_NCOLS(A) ((A)->is_csc ? (A)->vdim : (A)->vlen)


    GB_Mp_DECLARE(Mp, ) ;
    GB_Mp_PTR(Mp, Mask) ;

    GB_Mh_DECLARE(Mh, ) ;
    GB_Mh_PTR(Mh, Mask) ;

    GB_Mi_DECLARE(Mi, ) ;
    GB_Mi_PTR(Mi, Mask) ;

    int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;

    #pragma omp parallel num_threads(nthreads)
    {
        GrB_Index offset = -1 ;
        GB_DECLAREA (a_elem) ;
        GB_DECLAREB (b_elem) ;
        int64_t vlen = Mask->vlen ;

        if (GB_A_ISO)
        { 
            GB_GETA (a_elem, Ax, 0, true) ;
        }

        if (GB_B_ISO)
        { 
            GB_GETB (b_elem, Bx, 0, true) ;
        }

        #pragma omp for schedule(static)
        for (GrB_Index k = 0 ; k < Mask->nvec ; k++)
        {
            GrB_Index j = GBh_M (Mh, k) ;

            #ifdef GB_JIT_KERNEL
                int64_t pC = GBp_C(Cp, k, cvlen);
                int64_t pC_end = GBp_C(Cp, k + 1, cvlen);
            #else
                int64_t pC = P_IS_32 ?
                    ((const uint32_t *) P_PTR)[k] :
                    ((const uint64_t *) P_PTR)[k];

                int64_t pC_end = P_IS_32 ?
                    ((const uint32_t *) P_PTR)[k + 1] :
                    ((const uint64_t *) P_PTR)[k + 1];
            #endif

            int64_t pA_start = GBp_M (Mp, k, vlen) ;
            int64_t pA_end = GBp_M (Mp, k+1, vlen) ;

            for (GrB_Index p = pA_start; p < pA_end  && pC < pC_end ; p++)
            {
                if (!GBb_M (Mask->b, p)) continue ;

                int64_t i = GBi_M (Mi, p, vlen) ;
                GrB_Index Mrow = Mask->is_csc ? i : j ; GrB_Index Mcol = Mask->is_csc ? j : i ;

                // extract elements from A and B, 
                // initialize offset in MTi and MTx,
                // get result of op, place it in MTx

                if (GB_MASK_STRUCT || (Mask->iso ? ((bool *)Mask->x)[0] : ((bool *)Mask->x)[p]))
                {
                    GrB_Index arow = A_transpose ? (Mcol / bncols) : (Mrow / bnrows);
                    GrB_Index acol = A_transpose ? (Mrow / bnrows) : (Mcol / bncols);

                    GrB_Index brow = B_transpose ? (Mcol % bncols) : (Mrow % bnrows);
                    GrB_Index bcol = B_transpose ? (Mrow % bnrows) : (Mcol % bncols);

                    GB_LOOKUP_XOFFSET (offset, A, arow, acol) ;
                    if (offset == -1)
                    {
                        continue;
                    }

                    if (!GB_A_ISO)
                    {
                        GB_GETA (a_elem, Ax, offset, GB_A_ISO) ;
                    }

                    GB_LOOKUP_XOFFSET (offset, B, brow, bcol) ;
                    if (offset == -1)
                    {
                        continue;
                    }
                    if (!GB_B_ISO)
                    {
                        GB_GETB (b_elem, Bx, offset, GB_B_ISO) ;
                    }
                    
                    GrB_Index ix, jx, iy, jy ;
                    ix = A_transpose ? acol : arow ;
                    jx = A_transpose ? arow : acol ;
                    iy = B_transpose ? bcol : brow ;
                    jy = B_transpose ? brow : bcol ;

                    // compute C(iC,jC) = A(iA,jA) * B(iB,jB) into a temporary buffer
                    if (!GB_C_ISO) {
                        // if (flipij)
                        // {
                        //     GB_KRONECKER_OP (Cx, pC, a_elem, jx, ix, b_elem, jy, iy) ;
                        // }
                        // else 
                        // {
                            GB_KRONECKER_OP (Cx, pC, a_elem, ix, jx, b_elem, iy, jy) ;
                        // }
                    }
                    
                    //--------------------------------------------------------------
                    // check if C(iC,jC) should be retained
                    //--------------------------------------------------------------
                    
                    int64_t iC = i ;
                    int64_t jC = j ;
                    
                    // user-defined selector: call the function 
                    // to check the value
                    
                    #ifdef GB_HAS_SELECTOR
                        GB_TEST_KRON_VALUE_OF_ENTRY (result, CX_PTR(Cx, GB_C_ISO ? 0 : pC)) ;
                        if (!result) continue;
                    #endif
                    
                    // C(iC,jC) = A(iA,jA) * B(iB,jB)
                    if (!GB_C_IS_FULL)
                    { 
                        GB_ISET (Ci, pC, iC) ;
                    }
                    
                    pC++ ;
                }
            }
        }
    }
}

