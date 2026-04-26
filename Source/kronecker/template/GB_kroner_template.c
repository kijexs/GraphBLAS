//------------------------------------------------------------------------------
// GB_kroner_template: Kronecker product, C = kron (A,B)
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

    GB_Cp_DECLARE (Cp,      ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch,      ) ; GB_Ch_PTR (Ch, C) ;
    GB_C_NVALS (cnz) ;
    const int64_t cnvec = anvec * bnvec ;
    const int64_t nvec  = C->nvec ;
    const int64_t cvlen = C->vlen ;
    #define P_PTR  ((int64_t *) (C)->p)
    #else
    int64_t nvec = C->nvec ;
    #define P_PTR (p)
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

    #pragma omp parallel for num_threads(nthreads) schedule(guided)
    for (int64_t kC = 0 ; kC < cnvec ; kC++)
    { 
        int64_t kA = kC / bnvec ;
        int64_t kB = kC % bnvec ;

        // get B(:,jB), the (kB)th vector of B
        int64_t jB       = GBh_B (Bh, kB) ;
        int64_t pB_start = GBp_B (Bp, kB, bvlen) ;
        int64_t pB_end   = GBp_B (Bp, kB+1, bvlen) ;
        int64_t bknz     = pB_end - pB_start ;
        if (bknz == 0) continue ;

        // get C(:,jC), the (kC)th vector of C
        int64_t pC     = P_PTR [kC] ;
        int64_t pC_end = P_PTR [kC+1] ;

        // get A(:,jA), the (kA)th vector of A
        int64_t jA = GBh_A (Ah, kA) ;
        int64_t pA_start = GBp_A (Ap, kA, avlen) ;
        int64_t pA_end   = GBp_A (Ap, kA+1, avlen) ;

        //----------------------------------------------------------------------
        // get the iso values of A and B
        //----------------------------------------------------------------------

        GB_DECLAREA (a) ;
        if (GB_A_ISO)
        { 
            GB_GETA (a, Ax, 0, true) ;
        }
        GB_DECLAREB (b) ;
        if (GB_B_ISO)
        { 
            GB_GETB (b, Bx, 0, true) ;
        }

        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
        { 
            //--------------------------------------------------------------
            // a = A(iA,jA), typecasted to op->xtype
            //--------------------------------------------------------------

            int64_t iA = GBi_A (Ai, pA, avlen) ;
            int64_t iAblock = iA * bvlen ;
            if (!GB_A_ISO)
            { 
                GB_GETA (a, Ax, pA, false) ;
            }

            for (int64_t pB = pB_start ; pB < pB_end && pC < pC_end ; pB++)
            { 
                //--------------------------------------------------------------
                // b = B(iB,jB), typecasted to op->ytype
                //--------------------------------------------------------------

                int64_t iB = GBi_B (Bi, pB, bvlen) ;
                if (!GB_B_ISO) 
                { 
                    GB_GETB (b, Bx, pB, false) ;
                }
                // C(iC,jC) = A(iA,jA) * B(iB,jB)
                if (!GB_C_IS_FULL)
                { 
                    GB_ISET (Ci, pC, iAblock + iB) ;
                }
                if (!GB_C_ISO)
                { 
                    GB_KRONECKER_OP (Cx, pC, a, iA, jA, b, iB, jB) ;
                }
                else
                { 
                    pC++ ;
                }
            }
        }
    }
}
