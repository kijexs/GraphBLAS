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

    if (GB_C_IS_HYPER)
    { 
        //----------------------------------------------------------------------
        // hypersparse C: parallelize over physical vectors (kC)
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (int64_t kC = 0 ; kC < cnvec ; kC++)
        { 
            // logical vector index in C
            int64_t kA = kC / bnvec ;
            int64_t kB = kC % bnvec ;

            // get B(:,jB), the (kB)th vector of B
            int64_t jB       = GBh_B (Bh, kB) ;
            int64_t pB_start = GBp_B (Bp, kB, bvlen) ;
            int64_t pB_end   = GBp_B (Bp, kB+1, bvlen) ;
            int64_t bknz     = pB_end - pB_start ;
            if (bknz == 0) continue ;

            // get C(:,jC) physical pointers
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
    else
    {
        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {

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

            //----------------------------------------------------------------------
            // construct the task to compute Ci,Cx [pC:pC_end-1]
            //----------------------------------------------------------------------

            int64_t pC, pC_end ;
            GB_PARTITION (pC, pC_end, cnz, tid, nthreads) ;

            if (pC >= cnz || pC >= pC_end) continue ;
            // find where this task starts in C
            int64_t kC_task = GB_search_for_vector (Cp, GB_Cp_IS_32, pC, 0, cnvec,
                cvlen) ;


            int64_t pC_delta = pC - GBp_C (Cp, kC_task, cvlen) ;
            //----------------------------------------------------------------------
            // compute C(:,kC) for all vectors kC in this task
            //----------------------------------------------------------------------

            for (int64_t kC = kC_task ; kC < cnvec && pC < pC_end ; kC++)
            {
                //------------------------------------------------------------------
                // get the vectors C(:,jC), A(:,jA), and B(:,jB)
                //------------------------------------------------------------------
                // C(:,jC) = kron (A(:,jA), B(:,jB), the (kC)th vector of C,
                // where jC = GBh_C (Ch, kC)
                
                int64_t kA = kC / bnvec ;
                int64_t kB = kC % bnvec ;

                // get A(:,jA), the (kA)th vector of A
                int64_t jA = GBh_A (Ah, kA) ;
                int64_t pA_start = GBp_A (Ap, kA, avlen) ;
                int64_t pA_end   = GBp_A (Ap, kA+1, avlen) ;

                // get B(:,jB), the (kB)th vector of B
                int64_t jB = GBh_B (Bh, kB) ;
                int64_t pB_start = GBp_B (Bp, kB, bvlen) ;
                int64_t pB_end   = GBp_B (Bp, kB+1, bvlen) ;
                int64_t bknz = pB_end - pB_start ;

                if (bknz == 0) continue;
                // shift into the middle of A(:,jA) and B(:,jB) for the first
                // vector of C for this task.

                if (kC == kC_task && bknz > 0)
                { 
                    int64_t remaining = pC_delta;
                    int64_t pA = pA_start;
                    int64_t pB = pB_start;
                    bool found = false;
                    GB_C_TYPE dummy_buf[GB_VLA(csize)];
                     for ( ; pA < pA_end && !found; pA++)
                {
                    int64_t iA = GBi_A(Ai, pA, avlen);
                    if (!GB_A_ISO) GB_GETA(a, Ax, pA, false);

                    for (pB = pB_start; pB < pB_end; pB++)
                    {
                        int64_t iB = GBi_B(Bi, pB, bvlen);
                        if (!GB_B_ISO) GB_GETB(b, Bx, pB, false);

                        // Используем GB_KRONECKER_OP с фиктивным буфером,
                        // чтобы проверить, стал бы элемент ненулевым
                        int64_t dummy_pC = 0; 
                        GB_KRONECKER_OP(dummy_buf, dummy_pC, a, iA, jA, b, iB, jB);
                        // Макрос инкрементирует dummy_pC только если ненулевой
                        if (dummy_pC > 0)  // значит, был ненулевой
                        {
                            if (remaining == 0)
                            {
                                found = true;
                                pA_start = pA;
                                pB_start = pB;
                                break;
                            }
                            remaining--;
                        }
                    }
                }
                // На случай, если pC_delta оказался больше числа ненулевых в векторе
                if (!found)
                {
                    pA_start = pA_end;
                }
                pC_delta = 0;  // сброс для следующих векторов
                }

                //------------------------------------------------------------------
                // for all entries in A(:,jA), skipping entries for first vector
                //------------------------------------------------------------------

                int64_t pA = pA_start ;
                for ( ; pA < pA_end && pC < pC_end ; pA++)
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

                    //--------------------------------------------------------------
                    // for all entries in B(:,jB), skipping entries for 1st vector
                    //--------------------------------------------------------------

                    // scan B(:,jB), skipping to the first entry of C if this is
                    // the first time B is accessed in this task
                    int64_t pB = pB_start ;
                    for ( ; pB < pB_end && pC < pC_end ; pB++)
                    { 

                        //----------------------------------------------------------
                        // b = B(iB,jB), typecasted to op->ytype
                        //----------------------------------------------------------

                        int64_t iB = GBi_B (Bi, pB, bvlen) ;
                        if (!GB_B_ISO)
                        { 
                            GB_GETB (b, Bx, pB, false) ;
                        }

                        //----------------------------------------------------------
                        // C(iC,jC) = A(iA,jA) * B(iB,jB)
                        //----------------------------------------------------------

                        if (!GB_C_IS_FULL)
                        { 
                            // save the row index iC
                            // Ci [pC] = iAblock + iB ;
                            GB_ISET (Ci, pC, iAblock + iB) ;
                        }
                        // Cx [pC] = op (a, b)
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
    }
}

