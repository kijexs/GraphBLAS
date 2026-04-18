//------------------------------------------------------------------------------
// GB_kroner_sel_template: Kronecker product, C = kron (A,B)
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
        // cnz -> C->nvals
        #define CNZ_SET(val)    do { (C)->nvals = (val); } while(0)
        #define CNZ_INC()       do { (C)->nvals++; } while(0)
        #define CNZ_GET()       ((C)->nvals)  

        // nvec_nonempty -> C->nvec_nonempty 
        #define NVEC_NE_SET(val)    do { (C)->nvec_nonempty = (val); } while(0)
        #define NVEC_NE_INC()       do { (C)->nvec_nonempty++; } while(0)
        #define NVEC_NE_GET()       ((C)->nvec_nonempty)

        #define P_PTR  ((int64_t *) (C)->p)
        #define H_PTR  ((int64_t *) (C)->h)
        // hp -> C->i
        #define HP_PTR ((int64_t *) (C)->i)

        #define P_PTR_SET(ptr)   do { (C)->p = (void *)(ptr); } while(0)
        #define H_PTR_SET(ptr)   do { (C)->h = (void *)(ptr); } while(0)
        #define HP_PTR_SET(ptr)  do { (C)->i = (void *) (ptr); } while(0)

        GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
        GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
        const int64_t avlen = A->vlen ;

        GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
        GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
        const int64_t bvlen = B->vlen ;
        const int64_t bvdim = B->vdim ;
        const int64_t bnvec = B->nvec ;

        const int64_t  cnvec = C->nvec ;
        const int64_t  cvlen = C->vlen ;
        const size_t   csize = C->type->size ;
        const int64_t cnzmax = C->plen;

        #define OP_IS_POSITIONAL  ((C)->jumbled)

        #define H_MALLOC(sz)    GB_MALLOC_MEMORY((sz), sizeof(int64_t), NULL)
        #define HP_MALLOC(sz)   GB_MALLOC_MEMORY((sz), sizeof(int64_t), NULL)
        #define CHECK_SIZES(h_ptr, hp_ptr, h_sz, hp_sz) ((void)0)
        #define H_MEMSET(ptr, val, sz_var) GB_memset((ptr), (val), (sz_var), nthreads)
        #define HP_MEMSET(ptr, val, sz_var) GB_memset((ptr), (val), (sz_var), nthreads)
        
        size_t h_size = 0 ;
        size_t hp_size = 0 ;
        #define WERK_ARG NULL
    #else
        #define CNZ_SET(val)    do { cnz = (val); } while(0)
        #define CNZ_INC()       do { cnz++; } while(0)
        #define CNZ_GET()       (cnz)
        
        #define NVEC_NE_SET(val) do { nvec_nonempty = (val); } while(0)
        #define NVEC_NE_INC()    do { nvec_nonempty++; } while(0)
        #define NVEC_NE_GET()    (nvec_nonempty)
        
        #define P_PTR     (p)
        #define H_PTR     (h)
        #define HP_PTR    (hp)

        #define P_PTR_SET(ptr)  do { p = (ptr); } while(0)
        #define H_PTR_SET(ptr)  do { h = (ptr); } while(0)
        #define HP_PTR_SET(ptr) do { hp = (ptr); } while(0)

        #define H_MALLOC(sz)    GB_MALLOC_MEMORY((sz), sizeof(int64_t), &(h_size))
        #define HP_MALLOC(sz)   GB_MALLOC_MEMORY((sz), sizeof(int64_t), &(hp_size))
        #define H_MEMSET(ptr, val, sz_var) GB_memset((ptr), (val), (sz_var), nthreads)
        #define HP_MEMSET(ptr, val, sz_var) GB_memset((ptr), (val), (sz_var), nthreads)
        #define CHECK_SIZES(h_ptr, hp_ptr, h_sz, hp_sz) \
            ASSERT ((h_sz) == GB_Global_memtable_size (h_ptr) && \
                    (hp_sz) == GB_Global_memtable_size (hp_ptr))
        #define WERK_ARG Werk
        
    #endif

    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;

    //--------------------------------------------------------------------------
    // count nonzero elements in result
    //--------------------------------------------------------------------------

    //----------------------------------------------------------------------
    // compute C(:,kC) for all vectors kC in this task
    //----------------------------------------------------------------------

    if (!GB_C_ISO && !OP_IS_POSITIONAL)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (int64_t kC = 0; kC < cnvec; kC++)
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

            //------------------------------------------------------------------
            // get the vectors C(:,jC), A(:,jA), and B(:,jB)
            //------------------------------------------------------------------

            // C(:,jC) = kron (A(:,jA), B(:,jB), the (kC)th vector of C,
            // where jC = GBh_C (Ch, kC)
            int64_t kA = kC / bnvec ;
            int64_t kB = kC % bnvec ;

            // get A(:,jA), the (kA)th vector of A
            int64_t jA       = GBh_A (Ah, kA) ;
            int64_t pA_start = GBp_A (Ap, kA, avlen) ;
            int64_t pA_end   = GBp_A (Ap, kA+1, avlen) ;

            // get B(:,jB), the (kB)th vector of B
            int64_t jB       = GBh_B (Bh, kB) ;
            int64_t pB_start = GBp_B (Bp, kB, bvlen) ;
            int64_t pB_end   = GBp_B (Bp, kB+1, bvlen) ;
            int64_t bknz     = pB_end - pB_start ;
            if (bknz == 0) continue ;

            //------------------------------------------------------------------
            // for all entries in A(:,jA), skipping entries for first vector
            //------------------------------------------------------------------

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

                //--------------------------------------------------------------
                // for all entries in B(:,jB), skipping entries for 1st vector
                //--------------------------------------------------------------

                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
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

                    if (!GB_C_ISO && !OP_IS_POSITIONAL)
                    { 
                        GB_C_TYPE cwork[GB_VLA(csize)] ;
                        GB_KRONECKER_SELECTOR (cwork, a, b) ;
                    }
                }
            }
        }
        GB_cumsum (P_PTR, false, cnvec, NULL, nthreads, WERK_ARG) ;
        bool c_is_full = GB_C_IS_FULL ;
        if (!(c_is_full = (c_is_full && CNZ_GET() == cnzmax)))
        { 
            if (GB_C_IS_HYPER)
            { 
                H_PTR_SET(H_MALLOC(cnvec)) ;
                HP_PTR_SET(HP_MALLOC(cnvec)) ;

                CHECK_SIZES(H_PTR, HP_PTR, h_size, hp_size) ;
                H_MEMSET (H_PTR, 0, h_size) ;
                HP_MEMSET (HP_PTR, 0, hp_size) ;
                for (int64_t kC = 0 ; kC < cnvec ; kC++)
                { 
                    if (P_PTR [kC + 1] > P_PTR [kC])
                    { 
                        int64_t kA = kC / bnvec ;
                        int64_t kB = kC % bnvec ;
                        const int64_t jA = GBh_A (Ah, kA) ;
                        const int64_t jB = GBh_B (Bh, kB) ;

                        int64_t idx = NVEC_NE_GET() ;
                        H_PTR[idx] = jA * bvdim + jB ;
                        HP_PTR[idx + 1] = P_PTR[kC+1] ;
                        NVEC_NE_INC() ;
                    }
                }
                CNZ_SET(HP_PTR[NVEC_NE_GET()]) ;
            }
            else
            { 
                CNZ_SET(P_PTR[cnvec]) ;
            }
        }
    }

    else if (!GB_C_IS_FULL)
    { 
        H_PTR_SET(H_MALLOC(cnvec)) ;
        ASSERT (h_size == GB_Global_memtable_size (H_PTR)) ;
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (int64_t kC = 0 ; kC < cnvec ; kC++)
        { 
            const int64_t kA = kC / bnvec ;
            const int64_t kB = kC % bnvec ;

            // get A(:,jA), the (kA)th vector of A
            const int64_t jA = GBh_A (Ah, kA) ;
            const int64_t aknz = (Ap == NULL) ? avlen :
                (GB_IGET (Ap, kA+1) - GB_IGET (Ap, kA)) ;
            // get B(:,jB), the (kB)th vector of B
            const int64_t jB = GBh_B (Bh, kB) ;
            const int64_t bknz = (Bp == NULL) ? bvlen :
                (GB_IGET (Bp, kB+1) - GB_IGET (Bp, kB)) ;
            // determine # entries in C(:,jC), the (kC)th vector of C
            // int64_t kC = kA * bnvec + kB ;

            P_PTR [kC] = aknz * bknz ;

            if (GB_C_IS_HYPER)
            { 
                H_PTR [kC] = jA * bvdim + jB ;
            }
        }

        #ifdef GB_JIT_KERNEL
            int64_t nvec_ne_tmp = C->nvec_nonempty ;
            GB_cumsum (P_PTR, false, cnvec, &nvec_ne_tmp, nthreads, WERK_ARG) ;
            C->nvec_nonempty = nvec_ne_tmp ;
        #else
            GB_cumsum (P_PTR, false, cnvec, &(C->nvec_nonempty), nthreads, WERK_ARG) ;
        #endif
        CNZ_SET(P_PTR[cnvec]) ;
        if (GB_C_IS_HYPER) NVEC_NE_SET(cnvec) ;
    }
}

