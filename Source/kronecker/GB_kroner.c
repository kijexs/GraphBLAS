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
//
// If sel is not NULL, it is a user-defined selector function that determines
// whether a computed value C(iC,jC) should be kept in the result matrix.
// The selector is called as sel(&result, &value), where result is a bool
// that should be set to true if the value should be kept, false otherwise.

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
    const GrB_IndexUnaryOp select,  // optional selector for C, unused if NULL
    const bool flipij_sel,          // if true, i and j are flipped for selector
    const GrB_Scalar Thunk,         // third input: scalar y
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
    // get selector
    //--------------------------------------------------------------------------

    GxB_index_unary_function sel = (select == NULL) ? NULL : select->idxunop_function ;
    GB_Opcode opcode_sel = (select == NULL) ? GB_NOP_code : select->opcode ;
    // positional ops or user-defined idxunops never result in an iso matrix
    bool sel_is_positional = (select != NULL) &&
        (GB_OPCODE_IS_POSITIONAL(opcode_sel) || (opcode_sel == GB_USER_idxunop_code)) ;

    // allocate the ythunk scalars
    size_t ythunk_size = (sel != NULL) ? select->ytype->size : sizeof(int64_t) ;
    GB_void ythunk [GB_VLA(ythunk_size)] ;
    // ythunk = (op->ytype) Thunk
    if (Thunk != NULL && select != NULL)
    {
        GB_cast_scalar (ythunk, select->ytype->code, Thunk->x, Thunk->type->code, ythunk_size) ;
    }

    const GB_void *y = (sel == NULL) ? NULL : &ythunk ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    GB_Type_code ccode = ctype->code ;
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

    // C is hypersparse if either A or B are hypersparse.  It is never bitmap.
    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;
    bool C_is_full = GB_as_if_full (A) && GB_as_if_full (B) ;
    int C_sparsity = C_is_full ? GxB_FULL :
        ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    //--------------------------------------------------------------------------
    // count non-zero elements in result if selector is non-zero
    //--------------------------------------------------------------------------
    
    GrB_Index cnz = (sel == NULL) ? cnzmax : 0 ;
    int64_t nvec_nonempty = 0 ;
    
    int64_t *restrict p = NULL ;
    int64_t *restrict h = NULL ;
    int64_t *restrict hp = NULL ;
    size_t p_size, h_size = 0, hp_size = 0 ;
    
    if (sel != NULL && (!C_iso || sel_is_positional))
    { 
        int64_t kC ;
        p = GB_MALLOC_MEMORY (cnvec + 1, sizeof(int64_t), &(p_size)) ;
        ASSERT (p_size == GB_Global_memtable_size (p)) ;
        GB_memset (p, 0, p_size, nthreads) ;

        struct GB_Matrix_opaque stub_header ;
        GrB_Matrix C_stub = NULL ;
        GB_CLEAR_MATRIX_HEADER(C_stub, &stub_header) ;
        C_stub->magic = GB_MAGIC ;
        // map working arrays into the stub matrix
        C_stub->type  = ctype ;
        C_stub->iso = C_iso ;
        C_stub->p = p ;
        C_stub->nvec = 0 ;
        // via the JIT kernel
        info = GB_kroner_jit (C_stub, op, select, y, flipij, flipij_sel, A, B, nthreads) ;

        if (info == GrB_NO_VALUE)
        { 
            // via the generic kernel
            #define GB_A_TYPE GB_void
            #define GB_B_TYPE GB_void
            #define GB_C_TYPE GB_void
            #define GB_A_ISO A_iso
            #define GB_B_ISO B_iso
            #define GB_C_ISO C_iso
            #define P_PTR (p)
            const bool A_iso = A->iso ;
            const bool B_iso = B->iso ;
            const int64_t asize = A->type->size ;
            const int64_t bsize = B->type->size ;
            const GB_Type_code zcode = select->ztype->code ;
            const GB_Type_code xcode = (select->xtype == NULL) ? 0 : select->xtype->code ;
            const size_t zsize = select->ztype->size ;
            const size_t xsize = (select->xtype == NULL) ? 0 : select->xtype->size ;
            GB_cast_function cast_Z_to_bool, cast_C_to_X ;
    
            if (select->ztype == GrB_BOOL && select->xtype == ctype)
            { 

                //------------------------------------------------------------------
                // A is non-iso and no typecasting is required
                //------------------------------------------------------------------

                #undef  GB_TEST_KRON_VALUE_OF_ENTRY
                #define GB_TEST_KRON_VALUE_OF_ENTRY(keep,p)                     \
                    bool keep ;                                                 \
                    sel (&keep, p, flipij_sel ? jC : iC,                        \
                         flipij_sel ? iC : jC, y) ;                             

            }
            else
            { 

                //------------------------------------------------------------------
                // A is non-iso and typecasting is required
                //------------------------------------------------------------------

                cast_C_to_X = GB_cast_factory(xcode, ccode) ;
                cast_Z_to_bool = GB_cast_factory(GB_BOOL_code, zcode) ;

                #undef  GB_TEST_KRON_VALUE_OF_ENTRY
                #define GB_TEST_KRON_VALUE_OF_ENTRY(keep,p)                     \
                    bool keep ;                                                 \
                    GB_void z [GB_VLA(zsize)] ;                                 \
                    GB_void x [GB_VLA(xsize)] ;                                 \
                    cast_C_to_X (x, p, csize) ;                                 \
                    sel (z, x, flipij_sel ? jC : iC, flipij_sel ? iC : jC, y) ; \
                    cast_Z_to_bool (&keep, z, zsize) ;

            }

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

            #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
            #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

            GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
            GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;

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
                if (fmult != NULL)                              \
                {                                               \
                    /* standard binary operator */              \
                    fmult (Cx +(pC)*csize, a, b) ;              \
                }                                               \
                else                                            \
                {                                               \
                    /* index binary operator */                 \
                    if (flipij)                                 \
                    {                                           \
                        fmult_idx (Cx +(pC)*csize,              \
                            a, jx, ix, b, jy, iy, theta) ;      \
                    }                                           \
                    else                                        \
                    {                                           \
                        fmult_idx (Cx +(pC)*csize,              \
                            a, ix, jx, b, iy, jy, theta) ;      \
                    }                                           \
                }                                               \
            }

            #define GB_GENERIC
            #include "ewise/include/GB_ewise_shared_definitions.h"
            #include "kronecker/template/GB_kroner_count_sel_template.c"
        }
        
        if (info == GrB_SUCCESS) 
        { 
            p = (int64_t *) C_stub->p ;
        }

        GB_cumsum (p, false, cnvec, NULL, nthreads, Werk) ;

        if (!(C_is_full = (C_is_full && cnz == cnzmax) ))
        { 
            if (C_is_hyper)
            { 
                h = GB_MALLOC_MEMORY (cnvec, sizeof(int64_t), &(h_size)) ;
                hp = GB_MALLOC_MEMORY (cnvec, sizeof(int64_t), &(hp_size)) ;
                ASSERT (h_size == GB_Global_memtable_size (h) && hp_size == GB_Global_memtable_size (hp)) ;
                GB_memset (h, 0, h_size, nthreads) ;
                GB_memset (hp, 0, hp_size, nthreads) ;
                for (kC = 0 ; kC < cnvec ; kC++)
                { 
                    if (p [kC + 1] > p [kC])
                    { 
                        int64_t kA = kC / bnvec ;
                        int64_t kB = kC % bnvec ;
                        const int64_t jA = GBh_A (Ah, kA) ;
                        const int64_t jB = GBh_B (Bh, kB) ;

                        h [nvec_nonempty++] = jA * bvdim + jB ;
                        hp [nvec_nonempty] = p [kC + 1] ;
                    }
                }
                cnz = hp[nvec_nonempty] ;
            }
            else
            {
                cnz = p[cnvec];
            }
        }
    }

    //--------------------------------------------------------------------------
    // update C_iso based on the selector
    //--------------------------------------------------------------------------
    
    // if the selector is positional, it may have altered the values, so C 
    // can no longer be guaranteed to be iso
    C_iso = C_iso && !sel_is_positional ;
    if (C_iso)
    { 
        if (sel != NULL)
        {
            bool result = false ;
            sel (&result, cscalar, 0, 0, y) ;
            if (result) 
            {
                cnz = cnzmax ;
            }
        }
        
        // the values of A and B are no longer needed if C is iso
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, cnz, (int64_t) cvlen, (int64_t) cvdim, Werk) ;

    // C_is_hyper special case: allocate only nvec_nonempty vectors
    int64_t final_nvec = (C_is_hyper && !C_iso && sel != NULL) ? nvec_nonempty : cnvec ;
    
    if (cnz == 0) {
        C_sparsity = GxB_HYPERSPARSE ;
        final_nvec = 0 ;
    }
    
    GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_ph_malloc, C_is_csc,
        C_sparsity, true, B->hyper_switch, final_nvec, cnz, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;
    
    // quick return if C is empty
    if (cnz == 0 && C_iso)
    {
        C->magic = GB_MAGIC ;
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // compute the column counts of C: Cp and Ch if C is hypersparse
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    #define GB_Cp_IS_32 Cp_is_32

    if (!C_is_full)
    { 
        if (sel != NULL && !C_iso)
        { 
            if (C_is_hyper)
            { 
                C->nvec = nvec_nonempty ;
                GB_nvec_nonempty_set (C, nvec_nonempty) ;

                for (int64_t i = 0; i < nvec_nonempty; i++) 
                { 
                    GB_ISET (Ch, i, h[i]) ; 
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
        else if (cnz > 0)
        { 
            // C is sparse or hypersparse
            int64_t kC ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (kC = 0 ; kC < cnvec ; kC++)
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
                // Cp [kC] = aknz * bknz ;
                GB_ISET (Cp, kC, aknz * bknz) ;
                if (C_is_hyper)
                { 
                    // Ch [kC] = jA * bvdim + jB ;
                    GB_ISET (Ch, kC, jA * bvdim + jB) ;
                }
            }

            int64_t nvec_nonempty ;
            GB_cumsum (Cp, Cp_is_32, cnvec, &nvec_nonempty, nthreads, Werk) ;
            GB_nvec_nonempty_set (C, nvec_nonempty) ;
            C->nvals = GB_IGET (Cp, cnvec) ;
            if (C_is_hyper) C->nvec = cnvec ;
        }
    }
    
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and/or full
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        // C->x [0] = cscalar = op (A,B)
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        { 
            // no more work to do if C is iso and full
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_MEMORY (&p, p_size) ;
            GB_FREE_MEMORY (&h, h_size) ;
            GB_FREE_MEMORY (&hp, hp_size) ;
            GB_FREE_WORKSPACE ;
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // quick return if C is empty
    //--------------------------------------------------------------------------

    cnz = GB_nnz (C) ;
    if (cnz == 0)
    { 
        GB_FREE_MEMORY(&p, p_size) ;
        GB_FREE_MEMORY(&h, h_size) ;
        GB_FREE_MEMORY(&hp, hp_size) ;
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    // temporarily use full-length p so the fill kernel can index
    // all vectors 0..cnvec-1
    void *temporary_p = C->p ;
    if (sel != NULL && !C_iso)
    {
        C->p = p ;
    }
    
    // via the JIT kernel
    info = GB_kroner_jit (C, op, select, y, flipij, flipij_sel, A, B, nthreads) ;

    C->p = temporary_p ;
    
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
        GB_Type_code zcode, xcode ;
        size_t zsize, xsize ;
        GB_cast_function cast_Z_to_bool, cast_C_to_X ;

        if (sel != NULL && !C_iso)
        {
            zcode = select->ztype->code ;
            xcode = (select->xtype == NULL) ? 0 : select->xtype->code ;
            zsize = select->ztype->size ;
            xsize = (select->xtype == NULL) ? 0 :select->xtype->size ;
            
            if (select->ztype == GrB_BOOL && select->xtype == ctype)
            { 

                //------------------------------------------------------------------
                // A is non-iso and no typecasting is required
                //------------------------------------------------------------------

                #undef  GB_TEST_KRON_VALUE_OF_ENTRY
                #define GB_TEST_KRON_VALUE_OF_ENTRY(keep,p)                     \
                    bool keep ;                                                 \
                    sel (&keep, p, flipij_sel ? jC : iC,                        \
                         flipij_sel ? iC : jC, y) ;                             

            }
            else
            { 

                //------------------------------------------------------------------
                // A is non-iso and typecasting is required
                //------------------------------------------------------------------

                cast_C_to_X = GB_cast_factory(xcode, ccode) ;
                cast_Z_to_bool = GB_cast_factory(GB_BOOL_code, zcode) ;

                #undef  GB_TEST_KRON_VALUE_OF_ENTRY
                #define GB_TEST_KRON_VALUE_OF_ENTRY(keep,p)                     \
                    bool keep ;                                                 \
                    GB_void z [GB_VLA(zsize)] ;                                 \
                    GB_void x [GB_VLA(xsize)] ;                                 \
                    cast_C_to_X (x, p, csize) ;                                 \
                    sel (z, x, flipij_sel ? jC : iC, flipij_sel ? iC : jC, y) ; \
                    cast_Z_to_bool (&keep, z, zsize) ;

            }
        }

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
            if (fmult != NULL)                              \
            {                                               \
                /* standard binary operator */              \
                fmult (Cx +(pC)*csize, a, b) ;              \
            }                                               \
            else                                            \
            {                                               \
                /* index binary operator */                 \
                if (flipij)                                 \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, jx, ix, b, jy, iy, theta) ;      \
                }                                           \
                else                                        \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, ix, jx, b, iy, jy, theta) ;      \
                }                                           \
            }                                               \
        }

        #define GB_GENERIC
        #include "ewise/include/GB_ewise_shared_definitions.h"
        if (sel != NULL && !C_iso)
        { 
            #include "kronecker/template/GB_kroner_sel_template.c"
        }
        else
        { 
            #include "kronecker/template/GB_kroner_template.c"
        }
        
        info = GrB_SUCCESS ;
    }
    
    if (sel != NULL)
    { 
        GB_FREE_MEMORY (&p, p_size) ;
        GB_FREE_MEMORY (&h, h_size) ;
        GB_FREE_MEMORY (&hp, hp_size) ;
    }
    
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

