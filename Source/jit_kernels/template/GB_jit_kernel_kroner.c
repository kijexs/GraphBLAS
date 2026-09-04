//------------------------------------------------------------------------------
// GB_jit_kernel_kroner.c: kronecker product
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_search_for_vector.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_KRONER_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_KRONER_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
#ifdef GB_HAS_SELECTOR
    if (C->nvec == 0)
    { 
        #include "template/GB_kroner_count_sel_template.c"
    }
    else 
    { 
        if (select != NULL && !C->iso)
        { 
            #include "template/GB_kroner_sel_template.c"
        }
        else
        { 
            #include "template/GB_kroner_template.c"
        }
    }
#else
    if (C->nvec == 0)
    { 
        return (GrB_SUCCESS) ;
    }
    else
    { 
        #include "template/GB_kroner_template.c"
    }
#endif
    return (GrB_SUCCESS) ;
}

