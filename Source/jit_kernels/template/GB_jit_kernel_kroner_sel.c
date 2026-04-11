//------------------------------------------------------------------------------
// GB_jit_kernel_kroner_sel: kronecker product counting phase
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "include/GB_search_for_vector.h"
#include "include/GB_search_for_vector.h"
#include "cumsum/GB_cumsum.h"   
#include "memory/include/GB_memory_macros.h"

GB_JIT_GLOBAL GB_JIT_KERNEL_KRONER_SELECTOR_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_KRONER_SELECTOR_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    #include "template/GB_kroner_sel_template.c"
    return (GrB_SUCCESS) ;
}


