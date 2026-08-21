function test386
%TEST386 test kron with idxop and user selector
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0
rng ('default') ;
A = sprand (2, 3, 0.5) ;
B = sprand (3, 4, 0.5) ;

sel_names = { 'valuene', 'triu' } ;

for s = 1:length(sel_names)
    op_name = sel_names{s} ;
    sel.opname = op_name ;
    
    is_positional = ismember (op_name, {'triu', 'tril', 'diag', 'offdiag', ...
        'rowindex', 'colindex', 'diagindex', 'colle', 'colgt', 'rowle', 'rowgt'}) ;
        
    if (is_positional)
        sel.optype = 'int64' ;
        y.matrix = int64(0) ;
        y.class = 'int64' ;
    else
        sel.optype = 'double' ;
        y.matrix = 0 ;
        y.class = 'double' ;
    end
    
    for atrans = 0:1
        for btrans = 0:1
            C_full = GB_mex_kron_idx (A, B, atrans, btrans, 0) ;
            
            C_in.matrix = C_full ;
            C_in.sparsity = 1 ;
            C_in.class = 'double' ;
            
            A_in.matrix = C_full ;
            A_in.sparsity = 1 ;
            A_in.class = 'double' ;
            
            C2_struct = GB_spec_select_idxunop (C_in, [], [], sel, A_in, y, []) ;
            C2 = C2_struct.matrix ;
            
            for csc = 0:1
                C1 = GB_mex_kron_idx (A, B, atrans, btrans, csc, sel, y) ;
                assert (isequal (C1, C2)) ;
            end
        end
    end
end
fprintf ('\ntest386: all tests passed\n') ;