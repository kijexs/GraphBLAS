function test386
%TEST386 test kron with idxop and selectors

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sprand (2, 3, 0.5) ;
B = sprand (3, 4, 0.5) ;

sel_names = { 'valuene', 'triu' } ;

for s = 1:length (sel_names)
    sel.opname = sel_names {s} ;

    if (isequal (sel.opname, 'valuene'))
        sel.optype = 'double' ;
    else
        sel.optype = 'int64' ;
    end
    y.matrix = GB_mex_cast (0, sel.optype) ;
    y.class = sel.optype ;

    for atrans = 0:1
        for btrans = 0:1
            C0.matrix = GB_mex_kron_idx (A, B, atrans, btrans, 0) ;
            C0.sparsity = 1 ;
            C0.class = 'double' ;

            C2 = GB_spec_select_idxunop (C0, [ ], [ ], sel, C0, y, [ ]) ;
            C2 = C2.matrix ;

            for csc = 0:1
                C1 = GB_mex_kron_idx (A, B, atrans, btrans, csc, sel, y) ;
                assert (isequal (C1, C2)) ;
            end
        end
    end
end

fprintf ('\ntest386: all tests passed\n') ;
