function test326
%TEST326 test kron with iso matrices with user selector
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0
rng ('default') ;
A.matrix = sprand (5, 10, 0.4) ;
A.class = 'double' ;
B.matrix = ones (3, 2) ;
B.iso = true ;
B.class = 'double' ;
mult.opname = 'times' ;
mult.optype = 'double' ;
Cin = sparse (15, 20) ;

% --- Value selector ---
sel.opname = 'valuene' ;
sel.optype = 'double' ;
y.matrix = 0 ;
y.class = 'double' ;

C1 = GB_mex_kron  (Cin, [ ], [ ], mult, A, B, [ ], sel, y) ;
C_full = GB_mex_kron (Cin, [ ], [ ], mult, A, B, [ ]) ;
C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_kron  (Cin, [ ], [ ], mult, B, A, [ ], sel, y) ;
C_full = GB_mex_kron (Cin, [ ], [ ], mult, B, A, [ ]) ;
C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
GB_spec_compare (C1, C2) ;

Cin = sparse (9, 4) ;
C1 = GB_mex_kron  (Cin, [ ], [ ], mult, B, B, [ ], sel, y) ;
C_full = GB_mex_kron (Cin, [ ], [ ], mult, B, B, [ ]) ;
C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
GB_spec_compare (C1, C2) ;

% --- Index selector ---
sel.opname = 'triu' ;
sel.optype = 'int64' ;
y.matrix = int64(0) ;
y.class = 'int64' ;

Cin = sparse (15, 20) ; 

C1 = GB_mex_kron  (Cin, [ ], [ ], mult, A, B, [ ], sel, y) ;
C_full = GB_mex_kron (Cin, [ ], [ ], mult, A, B, [ ]) ;
C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
GB_spec_compare (C1, C2) ;

fprintf ('\ntest326: all tests passed\n') ;