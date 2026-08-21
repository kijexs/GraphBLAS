function test320
%TEST320 test mask C<M>=Z, iso case with user selector
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0
rng ('default') ;
GB_mex_burble (1) ;
n = 100 ;
Cin.matrix = spones (sprand (n, n, 0.01)) ;
Cin.sparsity = 1 ;
Cin.iso = true ;
Cin.class = 'double' ;
m = 10 ;
A.matrix = spones (sprand (m, m, 0.5)) ;
A.iso = true ;
A.class = 'double' ;
B.matrix = spones (sprand (m, m, 0.5)) ;
B.iso = true ;
B.class = 'double' ;
op.opname = 'times' ;
op.optype = 'double' ;
accum.opname = 'second' ;
accum.optype = 'double' ;
M = logical (sprand (n, n, 0.5)) ;

sel.opname = 'valuene' ;
sel.optype = 'double' ;
y.matrix = 0 ;
y.class = 'double' ;

K_full = GB_spec_kron (sparse(n, n), [], [], op, A, B, []) ;

C2 = GB_spec_select_idxunop (Cin, M, accum, sel, K_full, y, []) ;

C1 = GB_mex_kron  (Cin, M, accum, op, A, B, [], sel, y) ;
GB_spec_compare (C1, C2) ;

% --- Index selector ---
sel.opname = 'tril' ;
sel.optype = 'int64' ;
y.matrix = int64(0) ;
y.class = 'int64' ;

K_full = GB_spec_kron (sparse(n, n), [], [], op, A, B, []) ;
C2 = GB_spec_select_idxunop (Cin, M, accum, sel, K_full, y, []) ;

C1 = GB_mex_kron  (Cin, M, accum, op, A, B, [], sel, y) ;
GB_spec_compare (C1, C2) ;

% --- ISO matrices with selector that rejects all elements ---
sel.opname = 'valueeq' ;
sel.optype = 'double' ;
y.matrix = 0 ;
y.class = 'double' ;

K_full = GB_spec_kron (sparse(n, n), [], [], op, A, B, []) ;
C2 = GB_spec_select_idxunop (Cin, M, accum, sel, K_full, y, []) ;
C1 = GB_mex_kron (Cin, M, accum, op, A, B, [], sel, y) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest320: all tests passed\n') ;