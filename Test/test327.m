function test327
%TEST327 test GrB_kronecker with user selectors (extensive loop)
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0
[binops, ~, ~, ~, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
fprintf ('-------------- tests of GrB_kronecker with selectors:\n') ;
rng ('default') ;
dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;
types = { 'int32', 'int64', 'single', 'double' } ;
am = 5 ; an = 3 ; bm = 4 ; bn = 2 ;
Ax = sparse (100 * sprandn (am,an, 0.5)) ;
Bx = sparse (100 * sprandn (bm,bn, 0.5)) ; 
cm = am * bm ; cn = an * bn ;
Cx = sparse (cm,cn) ;
AT = Ax' ; BT = Bx' ;

sel_names = { 'valuene', 'tril' } ;

for s = 1:length(sel_names)
    op_name = sel_names{s} ;
    
    % Determine selector type and scalar type
    is_positional = ismember (op_name, {'triu', 'tril', 'diag', 'offdiag', ...
        'rowindex', 'colindex', 'diagindex', 'colle', 'colgt', 'rowle', 'rowgt'}) ;
    if (is_positional)
        sel_type = 'int64' ;
    else
        sel_type = 'same' ;
    end
    
    for k2 = [4 7 45:52 ]
        for k1 = 1:4
            type = types {k1} ;
            binop = binops {k2}  ;
            op.opname = binop ;
            op.optype = type ;
            if (GB_spec_is_positional (op.opname))
                if (~(isequal (type, 'int32') || isequal (type, 'int64')))
                    continue
                end
            end
            
            sel.opname = op_name ;
            if (isequal (sel_type, 'same'))
                sel.optype = type ;
                switch type
                    case 'int32', y.matrix = int32(0) ;
                    case 'int64', y.matrix = int64(0) ;
                    case 'single', y.matrix = single(0) ;
                    case 'double', y.matrix = 0 ;
                end
                y.class = type ;
            else
                sel.optype = sel_type ;
                y.matrix = int64(0) ;
                y.class = sel_type ;
            end
            
            fprintf ('[ %s %s sel:%s ] ', binop, type, op_name) ;

            c_rows_nn = am * bm ; c_cols_nn = an * bn ;
            c_rows_tn = an * bm ; c_cols_tn = am * bn ;
            c_rows_nt = am * bn ; c_cols_nt = an * bm ;
            c_rows_tt = an * bn ; c_cols_tt = am * bm ;

            for A_is_hyper = 0:1
                for A_is_csc   = 0:1
                    for B_is_hyper = 0:1
                        for B_is_csc   = 0:1
                            for C_is_csc   = 0:1
                                fprintf ('.') ;
                                clear A B C AT BT
                                
                                A.matrix = Ax ; A.is_hyper = A_is_hyper ; A.is_csc = A_is_csc ;
                                B.matrix = Bx ; B.is_hyper = B_is_hyper ; B.is_csc = B_is_csc ;
                                
                                % AT и BT тоже нужно обновить, так как мы их транспонируем
                                AT = Ax' ; BT = Bx' ;
                                
                                % --- C = kron(A,B) ---
                                C.matrix = sparse (c_rows_nn, c_cols_nn) ; C.is_csc = C_is_csc ;
                                C_full = GB_spec_kron (C, [ ], [ ], op, A, B, dnn) ;
                                C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dnn, sel, y) ;
                                GB_spec_compare (C2, C1) ;
                                
                                % --- C = kron(A',B) ---
                                C.matrix = sparse (c_rows_tn, c_cols_tn) ; C.is_csc = C_is_csc ;
                                C_full = GB_spec_kron (C, [ ], [ ], op, A, B, dtn) ;
                                C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dtn, sel, y) ;
                                GB_spec_compare (C2, C1) ;

                                % --- C = kron(A,B') ---
                                C.matrix = sparse (c_rows_nt, c_cols_nt) ; C.is_csc = C_is_csc ;
                                C_full = GB_spec_kron (C, [ ], [ ], op, A, B, dnt) ;
                                C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dnt, sel, y) ;
                                GB_spec_compare (C2, C1) ;

                                % --- C = kron(A',B') ---
                                C.matrix = sparse (c_rows_tt, c_cols_tt) ; C.is_csc = C_is_csc ;
                                C_full = GB_spec_kron (C, [ ], [ ], op, A, B, dtt) ;
                                C2 = GB_spec_select_idxunop (C_full, [ ], [ ], sel, C_full, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dtt, sel, y) ;
                                GB_spec_compare (C2, C1) ;
                            end
                        end
                    end
                end
            end
            fprintf ('\n') ;
        end
    end
end
fprintf ('\ntest327: all tests passed\n') ;
