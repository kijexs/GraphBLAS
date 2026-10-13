function test327
%TEST327 test kron with selectors

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

am = 5 ;
an = 3 ;
bm = 4 ;
bn = 2 ;

Ax = sparse (100 * sprandn (am,an, 0.5)) ;
Bx = sparse (100 * sprandn (bm,bn, 0.5)) ;

sel_names = { 'valuene', 'tril' } ;

for s = 1:length (sel_names)
    sel.opname = sel_names {s} ;

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

            % the value selector uses the operator type; tril uses int64
            if (isequal (sel.opname, 'valuene'))
                sel.optype = type ;
            else
                sel.optype = 'int64' ;
            end
            y.matrix = GB_mex_cast (0, sel.optype) ;
            y.class = sel.optype ;

            fprintf ('[ %s %s sel:%s ] ', binop, type, sel.opname) ;

            for A_is_hyper = 0:1
                for A_is_csc   = 0:1
                    for B_is_hyper = 0:1
                        for B_is_csc   = 0:1
                            for C_is_csc   = 0:1
                                fprintf ('.') ;

                                clear A
                                A.matrix = Ax ;
                                A.is_hyper = A_is_hyper ;
                                A.is_csc   = A_is_csc   ;

                                clear B
                                B.matrix = Bx ;
                                B.is_hyper = B_is_hyper ;
                                B.is_csc   = B_is_csc   ;

                                clear C
                                C.is_csc = C_is_csc ;

                                %---------------------------------------
                                % kron(A,B)
                                %---------------------------------------

                                C.matrix = sparse (am * bm, an * bn) ;
                                C0 = GB_spec_kron (C, [ ], [ ], op, A, B, dnn) ;
                                C2 = GB_spec_select_idxunop (C0, [ ], [ ], sel, C0, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dnn, sel, y) ;
                                GB_spec_compare (C2, C1) ;

                                %---------------------------------------
                                % kron(A',B)
                                %---------------------------------------

                                C.matrix = sparse (an * bm, am * bn) ;
                                C0 = GB_spec_kron (C, [ ], [ ], op, A, B, dtn) ;
                                C2 = GB_spec_select_idxunop (C0, [ ], [ ], sel, C0, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dtn, sel, y) ;
                                GB_spec_compare (C2, C1) ;

                                %---------------------------------------
                                % kron(A,B')
                                %---------------------------------------

                                C.matrix = sparse (am * bn, an * bm) ;
                                C0 = GB_spec_kron (C, [ ], [ ], op, A, B, dnt) ;
                                C2 = GB_spec_select_idxunop (C0, [ ], [ ], sel, C0, y, [ ]) ;
                                C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dnt, sel, y) ;
                                GB_spec_compare (C2, C1) ;

                                %---------------------------------------
                                % kron(A',B')
                                %---------------------------------------

                                C.matrix = sparse (an * bn, am * bm) ;
                                C0 = GB_spec_kron (C, [ ], [ ], op, A, B, dtt) ;
                                C2 = GB_spec_select_idxunop (C0, [ ], [ ], sel, C0, y, [ ]) ;
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
