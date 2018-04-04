from decode_aux import mat_dims, mat_elem, swap_rows, swap_cols, add_rows, add_cols, is_consistent, add_vecs, dot_pos, next_elem





def NullSpace(mat):
    # Compute a basis for the Null space of the input (binary) matrix.
    # We will follow the method outlined in https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination .
    # Construct an augmented matrix by concatenating a identity matrix of size nxn below the input matrix (whose size is mxn).
    # Reduce the augmented matrix to a column echoelen form.
    # The column vectors in the nxn matrix corresponding to zero columns in the mxn matrix are the basis vectors for the kernel.
    augm = np.zeros((mat.shape[0] + mat.shape[1], mat.shape[1]), dtype = np.int8)
    augm[:mat.shape[0], :] = mat
    augm[mat.shape[0]:, :] = np.identity(mat.shape[1])
    for i in range(mat.shape[0]):
        # If the diagnonal entry is 0, swap with a column that has a '1' at the i-th row.
        if (augm[i, i] == 0):
            toswap = i
            for j in range(i + 1, augm.shape[1]):
                if (augm[i, j] == 1):
                    toswap = j
                    break
            # swap columns i and j
            augm[:, [i, toswap]] = augm[:, [toswap, i]]

        for j in range(i + 1, augm.shape[1]):
            if (augm[i, j] == 1):
                # Add columns: j -> i + j
                augm[:, j] = np.mod(augm[:, i] + augm[:, j], 2)

    # Extract the columns of the nxn matrix corresponding to zero columns of the mxn matrix.
    rank = 0
    for j in range(mat.shape[1]):
        if (not np.any(augm[:mat.shape[0], j] > 0)):
            rank = rank + 1
    nullspace = np.zeros((rank, mat.shape[1]), dtype = np.int8)
    nvec = 0
    for j in range(mat.shape[1]):
        if (not (np.any(augm[:mat.shape[0], j] > 0))):
            nullspace[nvec, :] = augm[mat.shape[0]:, j]
            nvec = nvec + 1
    return nullspace




# computes the basis for the null space of the input matrix
def nullSpace(check_to_vert, vert_to_check):
    mat_pos_tr = [[check_to_vert[i][j] for j in range(len(check_to_vert[i]))] for i in range(len(check_to_vert))]
    mat_pos = [[vert_to_check[i][j] for j in range(len(vert_to_check[i]))] for i in range(len(vert_to_check))]
    
    ker_pos = []
	
    #print("Consistency check initially = %d" % is_consistent(mat_pos, mat_pos_tr))
    
    #(rows,cols) = H.shape
    (rows, cols) = mat_dims(mat_pos, mat_pos_tr)
    
    #print("rows = %d , columns = %d" % (rows,cols))
    
    #print "mat_pos"
    #print mat_pos
    #print "mat_pos_tr"
    #print mat_pos_tr

    id_pos = [[i] for i in range(rows)]
    id_pos_tr = [[i] for i in range(rows)]
	
    op_fail = 1
    
    # swapping and row reduce
    for i in range(cols):
        op_fail = 1
        for row_idx in mat_pos_tr[i]:
            if row_idx >= i and op_fail == 1:
                op_fail = 0
                if row_idx > i:
                    swap_rows(mat_pos, mat_pos_tr, row_idx, i)
                    swap_rows(id_pos, id_pos_tr, row_idx, i)
                else:
                    pass
                    
            else:
                pass

        rows_to_rem = [row_idx for row_idx in mat_pos_tr[i] if row_idx > i]
        
        #print("rows_to_rem at column %d" % i)
        #print rows_to_rem
        
        for rem_idx in rows_to_rem:
            add_rows(mat_pos, mat_pos_tr, rem_idx, i)
            add_rows(id_pos, id_pos_tr, rem_idx, i)

        #print mat_pos_tr

    #print "mat_pos_tr after row reduce"
    #print mat_pos_tr

    # identify the zero rows and store the corresponding rows of the identity matrix
    for i in range(rows):
        if not mat_pos[i]:
            #print "zero row"
            #print id_pos[i]
            ker_pos.append(id_pos[i])
        else:
            pass

    return ker_pos


# compute the logical operators of one-kind, of the CSS code specified by HX and HZ.
def hpc_logicals(mat_posZ, mat_posZ_tr, mat_posX, mat_posX_tr):
    #print "mat_posZ"
    #print mat_posZ
    #print "%%%%% \n"
    #print "mat_posZ_tr"
    #print mat_posZ_tr
    #print "%%%%% \n"
    #print "mat_posX"
    #print mat_posX
    #print "%%%%% \n"
    #print "mat_posX_tr"
    #print mat_posX_tr
    #print "%%%%% \n"
    
    logsZ = []
    logsX = []
    
    # read the HX and HZ parity check matrices
    
    #print "Kernels"
    #print "G1"
    G1 = nullSpace(mat_posZ, mat_posZ_tr)
    #print G1
    #print "G2"
    G2 = nullSpace(mat_posX, mat_posX_tr)
    #print G2
	
    k1 = len(G1)
    k2 = len(G2)
    
    #print("k1 = %d , k2 = %d" % (k1,k2))
    
    del1 = [0 for i in range(k1)]
    del2 = [0 for i in range(k2)]
    
    for i in range(k1):
        u = G1[i]
        c = 1
        del1[i] = 1
        for j in range(k2):
            v = G2[j]
            #if np.dot(u,v) % 2 == 1:
            if dot_pos(u, v) == 1:
                c = 0
                del2[j]=1
                logsX.append(u)
                logsZ.append(v)
                for k in range(k1):
                    if del1[k]==0:
                        #if np.dot(G1[k] , v) % 2 == 1:
                        if dot_pos(G1[k] , v) == 1:
                            #G1[k] = (G1[k] + u) % 2
                            add_vecs(G1,k, u)
            
                for k in range(k2):
                    if del2[k]==0:
                        #if np.dot(G2[k] , u) % 2 == 1:
                        if dot_pos(G2[k] , u) == 1:
                            #G2[k] = (G2[k] + v) % 2
                            add_vecs(G2, k, v)
                            #print "Z-logicals"
                            #print logsZ
							
    return (logsZ, logsX)
