import numpy as np
def matrix_multiply(A, B):
    ra=A.shape[0]
    ca=A.shape[1]
    rb=B.shape[0]
    cb=B.shape[1] 
    if ca!=rb: 
        raise ValueError("Number of columns in A must be equal to number of rows in B")
    res=np.zeros((ra,cb))
    for i in range (ra):
        for j in range (cb):
            total=0;
            for k in range(ca):
                total +=A[i][k]*B[k][j]
            res[i][j]=total
    return res

           
# test        
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(matrix_multiply(A, B))