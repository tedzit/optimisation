import numpy as np
import matplotlib.pyplot as plt

# Interpolation algorithm of a set of data  by a polynome in canonical basis

# Input data in 2D
def getPolynomial(D):
    size = D.shape
    n = D.shape[0]
    A = np.zeros((n,n))
    B = np.zeros((n,1))
    X = np.zeros((n,1))
    
    for i in range(n):
        for j in range(n):
            A[i][j] = np.power(D[i][0],j)
        B[i][0] = D[i][1]
    print(A)
    print(B)
    X = np.linalg.inv((A.transpose()).dot(A)).dot(A.transpose().dot(B))
    print(X)


# main function
if __name__ == "__main__":
    # Input data xi, yi
    X = np.array([[-1, 2],[1, 0],[2, 1]])
    getPolynomial(X)    
