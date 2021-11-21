import numpy as np
import matplotlib.pyplot as plt

# Interpolation algorithm of a set of data  by a polynomal in canonical basis

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
    return X

def drawPolynomial(xs, coeffs):
    n=len(coeffs)
    ys = np.zeros(len(xs))
    for i in range(n):
        ys += coeffs[i]*xs**i
    return ys


# main function
if __name__ == "__main__":
    # Input data xi, yi
    X = np.array([[-1, 2],[1, 0],[2, 1]])
    A = getPolynomial(X)    

    # Plot data
    xmin = -5
    xmax = 5
    # plot points to fit
    plt.scatter(X[:,0], X[:,1], color="red")
    # plot polynomial curb
    xs = np.linspace(xmin,xmax)
    ys = drawPolynomial(xs,A)
    plt.plot(xs,ys, color="blue")
    plt.show()


