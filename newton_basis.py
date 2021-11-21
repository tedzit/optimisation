import numpy as np
import matplotlib.pyplot as plt

# In terpolation aplgorithm of a data set by a polynomial in newton basis
def computePolynomial(X,i,j):
    tot = 1
    xj = X[j][0] 
    for k in range(i):
        tot *= (xj-X[k][0])
    return tot

# Input data in 2D
def getPolynomial(D):
    size = D.shape
    n = D.shape[0]
    X = np.zeros((n,1))
    print(n)

    X[0] = D[0][1]
    for j in range(1,n):
        sum = D[0][1]
        for i in range(1,j):
            pi = computePolynomial(D,i,j)
            sum += (X[i]*pi)
        pj = computePolynomial(D,j,j)
        X[j] = (D[j][1] - sum)/pj
    print(X)
    return X

# Input data in 2D
def drawPolynomial(xs, coeffs, X):
    n=len(coeffs)
    ys = np.zeros(len(xs))
    #ys = coeffs[0]
    
    for j in range(n):
        tot = 1
        for k in range(j):
            tot *= (xs-X[k][0])
        ys += coeffs[j]*tot
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
    ys = drawPolynomial(xs,A, X)
    plt.plot(xs,ys, color="blue")
    plt.show()

