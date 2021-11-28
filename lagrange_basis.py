import numpy as np
import matplotlib.pyplot as plt

def getPolynomial(D):
    n = D.shape[0]
    X = np.zeros((n,1))
    for j in range(n):
        X[j] = D[j][1]
    return X

# In terpolation aplgorithm of a data set by a polynomial in newton basis
def computePolynomial(x,X,i,n):
    tot = 1
    for k in range(n):
        if k != i:
            tot *= (x-X[k][0])/(X[i][0]-X[k][0])
        else:
            tot *= 1
    return tot

# Input data in 2D
def drawPolynomial(xs, coeffs, X):
    n=len(coeffs)
    ys = np.zeros(len(xs))
    for i in range(n):
        tot = computePolynomial(xs,X,i,n)
        ys += coeffs[i]*tot
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

