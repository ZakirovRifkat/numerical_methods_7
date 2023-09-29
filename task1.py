import numpy as np
import scipy.linalg as sp

def spectralRadius(matrix):
    eigenvalues = sp.eigvals(matrix)
    return max(abs(eigenvalues))

def q(x,y):
    return 1

def p(x,y):
    return 1+2*x

if __name__ == '__main__':
    print(5)    



