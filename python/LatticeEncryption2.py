import numpy as np
from numpy.polynomial import Polynomial as Poly
import math

def generate_modulus(n, epsilon):
    lower_bound = 2 ** (n ** epsilon)
    upper_bound = 2 * lower_bound
    # Generate a random odd number in the range [lower_bound, upper_bound]
    q = np.random.randint(lower_bound, upper_bound)
    # Ensure q is odd
    if q % 2 == 0:
        q += 1
    return q
def sampleNoise(q,n,l):
    return numpy.random.randint(0, high=q, size = (n, l))
def randomMatrix(m,n,q):
    return numpy.random.randint(0, high=q, size=(m, n))

def keyGen(k, n, m, epsilon):
    q =  generate_modulus(n, epsilon)
    l = round(epsilon*log(n,2))
    samples = sampleNoise(q,n,l)

