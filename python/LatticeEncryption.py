import numpy
from numpy.ma.core import transpose, innerproduct
from numpy.polynomial import Polynomial as Poly
import math
import random
def generate_modulus(n, epsilon):
    lower_bound = 2 ** (n ** epsilon)
    upper_bound = 2 * lower_bound
    # Generate a random odd number in the range [lower_bound, upper_bound]
    q = numpy.random.randint(lower_bound, upper_bound)
    # Ensure q is odd
    if q % 2 == 0:
        q += 1
    return q
def sampleNoise(q,n,l):
    return numpy.random.randint(0, high=q, size = (l, n))
def randomMatrix(m,n,q):
    return numpy.random.randint(0, high=q, size=(n, m))
def BGen(n,q, tau, sample ,samplen1, i,j,l):
    return numpy.inner(numpy.random.randint(0, high=q, size=(l, n)), sample) + 2*numpy.random.randint(0, high=q) + (2^tau * samplen1[i] * samplen1[j])

def BGen0(n,q, tau, sample,l):
    return numpy.inner(numpy.random.randint(0, high=q, size=(l, n)), sample) + 2+numpy.random.randint(0, high=q) +2^tau
def fullset(n,  q, samples, tau,l):
    bigset = []
    for sample in range (0, len(samples)):
        for i  in range(0,n):
            for j in range(i,n):
                for i in range(0, tau):
                    if sample !=0:
                        bigset.append([numpy.random.randint(0,q, size =n), BGen(n,q, tau, samples[sample] ,samples[sample-1], i,j,l)])
                        bigset.append([numpy.random.randint(0, q, size =n), BGen0(n, q, tau, samples[sample], l)])
    return bigset


def keyGen(n, m):
    q =  generate_modulus(n, 0.5)
    l = round(0.5*math.log(n,2))
    print(q)
    samples = sampleNoise(q,n,l)
    evk = fullset(n, q, samples, round(math.log(q,2)),l)
    A = numpy.random.randint(0,q, size =(n,m))
    e = numpy.random.randint(0,q, size =(m))
    b = A+(2 *e)
    print (b)
    privatekey = samples[-1]
    print(privatekey)
    return privatekey, A, b, q, evk

def encryption(A,b,m, bit):
    r = numpy.random.randint(0,1,size = m)
    v = A*r
    w = b*r + bit
    return [[v,w],0]

def decryption(cypher, s,q):
    return ((cypher[0][1] - innerproduct(transpose(cypher[0][0]), s))%q%2)[0][0]

def encryptBin(A,b,m,binary):
    data =[]
    for i in binary:
        data.append(encryption(A,b,m,int(i)))
    return data

def encryptNum(A,b,m,num):
    binary =bin(num)[2:]
    return encryptBin(A,b,m,binary)

def decryptBin(data,s,q):
    message =""
    for i in data:
        message += str(decryption(i,s,q))
    return message


