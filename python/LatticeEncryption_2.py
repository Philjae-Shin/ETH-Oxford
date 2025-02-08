# LatticeEncryption.py
import numpy
from numpy.polynomial import Polynomial as Poly
import math
import random


def generate_modulus(n, epsilon):
    lower_bound = 2 ** (int(n * epsilon))
    upper_bound = 2 * lower_bound
    # Generate a random odd number in the range [lower_bound, upper_bound]
    q = numpy.random.randint(lower_bound, upper_bound)
    # Ensure q is odd
    if q % 2 == 0:
        q += 1
    return q


def sampleNoise(q, n, l):
    return numpy.random.randint(0, high=q, size=(l, n))


def randomMatrix(m, n, q):
    return numpy.random.randint(0, high=q, size=(n, m))


def BGen(n, q, tau, sample, samplen1, i, j, l):
    return numpy.inner(
        numpy.random.randint(0, high=q, size=(l, n)),
        sample
    ) + 2 * numpy.random.randint(0, high=q) + (2 ** tau * samplen1[i] * samplen1[j])


def BGen0(n, q, tau, sample, l):
    return numpy.inner(
        numpy.random.randint(0, high=q, size=(l, n)),
        sample
    ) + 2 + numpy.random.randint(0, high=q) + 2 ** tau


def fullset(n, q, samples, tau, l):
    bigset = []
    for sidx in range(len(samples)):
        for i in range(n):
            for j in range(i, n):
                for _ in range(tau):  # (The original code re-used 'i'; here corrected to '_')
                    if sidx != 0:
                        bigset.append([
                            numpy.random.randint(0, q, size=n),
                            BGen(n, q, tau, samples[sidx], samples[sidx - 1], i, j, l)
                        ])
                        bigset.append([
                            numpy.random.randint(0, q, size=n),
                            BGen0(n, q, tau, samples[sidx], l)
                        ])
    return bigset


def keyGen(n, m):
    # 1) Generate the modulus q
    q = generate_modulus(n, 0.5)
    l = round(0.5 * math.log(n, 2))
    print("Generated q =", q)

    # 2) Noise samples
    samples = sampleNoise(q, n, l)  # shape (l, n)

    # 3) Create evk (a simple fullset)
    tau = round(math.log(q, 2))
    evk = fullset(n, q, samples, tau, l)

    # 4) Form the public key: A, B
    A = numpy.random.randint(0, q, size=(n, m))
    e = numpy.random.randint(0, q, size=m)
    # B = A + 2*e is from the original code; it differs from the typical (A, b) structure,
    # but we keep it unchanged
    B = A + 2 * e  # shape (n, m)

    # 5) The secret key is s_L, i.e. the last element of 'samples'
    privatekey = samples[-1]  # shape (n,)

    print("private key shape:", privatekey.shape)
    return privatekey, A, B, evk, q


# -------------------------------
# Encryption (from the first screenshot)
# -------------------------------
def encrypt(A, B, q, message):
    """
    Refer to Image 1:
      v = A^T * r    (mod q)
      w = b^T * r + message    (mod q)
    Here, we take B[0] as 'b' (shape (m,)),
    and sample r from {0,1}^m.

    ciphertext c = ((v, w), level=0)
    """
    n, m = A.shape  # A: (n,m)

    # b = B[0,:] is used as our 1D 'b' vector
    b = B[0]  # shape (m,)

    # r ~ {0,1}^m
    r = numpy.random.randint(0, 2, size=m)

    # v = A^T * r
    # A^T: (m,n), r: (m,) => v: (n,)
    v = numpy.dot(A.T, r) % q

    # w = b^T * r + message
    # b^T r => scalar
    w = (numpy.dot(b, r) + message) % q

    # level tag = 0
    return (v, w, 0)


# -------------------------------
# Decryption (from the second screenshot)
# -------------------------------
def decrypt(sk, ciphertext, q):
    """
    Refer to Image 2:
      c = ((v, w), L)
      Decryption = ( w - <v, s_L> ) mod q  mod 2
    """
    (v, w, level) = ciphertext
    # s_L = sk, shape (n,)
    # v shape (n,)
    phase = (w - numpy.dot(v, sk)) % q
    return int(phase % 2)


# -------------------------------
# Test (simple demonstration)
# -------------------------------
if __name__ == "__main__":
    # keyGen
    sk, A, B, evk, q = keyGen(n=50, m=30)

    # Encryption
    msg = 1  # For example, 1 in GF(2)
    ct = encrypt(A, B, q, msg)
    print("Ciphertext:", ct)

    # Decryption
    dec = decrypt(sk, ct, q)
    print("Decrypted =", dec)
