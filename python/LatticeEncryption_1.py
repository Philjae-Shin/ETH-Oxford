import numpy
from numpy.polynomial import Polynomial as Poly
import math
import random


def generate_modulus(n, epsilon):
    lower_bound = 2 ** (int(n * epsilon))
    upper_bound = 2 * lower_bound
    # Generate a random odd number in [lower_bound, upper_bound]
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
    return (numpy.inner(
        numpy.random.randint(0, high=q, size=(l, n)),
        sample
    )
            + 2 * numpy.random.randint(0, high=q)
            + (2 ** tau * samplen1[i] * samplen1[j]))


def BGen0(n, q, tau, sample, l):
    return (numpy.inner(
        numpy.random.randint(0, high=q, size=(l, n)),
        sample
    )
            + 2 + numpy.random.randint(0, high=q)
            + 2 ** tau)


def fullset(n, q, samples, tau, l):
    bigset = []
    for sidx in range(len(samples)):
        for i in range(n):
            for j in range(i, n):
                for _ in range(tau):  # (original code re-used 'i'; here, replaced with '_')
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
    q = generate_modulus(n, 0.5)
    l = round(0.5 * math.log(n, 2))
    print(q)
    samples = sampleNoise(q, n, l)
    evk = fullset(n, q, samples, round(math.log(q, 2)), l)
    A = numpy.random.randint(0, q, size=(n, m))
    e = numpy.random.randint(0, q, size=(m,))
    # B = A + 2*e (kept from the original code)
    B = A + 2 * e  # shape (n,m) with broadcasting

    print("B =", B)
    privatekey = samples[-1]  # shape (n,)
    print("privatekey =", privatekey)
    return privatekey, A, B, evk


# ===============================
# Encryption (from the first screenshot)
# ===============================
def encrypt(A, B, q, message):
    """
    First screenshot (Encryption) formula:
      - r in {0,1}^m
      - v = A^T * r   (mod q)
      - w = b^T * r + message   (mod q)
    Here, b is taken as B[0], i.e., the first row of B (shape (m,))
    """
    n, m = A.shape
    b = B[0]  # first row of B as our 'b' vector

    # r ~ {0,1}^m
    r = numpy.random.randint(0, 2, size=m)

    # v = A^T * r (mod q)
    # A^T: (m,n), r: (m,) => v: (n,)
    v = numpy.dot(A.T, r) % q

    # w = b^T * r + message (mod q)
    w = (numpy.dot(b, r) + message) % q

    # ciphertext = ((v, w), level=0)
    return (v, w, 0)


# ===============================
# Decryption (from the second screenshot)
# ===============================
def decrypt(sk, ct, q):
    """
    Second screenshot (Decryption) formula:
      - c = ((v, w), L)
      - Decryption result = (w - <v, sk>) mod q mod 2
    """
    (v, w, level) = ct
    # v: shape (n,), sk: shape (n,)
    phase = (w - numpy.dot(v, sk)) % q
    # finally mod 2
    return int(phase % 2)


# ===============================
# Test
# ===============================
if __name__ == "__main__":
    # Generate keys
    sk, A, B, evk = keyGen(50, 30)

    # Example message
    msg = 1
    # We either generate a new q or reuse the one from keyGen for encryption
    ct = encrypt(A, B, q=generate_modulus(50, 0.5), message=msg)
    print("Ciphertext:", ct)

    # Decrypt
    dec = decrypt(sk, ct, generate_modulus(50, 0.5))
    print("Decrypted =", dec)
