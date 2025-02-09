#!/usr/bin/env python3
import numpy as np
from Pyfhel import Pyfhel

def init_bfv_context():
    """
    Initialize a Pyfhel BFV context with a bigger plaintext modulus
    to avoid overflow (wraparound).
    """
    HE = Pyfhel()
    # Use a larger prime t=786433 (instead of default 65537).
    # This gives more room before sums wrap around modulo t.
    HE.contextGen(scheme='BFV', n=8192, t=786433)
    HE.keyGen()
    return HE

def encrypt_int(HE: Pyfhel, value: int):
    arr = np.array([value], dtype=np.int64)
    ptxt = HE.encodeInt(arr)
    ctxt = HE.encryptPtxt(ptxt)
    return ctxt

def decrypt_int(HE: Pyfhel, ctxt) -> int:
    ptxt = HE.decryptPtxt(ctxt)
    arr = HE.decodeInt(ptxt)
    return int(arr[0])

def hom_add(HE: Pyfhel, ctxtA, ctxtB):
    return ctxtA + ctxtB

def hom_add_plain(HE: Pyfhel, ctxtA, val: int):
    arr = np.array([val], dtype=np.int64)
    ptxt = HE.encodeInt(arr)
    return ctxtA + ptxt

def hom_mul_plain(HE: Pyfhel, ctxtA, val: int):
    arr = np.array([val], dtype=np.int64)
    ptxt = HE.encodeInt(arr)
    return ctxtA * ptxt

# Example network with 3 linear layers: W1 (3x1), W2 (3x3), W3 (1x3).
W1 = np.array([[5],  [10], [-5]], dtype=int)  # scaled x10 from your original floats
b1 = np.array([1, -2, 3], dtype=int)

W2 = np.array([[10, -5,  3],
               [ 7,  8, -2],
               [-3,  4,  9]], dtype=int)
b2 = np.array([0, 1, -1], dtype=int)

W3 = np.array([[12, -7, 5]], dtype=int)
b3 = np.array([1], dtype=int)

def linear_layer(HE, W, b, x_ciphers):
    """
    (W * x) + b in BFV, purely homomorphic.
    x_ciphers is a list of ciphertexts, one per dimension in x.
    """
    (m, n) = W.shape
    outputs = []
    for i in range(m):
        acc = hom_mul_plain(HE, x_ciphers[0], W[i, 0])
        for j in range(1, n):
            part = hom_mul_plain(HE, x_ciphers[j], W[i, j])
            acc = hom_add(HE, acc, part)
        acc = hom_add_plain(HE, acc, b[i])
        outputs.append(acc)
    return outputs

def forward_pass(HE, x0_cipher):
    """
    3-layer linear pass, no activation:
      x1 = W1*x0 + b1
      x2 = W2*x1 + b2
      x3 = W3*x2 + b3
    """
    x1 = linear_layer(HE, W1, b1, x0_cipher)
    x2 = linear_layer(HE, W2, b2, x1)
    x3 = linear_layer(HE, W3, b3, x2)
    return x3  # single-output => list of len=1

def main():
    HE = init_bfv_context()

    x0_val = 2
    print(f"Encrypting input x0 = {x0_val}")
    x0_cipher = [encrypt_int(HE, x0_val)]

    # Forward pass
    x3_cipher = forward_pass(HE, x0_cipher)

    final_val = decrypt_int(HE, x3_cipher[0])
    print("Decrypted final output:", final_val)

    # Plaintext reference
    def forward_pass_plain(x0: int):
        x1 = W1 @ [x0] + b1
        x2 = W2 @ x1 + b2
        x3 = W3 @ x2 + b3
        return x3[0]

    plain_result = forward_pass_plain(x0_val)
    print("Plaintext linear pass result:", plain_result)
    print("Difference =", final_val - plain_result)

if __name__ == "__main__":
    main()