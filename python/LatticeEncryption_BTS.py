#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import LatticeEncryption  # The original code (not modified here)
import numpy as np
import math
from numpy.ma.core import innerproduct, transpose

############################################################
# (1) Simple data structures for key switching and
#     dimension/modulus reduction
############################################################

def gen_switching_matrix(sh_privkey, hat_s, n, k, q, p):
    """
    [Academically simplified example]

    Generates a "switching matrix" (KSW) to transform from the old secret key
    (sh_privkey, dimension=n) to the new secret key (hat_s, dimension=k).

    In real GSW/BGV/CKKS-style schemes, you'd need:
      - Polynomial basis transformations
      - Gadget decomposition
      - RNS base changes
      and more.

    Here, we simply create a random matrix that (in a very contrived way)
    encodes some relationship between (sh_privkey, hat_s).
    """

    # Assume sh_privkey.shape = (n,) and hat_s.shape = (k,).
    # Suppose KSW has shape (n, k). In reality, it might be more complex.

    KSW = np.zeros((n, k), dtype=int)

    for i in range(n):
        for j in range(k):
            # Example (completely meaningless cryptographically):
            #   random value + s_i * (j+1) - hat_s_j * (i+1), all mod q
            rand_val = np.random.randint(0, q)
            KSW[i, j] = (rand_val
                         + int(sh_privkey[i]) * (j+1)
                         - int(hat_s[j]) * (i+1)) % q
    return KSW

def dimension_modulus_reduction(long_cipher, KSW, q, p):
    (v, w) = long_cipher
    n, k = KSW.shape

    v_hat = np.zeros(k, dtype=int)
    for j in range(k):
        acc = 0
        for i in range(n):
            row_sum = int(np.sum(v[i]))  # (m,) -> 스칼라
            tmp = (row_sum * KSW[i, j]) % q
            acc = (acc + tmp) % q

        v_hat[j] = acc % p

    w_hat = w % q
    w_hat = w_hat % p
    return (v_hat, w_hat)

    """
    [Simplified dimension/modulus reduction function]

    long_cipher = (v, w) with dimension n, modulus q
    short_cipher = (v_hat, w_hat) with dimension k, modulus p

    In reality, you'd need gadget decomposition, rounding, etc.
    Here, we simply assume we use KSW to "map" (v, w) down to dimension k,
    then reduce from q to p.
    """

    (v, w) = long_cipher
    n, k = KSW.shape

    # 1) Dimension reduce v -> v_hat
    #    For instance,
    #       v_hat_j = sum over i of [ v_i * KSW[i, j] ] mod p
    #    (highly simplified; real key switching uses a more elaborate approach)
    v_hat = np.zeros(k, dtype=int)
    for j in range(k):
        acc = 0
        for i in range(n):
            # v_i (mod q) * KSW[i,j] (mod q) => then mod p
            tmp = (v[i] * KSW[i, j]) % q
            acc = (acc + tmp) % q
        # Now reduce from q to p (e.g., rounding or simple mod p)
        v_hat[j] = acc % p

    # 2) w_hat = w mod q -> then mod p
    w_hat = w % q
    w_hat = w_hat % p

    return (v_hat, w_hat)

############################################################
# (2) BTS KeyGen
############################################################

def bts_keygen(sh_privkey, q,
               k=16,    # e.g., the "reduced" dimension
               p=127):  # e.g., the "reduced" modulus
    """
    Generates a new secret key hat_s for the "short dimension/modulus" scheme (BTS),
    and a key switching matrix (KSW) etc. as the evaluation key (hat_Psi).

    :param sh_privkey: The existing secret key from LatticeEncryption.py (length n)
    :param q: The original modulus
    :param k: The new, reduced dimension
    :param p: The new, reduced modulus
    :return: (hat_s, hat_Psi, p, k)
    """
    n = len(sh_privkey)

    # New secret key hat_s (dimension k, 0/1 vector in this simple example)
    hat_s = np.random.randint(0, 2, size=k)

    # Generate a switching matrix, or something analogous
    KSW = gen_switching_matrix(sh_privkey, hat_s, n, k, q, p)

    # The evaluation key hat_Psi = (KSW, possibly more data...)
    hat_Psi = {
        'KSW': KSW
        # In reality: bootstrapping circuits, GaloisKey, RotationKey, etc.
    }

    return hat_s, hat_Psi, p, k

############################################################
# (3) Encryption/Decryption in the "short dimension/modulus"
############################################################

def bts_encrypt(bit, hat_s, p):
    """
    Encryption in the short dimension (k) with modulus p.
    The ciphertext: (v_hat, w_hat).

    In real BTS:
      SH.Enc -> dimension_modulus_reduction -> bts_keyswitch ...
    Here, we keep it simple.

    :param bit: The plaintext bit (0 or 1)
    :param hat_s: The short secret key
    :param p: The short modulus
    :return: (v_hat, w_hat)
    """
    k = len(hat_s)
    v_hat = np.random.randint(0, p, size=k)
    # small noise e
    e_hat = np.random.randint(0, 2)
    w_hat = (bit + np.dot(v_hat, hat_s) + e_hat) % p
    return (v_hat, w_hat)

def bts_decrypt(ct_hat, hat_s, p):
    """
    Decryption in the short dimension.

    :param ct_hat: (v_hat, w_hat)
    :param hat_s: The short secret key
    :param p: The short modulus
    :return: The recovered bit (0 or 1)
    """
    (v_hat, w_hat) = ct_hat
    val = (w_hat - np.dot(v_hat, hat_s)) % p
    return val % 2

############################################################
# (4) Key Switching (from "long" ciphertext -> "short" ciphertext)
############################################################

def bts_keyswitch(long_cipher, hat_Psi, q, p):
    """
    [Assumption] long_cipher = (v, w) in dimension n, modulus q.

    We use hat_Psi['KSW'] to run dimension_modulus_reduction -> (v_hat, w_hat).

    Real BTS is more involved, but we'll just call dimension_modulus_reduction.
    """
    KSW = hat_Psi['KSW']
    short_cipher = dimension_modulus_reduction(long_cipher, KSW, q, p)
    return short_cipher

############################################################
# (5) Bootstrapping (simplified)
############################################################

def bts_bootstrap(sh_cipher, sh_privkey, q,
                  hat_Psi, hat_s, p):
    """
    1) The old scheme ciphertext sh_cipher = ((v, w), ...)
    2) Homomorphic or actual decryption -> get the bit
    3) Re-encrypt into the "short" scheme

    In a real scenario, you'd do "homomorphic decryption" to reduce noise.
    But here, we simply do actual decryption -> bts_encrypt.
    """
    # Suppose sh_cipher is [ (v, w), depth ] as in LatticeEncryption.py
    # We can call LatticeEncryption.decryption(...) to get the plaintext bit.
    bit_plain = LatticeEncryption.decryption(sh_cipher, sh_privkey, q)

    # Then re-encrypt the bit in the short dimension
    ct_short = bts_encrypt(bit_plain, hat_s, p)
    return ct_short

############################################################
# (6) Homomorphic operations on "short" ciphertexts (example)
############################################################

def bts_eval_add(ct1, ct2, p):
    """
    Example of homomorphic addition on short ciphertexts.
    """
    v1, w1 = ct1
    v2, w2 = ct2
    v_new = (v1 + v2) % p
    w_new = (w1 + w2) % p
    return (v_new, w_new)

def bts_eval_mul(ct1, ct2, hat_s, p):
    """
    For a real FHE multiplication, noise grows significantly,
    requiring bootstrapping or key switching.
    Here we simply "partial-decrypt -> multiply -> re-encrypt" as a toy example.
    """
    bit1 = bts_decrypt(ct1, hat_s, p)
    bit2 = bts_decrypt(ct2, hat_s, p)
    product = (bit1 * bit2) % 2
    return bts_encrypt(product, hat_s, p)

############################################################
# (7) Main Test
############################################################

if __name__ == "__main__":
    # 1) Generate SH keys from the original code
    #    privatekey, A, b, q = keyGen(50,30)
    sh_privkey, A, b, q = LatticeEncryption.keyGen(n=50, m=30)
    #  -> sh_privkey is length ~n=50 (actually the last noise vector etc.)

    # 2) Generate BTS keys (short dimension/modulus)
    hat_s, hat_Psi, p, k = bts_keygen(sh_privkey, q, k=16, p=257)
    # Example: p=257 (a prime), although real usage is more specialized.

    # 3) For example, encrypt bit=1 in the old scheme
    ct_sh = LatticeEncryption.encryption(A, b, 30, 1)
    # Suppose the result is ((v, w), depth)

    # 4) Bootstrapping (simplified):
    #    - In reality, a homomorphic circuit for decryption -> dimension reduction
    #    - Here, just do actual decryption -> bts_encrypt
    ct_bts = bts_bootstrap(ct_sh, sh_privkey, q, hat_Psi, hat_s, p)

    # 5) Check decryption
    dec_bit = bts_decrypt(ct_bts, hat_s, p)
    print(f"BTS decrypted bit: {dec_bit}")

    # 6) Demo: short ciphertext addition/multiplication
    ct_bts_zero = bts_encrypt(0, hat_s, p)
    ct_sum = bts_eval_add(ct_bts, ct_bts_zero, p)
    ct_mul = bts_eval_mul(ct_bts, ct_bts, hat_s, p)

    print("Add result:", bts_decrypt(ct_sum, hat_s, p))
    print("Mul result:", bts_decrypt(ct_mul, hat_s, p))

    # 7) Key switching test (conceptual)
    #    - (v, w) in dimension n, modulus q -> dimension k, modulus p
    #    - For example, directly KS the old ct_sh (though it should be done homomorphically).
    v_sh, w_sh = ct_sh[0]  # (v, w)
    reduced_ct = bts_keyswitch((v_sh, w_sh), hat_Psi, q, p)
    # reduced_ct = (v_hat, w_hat) in dimension k, modulus p

    # Decrypt with the short secret key (in practice, there's more to consider).
    print("KeySwitch-based short cipher -> decrypt (rough example):",
          bts_decrypt(reduced_ct, hat_s, p))
