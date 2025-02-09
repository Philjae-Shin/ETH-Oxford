#!/usr/bin/env python

import sys
import base64
import pickle
from Pyfhel import Pyfhel, PyCtxt
import numpy as np

"""
A small CLI utility that:
  - Generates a BFV context & keys (on every run, for demo).
  - Depending on command-line args, does:
      encrypt <int>         -> prints ciphertext base64
      mult <cipher_b64> <int> -> homomorphically multiply by an integer
      add <cipher_b64_1> <cipher_b64_2> -> homomorphically add two ciphertexts
      decrypt <cipher_b64>  -> prints the decrypted integer

In a real system, you'd re-use the same context/keys across calls
(but that means storing and reloading them from disk, or persisting in memory).
For a toy demo, we re-generate each time.
"""

# For this toy code, we’ll keep the BFV context in some global variable.
# But we generate it fresh each run. If you want consistent encryption
# across multiple calls, you'd need to keep the same keys from run to run.

#8192

HE = Pyfhel()
HE.contextGen(scheme='BFV', n=8192, t=65537)
HE.keyGen()

def b64_to_ctxt(b64string):
    # raw = base64.b64decode(b64string)
    # ctxt = HE.restoreCiphertext(raw)
    # return ctxt
    raw = base64.b64decode(b64string)
    ctxt = PyCtxt(pyfhel=HE)
    ctxt.from_bytes(raw)
    return ctxt

def ctxt_to_b64(ctxt):
    # raw = HE.serializeCiphertext(ctxt)
    raw = ctxt.to_bytes()
    return base64.b64encode(raw).decode('ascii')

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  fhe_ops.py encrypt <int>")
        print("  fhe_ops.py mult <cipher_b64> <int>")
        print("  fhe_ops.py add <cipher_b64_1> <cipher_b64_2>")
        print("  fhe_ops.py decrypt <cipher_b64>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "encrypt":
        if len(sys.argv) != 3:
            print("Usage: fhe_ops.py encrypt <int>")
            sys.exit(1)
        val = int(sys.argv[2])
        ptxt = HE.encodeInt(np.array([val], dtype=np.int64))
        ctxt = HE.encryptPtxt(ptxt)
        print(ctxt_to_b64(ctxt))

    elif mode == "mult":
        if len(sys.argv) != 4:
            print("Usage: fhe_ops.py mult <cipher_b64> <int>")
            sys.exit(1)
        ctxt_str = sys.argv[2]
        scalar = int(sys.argv[3])
        ctxt = b64_to_ctxt(ctxt_str)

        # Multiply in BFV by a plaintext integer (scalar)
        # Pyfhel allows direct multiplication by an encoded plaintext
        ptxt_scalar = HE.encodeInt(np.array([scalar], dtype=np.int64))
        ctxt_out = ctxt * ptxt_scalar
        print(ctxt_to_b64(ctxt_out))

    elif mode == "add":
        if len(sys.argv) != 4:
            print("Usage: fhe_ops.py add <cipher_b64_1> <cipher_b64_2>")
            sys.exit(1)
        ctxt_str1 = sys.argv[2]
        ctxt_str2 = sys.argv[3]
        ctxt1 = b64_to_ctxt(ctxt_str1)
        ctxt2 = b64_to_ctxt(ctxt_str2)

        ctxt_out = ctxt1 + ctxt2
        print(ctxt_to_b64(ctxt_out))

    elif mode == "decrypt":
        # if len(sys.argv) != 3:
        #     print("Usage: fhe_ops.py decrypt <cipher_b64>")
        #     sys.exit(1)
        # ctxt_str = sys.argv[2]
        # ctxt = b64_to_ctxt(ctxt_str)
        # ptxt = HE.decryptPtxt(ctxt)
        # arr = HE.decodeInt(ptxt)  # returns a numpy array
        # print(arr[0])            # just print the first int
        # If ciphertext is provided via stdin, read it.
        if not sys.stdin.isatty():
            ctxt_str = sys.stdin.read().strip()
        elif len(sys.argv) == 3:
            ctxt_str = sys.argv[2]
        else:
            print("Usage: fhe_ops.py decrypt <cipher_b64>")
            sys.exit(1)
        ctxt = b64_to_ctxt(ctxt_str)
        ptxt = HE.decryptPtxt(ctxt)
        arr = HE.decodeInt(ptxt)  # returns a numpy array
        print(arr[0])

    else:
        print("Unknown mode:", mode)
        sys.exit(1)

if __name__ == "__main__":
    main()
