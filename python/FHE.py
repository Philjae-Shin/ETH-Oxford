#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Pyfhel import Pyfhel, PyPtxt, PyCtxt

def main():
    # 1) Create a Pyfhel object
    HE = Pyfhel()
    print("[DEBUG] Pyfhel object created.")

    # 2) Generate a BFV context
    #    - n=16384, t=65537 are example parameters
    #    - Note that n(=2^14) is relatively large, so this may take some time in a demo.
    print("[DEBUG] Generating BFV context...")
    HE.contextGen(scheme='BFV', n=16384, t=65537)
    print("[DEBUG] Context generated.\n")

    # 3) Generate keys (secret key, public key, etc.)
    print("[DEBUG] Generating keys (secret/public, etc.)...")
    HE.keyGen()
    print("[DEBUG] Key generation done.\n")

    # 4) Define an example integer list
    plaintext_list = [1, 2, 3, 4, 5]
    print("[DEBUG] Original plaintext list:", plaintext_list)

    # 5) Encode the integer list into a Pyfhel format
    #    (For BFV, usually encodeInt is used)
    ptxt = HE.encodeInt(plaintext_list)
    print("[DEBUG] Encoded plaintext (PyPtxt):", ptxt, "\n")

    # 6) Encrypt
    ctxt = HE.encrypt(ptxt)
    print("[DEBUG] Encrypted ciphertext (PyCtxt) - showing partial bytes:")
    print(ctxt.toBytes()[:60], "... (truncated)\n")

    # 7) Decrypt
    decrypted_list = HE.decrypt(ctxt)  # BFV: decrypt returns an integer list
    print("[DEBUG] Decrypted list:", decrypted_list, "\n")

    # 8) Check the result
    if decrypted_list == plaintext_list:
        print("Encryption and Decryption succeeded! ✅")
    else:
        print("Encryption and Decryption failed. ❌")


if __name__ == "__main__":
    main()
