#!/usr/bin/env python3
from Pyfhel import Pyfhel, PyPtxt
import numpy as np
import base64

def int_to_array(x: int) -> np.ndarray:
    """Convert an integer to a 1-element NumPy array with dtype=int64."""
    return np.array([x], dtype=np.int64)

# Create a global Pyfhel instance and generate context/keys.
HE = Pyfhel()  
HE.contextGen(scheme='BFV', n=8192, t=65537)
HE.keyGen()

def encrypt_two() -> bytes:
    """
    Encrypts the hardcoded integer 2 and returns the serialized ciphertext as bytes.
    """
    num = 2
    ptxt = HE.encodeInt(int_to_array(num))
    ctxt = HE.encryptPtxt(ptxt)
    # Serialize the ciphertext to bytes.
    serialized_ctxt = HE.serializeCiphertext(ctxt)
    return serialized_ctxt

def decrypt_ciphertext(ctxt_serialized: bytes) -> int:
    """
    Decrypts a given serialized ciphertext and returns the decrypted integer.
    """
    ctxt = HE.restoreCiphertext(ctxt_serialized)
    ptxt = HE.decryptPtxt(ctxt)
    arr = HE.decodeInt(ptxt)  # Returns a NumPy array.
    return int(arr[0])

def main():
    # Encrypt the value 2 and print the serialized ciphertext (base64 encoded for display).
    serialized = encrypt_two()
    # Optionally encode to base64 to get a readable string.
    b64_str = base64.b64encode(serialized).decode('ascii')
    print("Encrypted value (base64):", b64_str)
    
    # Decrypt the ciphertext and print the result.
    decrypted = decrypt_ciphertext(serialized)
    print("Decrypted value:", decrypted)

if __name__ == '__main__':
    main()
