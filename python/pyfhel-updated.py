# from Pyfhel import Pyfhel, PyCtxt

# def main():
#     # Initialize the Pyfhel object
#     HE = Pyfhel()  
#     # Generate a context for the BFV scheme.
#     # Here we choose a plaintext modulus p and a polynomial modulus m.
#     # p=65537 is a common choice, and m must be a power-of-2 (here 8192).
#     HE.contextGen(scheme='BFV', n=8192, t=65537)
#     HE.keyGen()  # Generate public and secret keys

#     # Read two integer inputs from the user.
#     try:
#         num1 = int(input("Enter the first integer: "))
#         num2 = int(input("Enter the second integer: "))
#     except ValueError:
#         print("Invalid input. Please enter valid integers.")
#         return

#     # Encrypt the integers using Pyfhel's encryptInt function.
#     ctxt1 = HE.encryptInt(num1)
#     ctxt2 = HE.encryptInt(num2)

#     # Print intermediate ciphertexts (Note: these are not human-readable, but show object info).
#     print("\n--- Intermediate Encrypted Data ---")
#     print("Ciphertext 1:", ctxt1)
#     print("Ciphertext 2:", ctxt2)

#     # Homomorphically add the two ciphertexts.
#     ctxt_sum = ctxt1 + ctxt2
#     print("\nEncrypted sum (ciphertext):", ctxt_sum)

#     # Decrypt the resulting ciphertext.
#     decrypted_sum = HE.decryptInt(ctxt_sum)
#     print("\nDecrypted sum:", decrypted_sum)

#     # Check if the result is correct.
#     expected_sum = num1 + num2
#     print("Expected sum:", expected_sum)
#     if decrypted_sum == expected_sum:
#         print("The homomorphic addition result is correct!")
#     else:
#         print("There is an error in the homomorphic computation.")

# if __name__ == '__main__':
#     main()

from Pyfhel import Pyfhel, PyPtxt
import numpy as np

def int_to_array(x: int) -> np.ndarray:
    """Convert an integer to a 1-element NumPy array with dtype=int64."""
    return np.array([x], dtype=np.int64)

def main():
    # Initialize the Pyfhel object and context for BFV.
    HE = Pyfhel()  
    HE.contextGen(scheme='BFV', n=8192, t=65537)
    HE.keyGen()  # Generate public and secret keys

    # Hardcoded integer values:
    num1 = 2
    num2 = 5

    # Convert integers to a NumPy int64 array and encode.
    ptxt1 = HE.encodeInt(int_to_array(num1))
    ptxt2 = HE.encodeInt(int_to_array(num2))
    
    # Encrypt the encoded plaintexts.
    ctxt1 = HE.encryptPtxt(ptxt1)
    ctxt2 = HE.encryptPtxt(ptxt2)

    # Print intermediate ciphertexts.
    print("\n--- Intermediate Encrypted Data ---")
    print("Ciphertext 1:", ctxt1)
    print("Ciphertext 2:", ctxt2)


    # Homomorphically add the two ciphertexts.
    ctxt_sum = ctxt1 + 2*ctxt2 + 3
    ctxt_sum_array = np.array([ctxt_sum], dtype=object)
    print(ctxt_sum_array)
    tmp = HE.decryptPtxt(ctxt_sum_array[0])
    temp = np.dot(2, ctxt1) + 1
    temp = HE.decryptPtxt(temp)
    temp = HE.decodeInt(temp)
    print(temp)
    print("\nEncrypted sum (ciphertext):", ctxt_sum)

    # Decrypt the resulting ciphertext using decryptPtxt to obtain a PyPtxt object.
    ptxt_sum = HE.decryptPtxt(ctxt_sum)
    decoded_sum = HE.decodeInt(ptxt_sum)
    # decoded_sum is a NumPy array (e.g., array([sum], dtype=int64))
    result_int = int(decoded_sum[0])
    print("\nDecrypted sum:", result_int)

    # Check if the result is correct.
    expected_sum = num1 + num2
    print("Expected sum:", expected_sum)
    if result_int == expected_sum:
        print("The homomorphic addition result is correct!")
    else:
        print("There is an error in the homomorphic computation.")

if __name__ == '__main__':
    main()