from Pyfhel import Pyfhel, PyCtxt

def main():
    # Initialize the Pyfhel object
    HE = Pyfhel()  
    # Generate a context for the BFV scheme.
    # Here we choose a plaintext modulus p and a polynomial modulus m.
    # p=65537 is a common choice, and m must be a power-of-2 (here 8192).
    HE.contextGen(p=65537, m=8192)
    HE.keyGen()  # Generate public and secret keys

    # Read two integer inputs from the user.
    try:
        num1 = int(input("Enter the first integer: "))
        num2 = int(input("Enter the second integer: "))
    except ValueError:
        print("Invalid input. Please enter valid integers.")
        return

    # Encrypt the integers using Pyfhel's encryptInt function.
    ctxt1 = HE.encryptInt(num1)
    ctxt2 = HE.encryptInt(num2)

    # Print intermediate ciphertexts (Note: these are not human-readable, but show object info).
    print("\n--- Intermediate Encrypted Data ---")
    print("Ciphertext 1:", ctxt1)
    print("Ciphertext 2:", ctxt2)

    # Homomorphically add the two ciphertexts.
    ctxt_sum = ctxt1 + ctxt2
    print("\nEncrypted sum (ciphertext):", ctxt_sum)

    # Decrypt the resulting ciphertext.
    decrypted_sum = HE.decryptInt(ctxt_sum)
    print("\nDecrypted sum:", decrypted_sum)

    # Check if the result is correct.
    expected_sum = num1 + num2
    print("Expected sum:", expected_sum)
    if decrypted_sum == expected_sum:
        print("The homomorphic addition result is correct!")
    else:
        print("There is an error in the homomorphic computation.")

if __name__ == '__main__':
    main()
