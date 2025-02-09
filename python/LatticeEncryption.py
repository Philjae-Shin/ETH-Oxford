import numpy as np
from numpy.polynomial import Polynomial as Poly
def sample_small_poly(n, sigma):
    return Poly(np.round(np.random.normal(0, sigma, n)) % q)
# Sample a uniform polynomial (used for public key generation)
def sample_uniform_poly(n, q):
    return Poly(np.random.randint(0, q, n))  # Random coefficients in [0, q)
def generate_keys(n, q, sigma):
    """
    Generate Ring-LWE keys: (public key, private key)
    """
    s = sample_small_poly(n, sigma)  # Secret key (small polynomial)
    a = sample_uniform_poly(n, q)  # Random public polynomial
    e = sample_small_poly(n, sigma)  # Error polynomial (small noise)

    # Compute b = a * s + e mod q
    b = (a * s + e) % q
    public_key = (a, b)
    private_key = s

    return public_key, private_key

# Generate keys
public_key, private_key = generate_keys(n, q, sigma)
print("Public Key (a, b):", public_key)
print("Private Key (s):", private_key)

def encrypt(public_key, message, n, q, sigma):
    """
    Encrypt a message using Ring-LWE encryption.
    """
    a, b = public_key
    m = Poly([message] + [0] * (n - 1))  # Encode message as a polynomial
    # Sample small noise polynomials
    e1 = sample_small_poly(n, sigma)
    e2 = sample_small_poly(n, sigma)
    u = sample_small_poly(n, sigma)  # Random small polynomial

    # Compute ciphertext c1 and c2
    c1 = (a * u + e1) % q
    c2 = (b * u + e2 + (q // 2) * m) % q  # Embed message in c2

    return c1, c2

# Encrypt message
message = 100  # Message should be small (binary {0,1})
ciphertext = encrypt(public_key, message, n, q, sigma)
print("Ciphertext (c1, c2):", ciphertext)

def decrypt(private_key, ciphertext, n, q):
    """
    Decrypt a Ring-LWE ciphertext.
    """
    s = private_key
    c1, c2 = ciphertext
    # Compute v = c2 - s * c1 mod q
    v = (c2 - s * c1) % q
    # Decode message: check if v is closer to 0 or q/2
    threshold = q // 4  # Halfway between 0 and (q/2)
    m_decoded = 1 if np.round(v.coef[0]) > threshold else 0
    return m_decoded

# Decrypt ciphertext
decrypted_message = decrypt(private_key, ciphertext, n, q)
print("Decrypted Message:", decrypted_message)
