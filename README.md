# ETH-Oxford
# Confidential Privacy AI with Fully Homomorphic Encryption

Welcome to our project for the ETH Oxford Hackathon! Our team tackled the challenging theme of **Quantum Cryptography** by focusing on Fully Homomorphic Encryption (FHE) to enable confidential privacy in AI model training.

## Overview

In our project we implement a three-layer neural network that performs forward propagation on **encrypted data**. Our goal is to train AI models without ever exposing the underlying (plaintext) data. Using FHE, our network can process data while it remains encrypted, ensuring data privacy and security.

Key points include:
- **Fully Homomorphic Encryption (FHE):** We implemented our own encryption (see our lattice encryption file) and are actively working towards bootstrapping.
- **Neural Network Forward Propagation:** Our demo (`neuralnet-forwardpropag.py`) implements a 3-layer neural network. Using FHE, we verify the correctness of the forward propagation computation.
- **Confidential Model Training:** By training on encrypted data, the model has no access to the original plaintext inputs.
- **Potential for Lazy Evaluation:** Our design leaves room for further exploration on using lazy evaluation to reduce the number of operations (thus reducing noise), which could optimize overall performance.
- **Modular Implementation:** Our FHE operations are implemented separately (see `LatticeEncryption.py`), and we integrate these into our homomorphic forward propagation (`mlpForwardFHE.hs` and `neuralnet-forwardpropag.py`).

## Key Files

- **`neuralnet-forwardpropag.py`**  
  This file contains the Pyfhel-based implementation of the forward propagation in our encrypted neural network. It demonstrates how encrypted data can be processed, decrypted, and compared to a PyTorch model for correctness.

- **`LatticeEncryption.py`**  
  Implements our lattice-based encryption and decryption functions using Ring-LWE. This file forms the basis of our FHE scheme.

- **`mlpForwardFHE.hs`**  
  A Haskell implementation showcasing our homomorphic operations and integration with Python for FHE tools. Most of the Haskell files were stepping stones that helped us arrive at our finished program.

## Project Status and Next Steps

We have successfully verified the correctness of our homomorphic forward propagation in a three-layer network. The encryption pipeline preserves data privacy while still enabling the core mathematical operations required for neural network training. Current research directions include:
- **Bootstrapping:** Advancing beyond the current noise budget limitations.
- **Lazy Evaluation:** Investigating techniques to optimize computations and reduce the amount of operations on the data.
- **Model Training on Encrypted Data:** Expanding our pipeline to incorporate full model training on encrypted inputs.

## How to Run

*Installation instructions and detailed running steps will be specified later. For now, focus on the following main files:*
- `neuralnet-forwardpropag.py`
- `LatticeEncryption.py`
- `mlpForwardFHE.hs`

Example command for Python (ensure Pyfhel and PyTorch are installed):
```bash
python neuralnet-forwardpropag.py
```

For the Haskell portion, if desired:
```bash
ghc mlpForwardFHE.hs -o mlpForwardFHE && ./mlpForwardFHE
```

## Conclusion

Our project demonstrates the potential of using FHE to build AI models that never access the underlying plaintext data, thereby revolutionizing data privacy in machine learning. We look forward to further optimizing the system and exploring advanced features like bootstrapping and lazy evaluation.

---

*For further updates and detailed instructions on running the project, please check back or contact our team.*
Blockchain-themed hackathon at Oxford University
