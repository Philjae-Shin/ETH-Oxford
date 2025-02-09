import torch
import torch.nn as nn

import numpy as np

from Pyfhel import Pyfhel, PyPtxt
import numpy as np

def int_to_array(x: int) -> np.ndarray:
    """Convert an integer to a 1-element NumPy array with dtype=int64."""
    return np.array([x], dtype=np.int64)


####### NEURAL NETWORK ##########
def sigmoid(x):
    """
    Sigmoid activation function.
    Applies the function element-wise.
    """
    return 1 / (1 + np.exp(-x))

def forward_pass(x0):
    """
    Forward pass through a 3-layer neural network.
    
    Layer 1: 3 neurons
      - Weights: w1 (3×1)
      - Biases:  b1 (3×1)
    Layer 2: 3 neurons
      - Weights: w2 (3×3)
      - Biases:  b2 (3×1)
    Layer 3: 1 neuron
      - Weights: w3 (1×3)
      - Biases:  b3 (1×1)
    """
    # --- 1st Layer ---
    # w1 is a 3x1 matrix, b1 is a 3x1 column vector.
    w1 = np.array([[5],
                   [10],
                   [-5]])
    b1 = np.array([[1],
                   [-2],
                   [3]])
    # Multiply: (3x1) dot (1x1) -> (3x1), then add bias and apply sigmoid.
    x1 = np.dot(w1, x0) + b1
    
    # --- 2nd Layer ---
    # w2 is a 3x3 matrix, b2 is a 3x1 column vector.
    w2 = np.array([[ 1, -5,  3],
                   [ 7,  8, -2],
                   [-3,  4,  9]])
    b2 = np.array([[0],
                   [1],
                   [-1]])
    x2 = np.dot(w2, x1) + b2
    
    # --- 3rd Layer ---
    # w3 is a 1x3 matrix, b3 is a scalar (1x1 matrix).
    w3 = np.array([[ 1, -7, 5]])
    b3 = np.array([[5]])
    x3 = np.dot(w3, x2) + b3
    
    return x3



def main():
    # Initialize the Pyfhel object and context for BFV.
    HE = Pyfhel()  
    HE.contextGen(scheme='BFV', n=8192, t=65537)
    HE.keyGen()  # Generate public and secret keys

    # Create a single input value x0 = [[2.0]]
    x0 = HE.encodeInt(int_to_array(10)) # shape (1,1)
    x0 = HE.encryptPtxt(x0)
    
    # Compute forward propagation.
    result = forward_pass(x0)
    result = result[0,0] 
    result = HE.decryptPtxt(result)
    result = HE.decodeInt(result)
    print("\nOutput after forward propagation using FHE:")
    print(result[0])

    # Instantiate the model
    model = SimpleNN()

    # Define input (equivalent to `x0 = np.array([[10]])` in NumPy)
    x0 = torch.tensor([[10.0]])  # Shape (1,1)

    # Perform forward pass
    output = model(x0)

    # Print output
    print("\nOutput after forward propagation pyTorch library:")
    print(output.item())  # Extract scalar value
    if result[0] == output.item():
        print("The fully homomorphic encription forward propagation result is correct!")


class SimpleNN(nn.Module):
  def __init__(self):
      super(SimpleNN, self).__init__()
      
      # Define weights and biases exactly as in NumPy implementation
      self.w1 = nn.Parameter(torch.tensor([[5.0], [10.0], [-5.0]]))  # (3x1)
      self.b1 = nn.Parameter(torch.tensor([[1.0], [-2.0], [3.0]]))  # (3x1)

      self.w2 = nn.Parameter(torch.tensor([[ 1.0, -5.0,  3.0],
                                            [ 7.0,  8.0, -2.0],
                                            [-3.0,  4.0,  9.0]]))  # (3x3)
      self.b2 = nn.Parameter(torch.tensor([[0.0], [1.0], [-1.0]]))  # (3x1)

      self.w3 = nn.Parameter(torch.tensor([[ 1.0, -7.0, 5.0]]))  # (1x3)
      self.b3 = nn.Parameter(torch.tensor([[5.0]]))  # (1x1)

  def forward(self, x0):
      """
      Performs a forward pass with NO activation functions.
      """
      x1 = torch.matmul(self.w1, x0) + self.b1  # First layer
      x2 = torch.matmul(self.w2, x1) + self.b2  # Second layer
      x3 = torch.matmul(self.w3, x2) + self.b3  # Third layer
      return x3  # Final output






if __name__ == "__main__":
    main()

