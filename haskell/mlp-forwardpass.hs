module Main where

import qualified Numeric.LinearAlgebra as LA


-- Define the sigmoid activation function
sigmoid :: LA.Matrix Double -> LA.Matrix Double
sigmoid x = 1 / (1 + LA.cmap exp (LA.scale (-1) x))

-- Forward pass function for a 3-layer neural network
forwardPass :: LA.Matrix Double -> LA.Matrix Double
forwardPass x0 =
    let 
        -- 1st Layer: W1 is 3x1, b1 is 3x1.
        -- Replace the ellipsis with your actual numbers.
        w1 = (3 LA.>< 1) [0.5, 1.0, -0.5]   -- example values
        b1 = LA.asColumn (LA.vector [0.1, -0.2, 0.3])
        x1 = sigmoid $ (w1 LA.<> x0) + b1
        
        -- 2nd Layer: W2 is 3x3, b2 is 3x1.
        w2 = (3 LA.>< 3) [ 1.0, -0.5,  0.3,
                        0.7,  0.8, -0.2,
                       -0.3,  0.4,  0.9 ]
        b2 = LA.asColumn (LA.vector [0.0, 0.1, -0.1])
        x2 = sigmoid $ (w2 LA.<> x1) + b2

        -- 3rd Layer: W3 is 1x3, b3 is 1x1.
        w3 = (1 LA.>< 3) [ 1.2, -0.7, 0.5 ]
        b3 = LA.scalar 0.05
        x3 = (w3 LA.<> x2) + b3
    in x3

main :: IO ()
main = do
    -- Single input value: x0 = [[2.0]]
    let x0 = (1 LA.><1) [2.0]  -- 1x1 matrix containing the value 2.0
    putStrLn "Input:"
    print x0

    let result = forwardPass x0
    putStrLn "\nOutput after forward propagation:"
    print result

    -- If you had a target output and wanted to compute error, you could do so here.
