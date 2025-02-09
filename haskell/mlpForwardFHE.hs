{-# LANGUAGE OverloadedStrings #-}

module Main where

import System.Process (readProcess)
import Control.Monad (foldM)
import Text.Printf (printf)

--------------------------------------------------------------------------------
-- 1) Helper calls to Python (fhe_ops.py)
--------------------------------------------------------------------------------

-- | runFheOps takes a list of args like ["encrypt", "5"] and
--   returns the stdout from fhe_ops.py.
runFheOps :: [String] -> IO String
runFheOps args = do
    -- Adjust this if "fhe_ops.py" isn’t in the same folder or in PATH
    let script = "./fhe_ops.py"
    output <- readProcess script args ""
    return (init output) -- remove trailing newline

-- | encrypt an Int, returning the ciphertext string (base64)
encryptVal :: Int -> IO String
encryptVal x = runFheOps ["encrypt", show x]

-- | homomorphic multiply a ciphertext by an Int
homMulInt :: String -> Int -> IO String
homMulInt ctxt scalar = runFheOps ["mult", ctxt, show scalar]

-- | homomorphic add two ciphertexts
homAdd :: String -> String -> IO String
homAdd c1 c2 = runFheOps ["add", c1, c2]

-- | decrypt a ciphertext string, returning the integer result
decryptVal :: String -> IO Int
decryptVal ctxt = do
    out <- runFheOps ["decrypt", ctxt]
    return (read out :: Int)

--------------------------------------------------------------------------------
-- 2) Homomorphic "matrix multiply + bias" for a single layer
--    We'll treat all weights as Ints. For a matrix W (size m x n) and
--    an input vector x of length n (encrypted each element),
--    we produce output vector y of length m (encrypted each element).
--------------------------------------------------------------------------------

-- We'll store a vector of ciphertexts as [String].
-- We'll store a matrix of plaintext integers as [[Int]].
-- We'll store a bias vector of plaintext integers as [Int].

-- singleLayer: y_i = sum_j( W[i][j] * x_j ) + b[i]
-- Because x_j is ciphertext, W[i][j] is int -> we do homMulInt
-- Then we fold homAdd across them. Finally we add the bias b[i].
-- But b[i] also must be turned into ciphertext with `encryptVal`.

singleLayer
    :: [[Int]]    -- ^ W (m x n)
    -> [Int]      -- ^ b (length m)
    -> [String]   -- ^ x (length n) - ciphertexts
    -> IO [String] -- ^ y (length m) - ciphertexts
singleLayer w b xCtxts = do
    let m = length w
    let n = length (head w)  -- assume rectangular
    -- For each row i in w, produce y_i
    mapM (computeRow xCtxts) (zip w b)
  where
    computeRow :: [String] -> ([Int], Int) -> IO String
    computeRow xCtxts (weights, biasVal) = do
        -- Multiply each x_j by W[i][j], then reduce
        --  partialSum_0 = w[i][0] * xCtxts[0]
        --  partialSum_1 = partialSum_0 + (w[i][1] * xCtxts[1])
        --  ...
        --  partialSum_{n-1} + bias
        let pairs = zip weights xCtxts
        -- Start with the first multiplication:
        let(w0, x0) = head pairs
        firstTerm <- homMulInt x0 w0

        -- fold over the remaining pairs with homAdd
        partialSum <- foldM
            (\acc (wi, xci) -> do
                multTerm <- homMulInt xci wi
                homAdd acc multTerm)
            firstTerm
            (tail pairs)

        -- Now add bias
        biasCtxt <- encryptVal biasVal
        homAdd partialSum biasCtxt

--------------------------------------------------------------------------------
-- 3) The forward pass (3-layer)
--    We'll keep the same shape as your sample code, but skip sigmoid.
--------------------------------------------------------------------------------

forwardPassFHE :: String -> IO String
forwardPassFHE x0Ctxt = do
    -- First layer: W1 is 3x1, b1 is length 3
    let w1 = [[ 1 ],  -- row 1
              [ 2 ],  -- row 2
              [-1 ]]  -- row 3
        b1 = [ 0,  0,  0 ]

    -- x0 is a single ciphertext, so x0Ctxts = [x0Ctxt]
    x1Ctxts <- singleLayer w1 b1 [x0Ctxt]  -- result is 3 ciphertexts

    -- Second layer: W2 is 3x3, b2 is length 3
    -- Suppose each row has 3 integers
    let w2 = [ [ 1, -1,  1 ],
               [ 0,  2,  1 ],
               [ 2,  1,  0 ] ]
        b2 = [ 1, 2, 3 ]
    x2Ctxts <- singleLayer w2 b2 x1Ctxts  -- 3 ciphertexts in -> 3 out

    -- Third layer: W3 is 1x3, b3 is length 1
    let w3 = [ [1, 1, 1] ]  -- 1 row, 3 columns
        b3 = [5]
    x3Ctxts <- singleLayer w3 b3 x2Ctxts  -- 3 in -> 1 out

    -- final output is a single ciphertext
    return (head x3Ctxts)

--------------------------------------------------------------------------------
-- 4) Main: demo
--------------------------------------------------------------------------------

main :: IO ()
main = do
    putStrLn "FHE MLP Forward Pass Demo"

    -- 1) We first encrypt the integer input, say 2
    putStrLn "Encrypting input = 2"
    x0Ctxt <- encryptVal 2

    -- 2) Run the forward pass
    putStrLn "Running forward pass homomorphically in Haskell (via Python calls)..."
    resultCtxt <- forwardPassFHE x0Ctxt

    -- 3) Decrypt the result
    decryptedResult <- decryptVal resultCtxt

    putStrLn $ "Decrypted result = " ++ show decryptedResult
    putStrLn "Done."
