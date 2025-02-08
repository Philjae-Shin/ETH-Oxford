{-# LANGUAGE NumericUnderscores #-}
module Main where

{-
  Minimal LWE toy example in Haskell.

  Compile:
    ghc -package random LWEExample.hs -o LWEExample

  Run:
    ./LWEExample

  Note: It is NOT secure and does NOT represent a real-world, large-dimension LWE scheme.
-}

import System.Random (randomRIO)
import Control.Monad (replicateM)

--------------------------------------------------------------------------------
-- 1) Parameters
--------------------------------------------------------------------------------

q :: Int
q = 23  -- small prime modulus for demonstration

n :: Int
n = 4   -- dimension of secret s (toy, not secure)

m :: Int
m = 6   -- number of rows in matrix A, can be >= n

-- Error distribution range
errorMin, errorMax :: Int
errorMin = -1
errorMax = 1

--------------------------------------------------------------------------------
-- 2) Helper: modular arithmetic
--------------------------------------------------------------------------------

modQ :: Int -> Int
modQ x = ((x `mod` q) + q) `mod` q

addMod :: Int -> Int -> Int
addMod a b = modQ (a + b)

mulMod :: Int -> Int -> Int
mulMod a b = modQ (a * b)

--------------------------------------------------------------------------------
-- 3) Generate random vector/matrix
--------------------------------------------------------------------------------

-- Generate random integer in [0, q-1]
randModQ :: IO Int
randModQ = randomRIO (0, q-1)

-- Generate random vector of length len, each in [0, q-1]
randVector :: Int -> IO [Int]
randVector len = replicateM len randModQ

-- Generate random matrix of size (rows x cols)
randMatrix :: Int -> Int -> IO [[Int]]
randMatrix rows cols = replicateM rows (randVector cols)

-- Generate small error in [errorMin, errorMax]
randError :: IO Int
randError = randomRIO (errorMin, errorMax)

-- Generate random error vector of length len
randErrorVector :: Int -> IO [Int]
randErrorVector len = replicateM len randError

--------------------------------------------------------------------------------
-- 4) Key Generation (Toy LWE)
--------------------------------------------------------------------------------

-- Secret key s: random vector in [0, q-1]^n
-- Public key: (A, b) where b = A*s + e (mod q)
keyGen :: IO ([Int], [[Int]], [Int])
keyGen = do
    -- s in [0,q-1]^n
    s <- randVector n
    -- A in [0,q-1]^(m x n)
    a <- randMatrix m n
    -- e in (small error range)^(m)
    eVec <- randErrorVector m

    -- b = A*s + e (mod q), size m
    let b = map (\(row,e_i) -> modQ (dotProd row s + e_i)) (zip a eVec)
    return (s, a, b)

--------------------------------------------------------------------------------
-- 5) Encryption (Toy version)
--------------------------------------------------------------------------------
-- We'll represent ciphertext as (c1, c2):
--   c1 = a random combination of rows in A
--   c2 = same random combination applied to b plus message
-- In real LWE, we do it slightly differently, but let's do a simplified approach.

encrypt :: [[Int]] -> [Int] -> Int -> IO ([Int], Int)
encrypt a b msg = do
    -- pick a random r in {0,1}^m (or small range) - let's keep it tiny
    -- for a real LWE, we'd pick r in [0,q-1]^m or a small hamming weight vector
    r <- replicateM m (randomRIO (0::Int,1::Int))

    -- c1 = sum(r_i * row_i(A)) (mod q), dimension n
    let c1 = foldl (vecAddMod n) (replicate n 0)
               [ scalarVec r_i row | (r_i, row) <- zip r a ]

    -- c2 = sum(r_i * b_i) + msg (mod q)
    let c2 = modQ (dotProd r b + msg)

    return (c1, c2)

--------------------------------------------------------------------------------
-- 6) Decryption
--------------------------------------------------------------------------------
-- The user has secret s, can compute: c2 - c1 * s (mod q) = msg + some small error
-- We'll do a naive "round" to recover msg in {0,1}
decrypt :: [Int] -> ([Int], Int) -> Int
decrypt s (c1, c2) =
    let x = modQ (c2 - dotProd c1 s)
    in -- If x is close to 0 mod q => 0, if close to 1 => 1
       -- naive threshold:
       if x < (q `div` 2)
         then x  -- possibly 0 or small number
         else x - q -- negative range
    -- For a robust approach, we might do something like "round x to 0 or 1"
    -- but let's keep it simple:
    -- We'll interpret "abs(x) < abs(x-1)" -> 0 else 1.

decryptGuess :: [Int] -> ([Int], Int) -> Int
decryptGuess s ciph =
    let x = decrypt s ciph
    in if abs(x) <= abs(x-1) then 0 else 1

--------------------------------------------------------------------------------
-- 7) Utility: dot product and vector ops
--------------------------------------------------------------------------------

dotProd :: [Int] -> [Int] -> Int
dotProd xs ys = modQ (sum [ mulMod x y | (x,y) <- zip xs ys ])

scalarVec :: Int -> [Int] -> [Int]
scalarVec alpha vec = map (\v -> mulMod alpha v) vec

vecAddMod :: Int -> [Int] -> [Int] -> [Int]
vecAddMod _ xs ys = zipWith addMod xs ys

--------------------------------------------------------------------------------
-- 8) Main demonstration
--------------------------------------------------------------------------------
{-
 Steps:
   1) keyGen -> (s, A, b)
   2) encrypt a message (0 or 1)
   3) decrypt
-}

main :: IO ()
main = do
    putStrLn $ "==== Toy LWE Demo (mod " ++ show q ++ ", dimension n=" ++ show n ++ ", m=" ++ show m ++ ") ===="

    (s, a, b) <- keyGen
    putStrLn $ "Secret key s = " ++ show s
    putStrLn $ "Public key A = " ++ show a
    putStrLn $ "Public key b = " ++ show b

    let msg0 = 0
    ciphertext0 <- encrypt a b msg0
    let dec0 = decryptGuess s ciphertext0

    let msg1 = 1
    ciphertext1 <- encrypt a b msg1
    let dec1 = decryptGuess s ciphertext1

    putStrLn $ "\nEncrypting message 0 -> ciphertext: " ++ show ciphertext0
    putStrLn $   "Decrypt => " ++ show dec0

    putStrLn $ "\nEncrypting message 1 -> ciphertext: " ++ show ciphertext1
    putStrLn $   "Decrypt => " ++ show dec1

    putStrLn "\n(End of demo)"