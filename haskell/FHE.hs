module Main where

import System.Random
import Data.Bits (shiftR)
import Data.List (foldl')
import Control.Monad (replicateM)

--------------------------------------------------------------------------------
-- Practical Parameter Considerations
--------------------------------------------------------------------------------

--   Typically, q would be on the order of 2^30 or larger, and n could be in the
--   thousands for adequate security against known attacks.
data FHEParams = FHEParams
  { fheN         :: Int      -- dimension (n)
  , fheEll       :: Int      -- number of samples for public key
  , fheQ         :: Integer  -- modulus (q)
  , fheT         :: Integer  -- plaintext space representative (often t=2 or similar)
  , fheNoiseBd   :: Int      -- noise bound for sampling
  } deriving (Show, Eq)

-- Example (still small) parameters:
defaultParams :: FHEParams
defaultParams = FHEParams
  { fheN       = 512
  , fheEll     = 600
  , fheQ       = 2^(20) + 3 -- Example: ~1 million
  , fheT       = 2
  , fheNoiseBd = 8
  }

--------------------------------------------------------------------------------
-- Key & Cipher Structures
--------------------------------------------------------------------------------

-- | SecretKey: a list of integers as the secret vector s.
newtype SecretKey = SK { unSK :: [Integer] }
  deriving (Show, Eq)

-- | PublicKey: (a_i, b_i) pairs. For convenience, we store a_i in a matrix,
--   b_i in a list.
data PublicKey = PK
  { pkA :: [[Integer]]  -- Dimension: ell x n
  , pkB :: [Integer]    -- Dimension: ell
  } deriving (Show, Eq)

-- | LWE-based ciphertext: (alpha, beta) where alpha is a vector, beta is an integer.
data Ciphertext = CT
  { ctAlpha :: [Integer]
  , ctBeta  :: Integer
  } deriving (Show, Eq)

--------------------------------------------------------------------------------
-- Modular Arithmetic
--------------------------------------------------------------------------------

-- | Modular reduction with respect to q
modQ :: FHEParams -> Integer -> Integer
modQ params x = x `mod` fheQ params

-- | Batch modular reduction over a list
modQList :: FHEParams -> [Integer] -> [Integer]
modQList params = map (modQ params)

--------------------------------------------------------------------------------
-- Randomness and Noise
--------------------------------------------------------------------------------

-- | Generate a random integer in [0, q-1].
getRand :: FHEParams -> IO Integer
getRand params = do
  let qVal = fheQ params
  r <- randomRIO (0, qVal - 1)
  return r

-- | Sample noise from a small uniform distribution in [-noiseBd, noiseBd].
--   In production, a discrete Gaussian (with cryptographically secure PRNG)
--   is typically used.
getNoise :: FHEParams -> IO Integer
getNoise params = do
  let bd = fheNoiseBd params
  r <- randomRIO (-bd, bd)
  return (toInteger r)

--------------------------------------------------------------------------------
-- Key Generation
--------------------------------------------------------------------------------

-- | Generate LWE keys. The secret key s has dimension n.
--   The public key has ell samples, each sample (a_i, b_i).
--   b_i = -(a_i · s) + e_i (mod q)
keyGen :: FHEParams -> IO (SecretKey, PublicKey)
keyGen params = do
  -- Secret key s
  sVals <- replicateM (fheN params) (getRand params)
  let s = map (modQ params) sVals

  -- Create a_i, and compute b_i
  let ell = fheEll params
  aMat <- mapM (\_ -> replicateM (fheN params) (getRand params)) [1..ell]
  eVec <- mapM (\_ -> getNoise params) [1..ell]

  -- b_i = -(a_i · s) + e_i (mod q)
  let bList = zipWith (\aRow e ->
                         let dotProd = sum (zipWith (*) aRow s) `mod` fheQ params
                             val     = (- dotProd + e) `mod` fheQ params
                         in val
                      ) aMat eVec

  return (SK s, PK aMat bList)

--------------------------------------------------------------------------------
-- Encryption & Decryption
--------------------------------------------------------------------------------

-- | Encrypt a message m in {0, 1} (or more generally in [0, t-1]) into an LWE ciphertext.
--   We encode m by multiplying with (q / t). For t=2, that is roughly q/2.
encrypt :: FHEParams -> PublicKey -> Integer -> IO Ciphertext
encrypt params (PK aMat bList) m = do
  let ell = fheEll params
      qVal = fheQ params
      -- Scale message from [0..t-1] to an integer near q
      mScaled = (m * (qVal `div` fheT params)) `mod` qVal

  -- Generate random r_i in {0,1} or small subset. For better security,
  -- we often sample from a binomial or similar distribution.
  rVec <- replicateM ell (randomRIO (0,1) :: IO Integer)
  e    <- getNoise params

  -- alpha = Σ_i r_i * a_i (mod q)
  let alpha = foldl' (zipWith (+)) (replicate (fheN params) 0) $
              zipWith (\r aRow -> map (* r) aRow) rVec aMat
      alphaMod = modQList params alpha

  -- beta = Σ_i r_i * b_i + mScaled + e (mod q)
  let beta = foldl' (+) 0 (zipWith (*) rVec bList)
      betaM = (beta + mScaled + e) `mod` qVal

  return $ CT alphaMod betaM

-- | Decrypt the ciphertext (alpha, beta) using s.
--   Recovers m by checking if (beta + alpha·s) is closer to 0 or q/2 (for t=2).
decrypt :: FHEParams -> SecretKey -> Ciphertext -> Integer
decrypt params (SK s) (CT alpha beta) =
  let qVal   = fheQ params
      halfQ  = qVal `div` 2
      dotVal = sum (zipWith (*) alpha s) `mod` qVal
      c      = (beta + dotVal) `mod` qVal
  in if c >= halfQ then 1 else 0

--------------------------------------------------------------------------------
-- Homomorphic Operations
--------------------------------------------------------------------------------

-- | Ciphertext addition (mod q).
--   (alpha1 + alpha2, beta1 + beta2)
addCT :: FHEParams -> Ciphertext -> Ciphertext -> Ciphertext
addCT params (CT a1 b1) (CT a2 b2) =
  let alpha' = zipWith (+) a1 a2
      alphaMod = modQList params alpha'
      betaMod = modQ params (b1 + b2)
  in CT alphaMod betaMod

-- | Ciphertext subtraction (optional).
subCT :: FHEParams -> Ciphertext -> Ciphertext -> Ciphertext
subCT params (CT a1 b1) (CT a2 b2) =
  let alpha' = zipWith (-) a1 a2
      alphaMod = modQList params alpha'
      betaMod = modQ params (b1 - b2)
  in CT alphaMod betaMod

-- | Ciphertext multiplication (naive version).
--   In a real FHE scheme, after multiplication, we must do "relinearization"
--   to reduce the ciphertext back to a normal LWE form. Otherwise, alpha
--   effectively becomes a "second-degree" object.
multCT :: FHEParams -> Ciphertext -> Ciphertext -> Ciphertext
multCT params (CT a1 b1) (CT a2 b2) =
  let -- naive approach: alpha' = a1*b2 + a2*b1
      alpha1 = map (* b2) a1
      alpha2 = map (* b1) a2
      alpha' = zipWith (+) alpha1 alpha2
      alphaMod = modQList params alpha'

      -- the product of the message part: b1*b2
      beta' = (b1 * b2) `mod` fheQ params
  in CT alphaMod beta'

--------------------------------------------------------------------------------
-- Relinearization (Simplified Stub)
--------------------------------------------------------------------------------

-- | In a true LWE-based FHE, relinearization uses additional key-switching data
--   to convert from a "degree-2" ciphertext to a "degree-1" ciphertext.
--   Here, we show only a simplified placeholder that scales alpha down, etc.
relinearize :: FHEParams -> PublicKey -> Ciphertext -> Ciphertext
relinearize params (PK _ _) (CT alpha beta) =
  -- For demonstration, we do a naive dimension/modulus "shrinking".
  -- This is not secure or correct for production usage; real relinearization
  -- uses a carefully crafted key-switch ciphertext.
  let halfAlpha = map (`shiftR` 1) alpha        -- alpha // 2
      alphaMod  = modQList params halfAlpha
      betaMod   = modQ params (beta `shiftR` 1) -- beta // 2
  in CT alphaMod betaMod

--------------------------------------------------------------------------------
-- Dimension-Modulus Reduction (Simplified Stub)
--------------------------------------------------------------------------------

-- | Demonstration of dimension-modulus reduction. This tries to mimic
--   the idea of "Squashing-free FHE" from the literature, but in practice,
--   it must be carefully engineered.
dimensionModulusReduce :: FHEParams -> Ciphertext -> Ciphertext
dimensionModulusReduce params (CT alpha beta) =
  let halfN = fheN params `div` 2
      alpha1 = take halfN alpha
      alpha2 = drop halfN alpha
      alphaCombined = zipWith (+) alpha1 alpha2
      alphaMod = modQList params alphaCombined
      betaMod  = modQ params (beta `shiftR` 1)
  in CT alphaMod betaMod

--------------------------------------------------------------------------------
-- Test / Usage Example
--------------------------------------------------------------------------------

-- | A small test routine. In real usage, you'd integrate with a larger library
--   or system, reading parameters from config, generating keys once, and then
--   encrypting/decrypting many times.
mainTest :: IO ()
mainTest = do
  putStrLn "=== LWE-based FHE Example (Production-Style Skeleton) ==="

  let params = defaultParams
  putStrLn $ "Using defaultParams: " ++ show params

  -- Key generation
  (sk, pk) <- keyGen params
  putStrLn "[+] Key generation done."

  -- Example messages
  let m1 = 1
  let m2 = 0

  -- Encrypt
  ct1 <- encrypt params pk m1
  ct2 <- encrypt params pk m2
  putStrLn $ "[+] Encrypted m1: " ++ show ct1
  putStrLn $ "[+] Encrypted m2: " ++ show ct2

  -- Homomorphic add
  let ctAdd = addCT params ct1 ct2
  let ctSub = subCT params ct1 ct2
  putStrLn $ "[+] ctAdd (m1 + m2): " ++ show ctAdd
  putStrLn $ "[+] ctSub (m1 - m2): " ++ show ctSub

  -- Homomorphic multiply
  let ctMul = multCT params ct1 ct2
  putStrLn $ "[+] ctMul (m1 * m2): " ++ show ctMul

  -- Relinearize and dimension-modulus reduce (demonstration)
  let ctMulRe = relinearize params pk ctMul
  let ctMulDM = dimensionModulusReduce params ctMulRe

  -- Decrypt
  let decM1    = decrypt params sk ct1
  let decM2    = decrypt params sk ct2
  let decAdd   = decrypt params sk ctAdd
  let decSub   = decrypt params sk ctSub
  let decMul   = decrypt params sk ctMul
  let decMulRe = decrypt params sk ctMulRe
  let decMulDM = decrypt params sk ctMulDM

  putStrLn $ "[-] Decrypted m1    = " ++ show decM1
  putStrLn $ "[-] Decrypted m2    = " ++ show decM2
  putStrLn $ "[-] Decrypted ctAdd = " ++ show decAdd
  putStrLn $ "[-] Decrypted ctSub = " ++ show decSub
  putStrLn $ "[-] Decrypted ctMul = " ++ show decMul
  putStrLn $ "[-] Decrypted ctMulRe = " ++ show decMulRe
  putStrLn $ "[-] Decrypted ctMulDM = " ++ show decMulDM

main :: IO ()
main = mainTest