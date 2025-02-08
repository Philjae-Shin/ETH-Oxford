-- {-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- --type for encrypted numbers
-- newtype Encrypted a = Encrypted a
--   deriving (Show, Eq, Num)

-- -- if function for encrypted data.
-- -- It assumes that the encrypted condition is either 1 (true) or 0 (false).
-- ifEnc :: Num a => Encrypted a -> Encrypted a -> Encrypted a -> Encrypted a
-- ifEnc b x y = b * x + (1 - b) * y

-- -- Example usage:
-- -- Let’s simulate "if b then x else y" where:
-- -- b = Encrypted 1  (i.e., True)
-- -- x = Encrypted 100
-- -- y = Encrypted 50
-- example :: Encrypted Integer
-- example = ifEnc (Encrypted 1) (Encrypted 100) (Encrypted 50)
-- -- This should evaluate to Encrypted 100.


-------------------------------------------------------------------------------


-- class Homomorphic a where
--   hAdd :: a -> a -> a
--   hMul :: a -> a -> a
--   hSub :: a -> a -> a
--   hOne :: a  -- Multiplicative identity

-- -- A conditional selection function using homomorphic operations.
-- ifEncHom :: Homomorphic a => a -> a -> a -> a
-- ifEncHom b x y = hAdd (hMul b x) (hMul (hSub hOne b) y)

-------------------------------------------------------------------------------

{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- A simple wrapper to simulate encrypted data.
newtype Encrypted a = Encrypted a
    deriving (Show, Eq)

-- func compares two encrypted values.
ifEqEnc :: Eq a => Encrypted a -> Encrypted a -> String
ifEqEnc a b = if a == b then "True" else "False"

-- Example usage
main :: IO ()
main = do
  let a = Encrypted 42
      b = Encrypted 42
      c = Encrypted 13
  putStrLn $ "Comparing a and b: " ++ ifEqEnc a b 
  putStrLn $ "Comparing a and c: " ++ ifEqEnc a c  

