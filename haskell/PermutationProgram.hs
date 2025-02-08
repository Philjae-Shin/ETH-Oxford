module Main where

import System.Environment (getArgs)
import Data.List (permutations)
import Data.Aeson (encode, ToJSON, toJSON, object, (.=))
import qualified Data.ByteString.Lazy.Char8 as BL

-- | A simple data type to serialize results as JSON.
data PermResult = PermResult
  { perm  :: String  -- The permutation string
  , value :: Int     -- For example, some computed value based on the permutation
  }

-- JSON instance (using aeson)
instance ToJSON PermResult where
  toJSON (PermResult p v) =
    object [ "permutation" .= p
           , "value"       .= v ]

main :: IO ()
main = do
    args <- getArgs
    -- Example args: ["3","2"] => A=3, B=2
    case args of
      [aStr, bStr] -> do
          let aCount = read aStr :: Int
              bCount = read bStr :: Int

          -- Suppose we need O(n^3) computations; here, as an example,
          -- we generate all permutations of 'A' repeated aCount times and
          -- 'B' repeated bCount times.
          -- Note that if there are duplicates, permutations will also contain duplicates.
          let inputChars = replicate aCount 'A' ++ replicate bCount 'B'
              allPerms   = permutations inputChars

          -- For demonstration, we assign a dummy calculation to each permutation,
          -- storing both the permutation string and the computed value in 'PermResult'.
          let results = map (\p -> PermResult p (calcValue p)) allPerms

          -- Output the results as JSON to stdout.
          BL.putStrLn (encode results)

      _ -> putStrLn "Usage: PermutationProgram <numA> <numB>"

-- | A simple example function that calculates an integer based on a permutation string.
calcValue :: String -> Int
calcValue permStr = length permStr * 7  -- Just a trivial logic for the hackathon demo
