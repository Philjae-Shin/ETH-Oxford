module Main where

-- stack ghc -- PermutationProgram8.hs -o PermutationProgram8
-- haskell % ghc PermutationProgram8.hs -o PermutationProgram8

-- ./PermutationProgram8 A A B A B B A B

import System.Environment (getArgs)
import Data.List (permutations, nub)

main :: IO ()
main = do
    args <- getArgs

    -- 1) Check if exactly 8 arguments are provided
    if length args /= 8
        then putStrLn "Please provide exactly 8 arguments (each 'A' or 'B'). Example: A A B A B B A B"
        else do
            -- 2) Check if all arguments are either 'A' or 'B'
            let valid = all (`elem` ["A","B"]) args
            if not valid
                then putStrLn "Invalid argument found. All must be either 'A' or 'B'."
                else do
                    -- 3) Generate permutations of the given arguments
                    --    (remove duplicates using nub because some permutations may be identical)
                    let allPerms = nub (permutations args)

                    putStrLn "All unique permutations of the 8 inputs:"
                    mapM_ print allPerms
                    putStrLn $ "\nTotal unique permutations: " ++ show (length allPerms)
