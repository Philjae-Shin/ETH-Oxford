module FHE where

import System.Random
import Data.Bits
import Data.List
import Data.Ord
import Control.Monad (replicateM)
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV
import Control.Monad.ST
import Data.STRef

--------------------------------------------------------------------------------
-- Config

data FHEParams = FHEParams
  { fheN        :: !Int
  , fhePrimes   :: ![Integer]
  , fheRoots    :: ![Integer]
  , fheSmallBD  :: !Int
  , fheKeyDist  :: !Int
  }

defaultFHEParams :: FHEParams
defaultFHEParams = FHEParams
  { fheN        = 1024
  , fhePrimes   = [ 65537, 12289, 114689 ]
  , fheRoots    = [ 3, 5, 7 ]
  , fheSmallBD  = 1
  , fheKeyDist  = 2
  }

--------------------------------------------------------------------------------
-- Basic Structures

type PolyMod = [Integer]
type PolyCRT = [PolyMod]

data ModulusNTT = ModulusNTT
  { theMod     :: !Integer
  , rootFwd    :: !Integer
  , rootInv    :: !Integer
  , fwdFactors :: ![Integer]
  , invFactors :: ![Integer]
  }

data FHEContext = FHEContext
  { ctxParams  :: !FHEParams
  , ctxNTTData :: ![ModulusNTT]
  }

type SecretKey = PolyCRT
type PublicKey = (PolyCRT, PolyCRT)
type RelinearKey = (PolyCRT, PolyCRT)
type GaloisKey   = (PolyCRT, PolyCRT)
type Ciphertext2 = (PolyCRT, PolyCRT)
type Ciphertext3 = (PolyCRT, PolyCRT, PolyCRT)

--------------------------------------------------------------------------------
-- Mod Utils

modQ :: Integer -> Integer -> Integer
modQ q x = let r = x `mod` q in if r < 0 then r + q else r

addMod :: Integer -> Integer -> Integer -> Integer
addMod q a b = modQ q (a + b)

subMod :: Integer -> Integer -> Integer -> Integer
subMod q a b = modQ q (a - b)

mulMod :: Integer -> Integer -> Integer -> Integer
mulMod q a b = modQ q (a * b)

negMod :: Integer -> Integer -> Integer
negMod q x = modQ q (q - x)

egcd :: Integer -> Integer -> (Integer, Integer, Integer)
egcd 0 b = (b, 0, 1)
egcd a b =
  let (q, r) = b `divMod` a
      (g, s, t) = egcd r a
  in (g, t - q*s, s)

modInverse :: Integer -> Integer -> Integer
modInverse a m =
  let (g, x, _) = egcd a m
  in if g /= 1 then error "no inverse" else modQ m x

--------------------------------------------------------------------------------
-- Bit Reverse

bitReverse :: Int -> Int -> Int
bitReverse bits x =
  let go acc i
        | i == bits = acc
        | otherwise = go ((acc `shiftL` 1) .|. ((x `shiftR` i) .&. 1)) (i+1)
  in go 0 0

--------------------------------------------------------------------------------
-- In-Place NTT (Cooley-Tukey)

inPlaceNTT :: Integer -> [Integer] -> [Integer] -> Int -> [Integer]
inPlaceNTT qVal twiddle input logN =
  runST $ do
    let size = length input
    arr <- V.thaw (V.fromList input)
    -- bit-reverse reorder
    forM_ [0..(size-1)] $ \i -> do
      let j = bitReverse logN i
      if i < j then do
        vi <- MV.read arr i
        vj <- MV.read arr j
        MV.write arr i vj
        MV.write arr j vi
      else return ()
    let stepFunc len = do
          let half = len
              step = len * 2
          if step > size
            then return ()
            else do
              let delta = size `div` step
              forM_ [0,step..(size-1)] $ \start -> do
                forM_ [0..(half-1)] $ \k -> do
                  t <- MV.read arr (start + k + half)
                  let root = twiddle !! (delta * k)
                  let tt = mulMod qVal t root
                  top <- MV.read arr (start + k)
                  let new1 = addMod qVal top tt
                  let new2 = subMod qVal top tt
                  MV.write arr (start + k) new1
                  MV.write arr (start + k + half) new2
              stepFunc step
    stepFunc 1
    V.toList <$> V.freeze arr

nttForward :: Integer -> [Integer] -> [Integer] -> [Integer]
nttForward qVal fwdF poly =
  let l = round (logBase 2 (fromIntegral (length poly)))
  in inPlaceNTT qVal fwdF poly l

nttInverse :: Integer -> [Integer] -> [Integer] -> [Integer]
nttInverse qVal invF poly =
  let l = round (logBase 2 (fromIntegral (length poly)))
      arr = inPlaceNTT qVal invF poly l
      invN = modInverse (fromIntegral (length poly)) qVal
  in map (\x -> mulMod qVal x invN) arr

--------------------------------------------------------------------------------
-- Build NTT data

buildNTTDataFor :: FHEParams -> Integer -> Integer -> IO ModulusNTT
buildNTTDataFor pms prime base = do
  let n2 = fheN pms
  let step = (prime - 1) `div` fromIntegral n2
  let fw i = modQ prime (base ^ (step * fromIntegral i))
  let iw i = modQ prime (modInverse base prime ^ (step * fromIntegral i))
  let fwd = [ fw i | i <- [0..(n2-1)] ]
  let inv = [ iw i | i <- [0..(n2-1)] ]
  return $ ModulusNTT
    { theMod     = prime
    , rootFwd    = base
    , rootInv    = modInverse base prime
    , fwdFactors = fwd
    , invFactors = inv
    }

initFHEContext :: FHEParams -> IO FHEContext
initFHEContext pms = do
  let ps  = fhePrimes pms
  let rs  = fheRoots pms
  d <- mapM (\(pp, rr) -> buildNTTDataFor pms pp rr) (zip ps rs)
  return $ FHEContext pms d

--------------------------------------------------------------------------------
-- PolyCRT Ops

normalizePoly :: Int -> [Integer] -> [Integer]
normalizePoly dim xs = take dim xs ++ replicate (max 0 (dim - length xs)) 0

polyAddMod :: Integer -> [Integer] -> [Integer] -> [Integer]
polyAddMod q p1 p2 = zipWith (addMod q) p1 p2

polySubMod :: Integer -> [Integer] -> [Integer] -> [Integer]
polySubMod q p1 p2 = zipWith (subMod q) p1 p2

polyMulNTT :: ModulusNTT -> [Integer] -> [Integer]
polyMulNTT info a =
  let qv = theMod info
      fwd = fwdFactors info
      inv = invFactors info
      aNTT = nttForward qv fwd a
  in aNTT

polyPointwiseMul :: Integer -> [Integer] -> [Integer] -> [Integer]
polyPointwiseMul q p1 p2 = zipWith (mulMod q) p1 p2

inverseNTT :: ModulusNTT -> [Integer] -> [Integer]
inverseNTT info arr =
  nttInverse (theMod info) (invFactors info) arr

polyCRTAdd :: FHEContext -> PolyCRT -> PolyCRT -> PolyCRT
polyCRTAdd ctx x y = zipWith (\info (a,b) ->
  zipWith (addMod (theMod info)) a b
  ) (ctxNTTData ctx) (zip x y)

polyCRTSub :: FHEContext -> PolyCRT -> PolyCRT -> PolyCRT
polyCRTSub ctx x y = zipWith (\info (a,b) ->
  zipWith (subMod (theMod info)) a b
  ) (ctxNTTData ctx) (zip x y)

polyCRTMul :: FHEContext -> PolyCRT -> PolyCRT -> PolyCRT
polyCRTMul ctx x y =
  let d = ctxNTTData ctx
  in zipWith (\info (px, py) ->
      let xf = nttForward (theMod info) (fwdFactors info) px
          yf = nttForward (theMod info) (fwdFactors info) py
          xy = zipWith (mulMod (theMod info)) xf yf
      in nttInverse (theMod info) (invFactors info) xy
     ) d (zip x y)

--------------------------------------------------------------------------------
-- PolyCRT Extended

buildPolyCRT :: FHEContext -> [Integer] -> PolyCRT
buildPolyCRT ctx msg =
  let pms = ctxParams ctx
      d   = ctxNTTData ctx
      dim = fheN pms
      nm  = normalizePoly dim msg
  in map (\info -> map (modQ (theMod info)) nm) d

extractPolyCRT :: FHEContext -> PolyCRT -> [Integer]
extractPolyCRT ctx xs =
  let first = head xs
      m1 = theMod (head (ctxNTTData ctx))
      center v = if v > (m1 `div` 2) then v - m1 else v
  in map center first

--------------------------------------------------------------------------------
-- Random Polynomials

genSmallPolyCRT :: FHEContext -> IO PolyCRT
genSmallPolyCRT ctx = do
  let d   = ctxNTTData ctx
  let pms = ctxParams ctx
  let bd  = fheSmallBD pms
  let genOne q = do
        r <- randomRIO (-bd, bd)
        return (modQ q (fromIntegral r))
  mapM (\info -> replicateM (fheN pms) (genOne (theMod info))) d

genBinaryPolyCRT :: FHEContext -> IO PolyCRT
genBinaryPolyCRT ctx = do
  let d   = ctxNTTData ctx
  mapM (\info -> replicateM (fheN (ctxParams ctx)) (do
            b <- randomRIO (0,1 :: Int)
            return (fromIntegral b `mod` theMod info)
        )) d

genUniformPolyCRT :: FHEContext -> IO PolyCRT
genUniformPolyCRT ctx = do
  let d = ctxNTTData ctx
  mapM (\info -> replicateM (fheN (ctxParams ctx)) (randomRIO (0, theMod info - 1))) d

--------------------------------------------------------------------------------
-- KeyGen

keyGen :: FHEContext -> IO (SecretKey, PublicKey, RelinearKey)
keyGen ctx = do
  s <- genSmallPolyCRT ctx
  a <- genUniformPolyCRT ctx
  e <- genSmallPolyCRT ctx
  let as = polyCRTMul ctx a s
  let asE = polyCRTAdd ctx as e
  let pk0 = zipWith (\inf pol -> map (negMod (theMod inf)) pol) (ctxNTTData ctx) asE
  let pk1 = a
  s2 <- polyCRTMul ctx s s
  a2 <- genUniformPolyCRT ctx
  e2 <- genSmallPolyCRT ctx
  let a2s2 = polyCRTMul ctx a2 s2
  let a2s2e2 = polyCRTAdd ctx a2s2 e2
  let rk0 = zipWith (\inf pol -> map (negMod (theMod inf)) pol) (ctxNTTData ctx) a2s2e2
  let rk1 = a2
  return (s, (pk0, pk1), (rk0, rk1))

--------------------------------------------------------------------------------
-- Encrypt / Decrypt

encrypt :: FHEContext -> PublicKey -> [Integer] -> IO Ciphertext2
encrypt ctx (pk0, pk1) msg = do
  let mCRT = buildPolyCRT ctx msg
  u  <- genSmallPolyCRT ctx
  e1 <- genSmallPolyCRT ctx
  e2 <- genSmallPolyCRT ctx
  let pk0u = polyCRTMul ctx pk0 u
  let c0   = polyCRTAdd ctx pk0u e1
  let c0'  = polyCRTAdd ctx c0 mCRT
  let pk1u = polyCRTMul ctx pk1 u
  let c1   = polyCRTAdd ctx pk1u e2
  return (c0', c1)

decrypt :: FHEContext -> SecretKey -> Ciphertext2 -> [Integer]
decrypt ctx s (c0, c1) =
  let sc1 = polyCRTMul ctx c1 s
      p   = polyCRTAdd ctx c0 sc1
  in extractPolyCRT ctx p

--------------------------------------------------------------------------------
-- Homomorphic Ops

cipherAdd :: FHEContext -> Ciphertext2 -> Ciphertext2 -> Ciphertext2
cipherAdd ctx (a0,a1) (b0,b1) = (polyCRTAdd ctx a0 b0, polyCRTAdd ctx a1 b1)

cipherMulNoRelin :: FHEContext -> Ciphertext2 -> Ciphertext2 -> Ciphertext3
cipherMulNoRelin ctx (c0a, c1a) (c0b, c1b) =
  let c0c0 = polyCRTMul ctx c0a c0b
      c0c1 = polyCRTMul ctx c0a c1b
      c1c0 = polyCRTMul ctx c1a c0b
      c1c1 = polyCRTMul ctx c1a c1b
      mid  = polyCRTAdd ctx c0c1 c1c0
  in (c0c0, mid, c1c1)

relinearize :: FHEContext -> RelinearKey -> Ciphertext3 -> Ciphertext2
relinearize ctx (rk0, rk1) (c0, c1, c2) =
  let t0 = polyCRTMul ctx c2 rk0
      t1 = polyCRTMul ctx c2 rk1
      nc0 = polyCRTAdd ctx c0 t0
      nc1 = polyCRTAdd ctx c1 t1
  in (nc0, nc1)

--------------------------------------------------------------------------------
-- Galois (Rotation) Key (Optional Example)

genGaloisKey :: FHEContext -> SecretKey -> Int -> IO GaloisKey
genGaloisKey ctx s rot = do
  a  <- genUniformPolyCRT ctx
  e  <- genSmallPolyCRT ctx
  let sRot = rotatePolyCRT ctx s rot
  let aS   = polyCRTMul ctx a sRot
  let aSE  = polyCRTAdd ctx aS e
  let g0 = zipWith (\inf pol -> map (negMod (theMod inf)) pol) (ctxNTTData ctx) aSE
  let g1 = a
  return (g0, g1)

rotatePolyCRT :: FHEContext -> PolyCRT -> Int -> PolyCRT
rotatePolyCRT ctx p shiftBy =
  let n = fheN (ctxParams ctx)
  in map (\pm -> rotatePoly pm shiftBy) p

rotatePoly :: [Integer] -> Int -> [Integer]
rotatePoly arr k =
  let n = length arr
      r = k `mod` n
  in drop r arr ++ take r arr

applyGaloisKey :: FHEContext -> GaloisKey -> Ciphertext2 -> Int -> Ciphertext2
applyGaloisKey ctx (g0, g1) (c0, c1) shiftBy =
  let rc0 = map (\pm -> rotatePoly pm shiftBy) c0
      rc1 = map (\pm -> rotatePoly pm shiftBy) c1
      sc1 = polyCRTMul ctx rc1 g1
      c0a = polyCRTAdd ctx rc0 sc1
      c0b = zipWith (\inf pol -> map (negMod (theMod inf)) pol) (ctxNTTData ctx) c0a
      c0g = polyCRTMul ctx rc1 g0
      new0 = polyCRTAdd ctx rc0 c0g
  in (new0, sc1)

--------------------------------------------------------------------------------
-- Demo (Optional)

demoMain :: IO ()
demoMain = do
  ctx <- initFHEContext defaultFHEParams
  (s, pk, rk) <- keyGen ctx
  let msgA = [1, 2, 3, 0, 4, 5]
  let msgB = [2, 1, 1, 1, 0]
  ctA <- encrypt ctx pk msgA
  ctB <- encrypt ctx pk msgB
  let decA = decrypt ctx s ctA
  let decB = decrypt ctx s ctB
  let addCT = cipherAdd ctx ctA ctB
  let decAdd = decrypt ctx s addCT
  let mulCT3 = cipherMulNoRelin ctx ctA ctB
  let mulCT2 = relinearize ctx rk mulCT3
  let decMul = decrypt ctx s mulCT2
  putStrLn $ "A  = " ++ show decA
  putStrLn $ "B  = " ++ show decB
  putStrLn $ "A+B= " ++ show decAdd
  putStrLn $ "A*B= " ++ show decMul

--------------------------------------------------------------------------------
-- Helpers

forM_ :: (Monad m, Integral i) => [i] -> (i -> m ()) -> m ()
forM_ = flip mapM_

replaceAtVec :: MV.MVector s a -> Int -> a -> ST s ()
replaceAtVec v i x = MV.write v i x