# cuFHE
CUDA-accelerated Fully Homomorphic Encryption Library

v1.0_beta -- release on Mar/14/2018

## What is cuFHE?
The cuFHE library is an open-source library for Fully Homomorphic Encryption (FHE) on CUDA-enabled GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in CUDA C++. Compared to the [TFHE lib](https://github.com/tfhe/tfhe) which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHE library yields roughly 20 times of speedup on an NVIDIA Titan Xp graphics card. The cuFHE library benefits greatly from an improved CUDA implementation of the number-theoretic transform (NTT) proposed in the [cuHE library](https://github.com/vernamlab/cuHE) [Dai15] by Dai and Sunar.

| [TFHE lib](https://github.com/tfhe/tfhe) | cuFHE | Speedup |
|---|---|---|
| 13 ms | 0.6 ms | 22 times |

## Installation


## Change Log
- **version 1.0_beta** -- released on Mar/14/2018.
  - Supports single-bit unpacked encryption / decryption / gates.
  - C++ interface with CPU and GPU separate libraries.

## Reference

[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.

[Dai15]: Dai, W., & Sunar, B. (2015, September). cuHE: A homomorphic encryption accelerator library. In International Conference on Cryptography and Information Security in the Balkans (pp. 169-186). Springer, Cham.
