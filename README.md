# cuFHE
CUDA-accelerated Torus Fully Homomorphic Encryption Library. This fork is maintained as a sub project of Virtual Secure Platform.

v1.0_beta -- release on Mar/14/2018 original [cuFHE](https://github.com/vernamlab/cuFHE)

v1 -- kvsp v29 compatible version.

v2 -- refctored & reduced shared memory usage (depends on [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp) for parameter set select, runnable on old GPUs like GTX 1060Ti but slow)

v3 -- l parallel NTT & fixed shared memory read before write bug by adding __syncthreads(). v3 API is the same as v2.

## What is cuFHE?
The cuFHE library is an open-source library for Fully Homomorphic Encryption (FHE) on CUDA-enabled GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in CUDA C++. Compared to the [TFHE lib](https://github.com/tfhe/tfhe) which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHE library yields almost same performance per SM. Since GPU has a lot of SMs (128 in A100), cuFHE gives better performace if there are enough number of parallely evaluable tasks. The cuFHE library benefits greatly from an improved CUDA implementation of the number-theoretic transform (NTT) proposed in the [cuHE library](https://github.com/vernamlab/cuHE) [Dai15] by Dai and Sunar.

| [TFHE lib](https://github.com/tfhe/tfhe) | cuFHE |
|---|---|
| 10 ms | 13 ms |

### System Requirements
**The library has been tested on Ubuntu Desktop 20.04 & NVIDIA A100 only.**
GPU support requires NVIDIA Driver and NVIDIA CUDA Toolkit.

### Installation (Linux)
Do the standard CMake compulation process.
```
cd cufhe
mkdir build
cmake .. -DENABLE_TEST=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

### User Manual
Use files in `cufhe/test/` as examples. To summarize, follow the following function calling procedures.
```c++
SetGPUNum(2); //Set number of gpu to use. Now use 2 GPU. If you do not specify GPU number, use only 1 GPU.
SetSeed(); // init random generator seed
PriKey pri_key;
PubKey pub_key;
KeyGen(pub_key, pri_key); // key generation
// alternatively, write / read key files
Ptxt pt[2];
pt[0] = 0; // 0 or 1, single bit
pt[1] = 1;

Ctxt ct[4];
Encrypt(ct[0], pt[0], pri_key);
Encrypt(ct[1], pt[1], pri_key);
Encrypt(ct[2], pt[0], pri_key);
Encrypt(ct[3], pt[1], pri_key);

Initialize(pub_key); // for GPU library

Stream stream_gpu_0(0); //Create Stream runs on GPU0
stream_gpu_0.Create();

Stream stream_gpu_1(1); //Create Stream runs on GPU1
stream_gpu_1.Create();

Nand(ct[0], ct[0], ct[1], stream_gpu_0); //Run Nand on GPU0
Nand(ct[2], ct[2], ct[3], stream_gpu_1); //Run Nand on GPU1

Synchronize(); //Synchronize All GPU

Decrypt(pt[0], ct[0], pri_key);

stream_gpu_0.Destroy(); //Destroy Stream
stream_gpu_1.Destory(); 

CleanUp(); // for GPU library
```

Currently implemented gates are `And, AndNY, AndYN, Or, OrNY, OrYN Nand, Nor, Xor, Xnor, Not, Mux, Copy`.

## Acknowledgement
- We appreciate any bug reports or compiling issues.
- Dai and Sunar’s work was in part provided by the US National Science Foundation CNS Award #1319130 and #1561536.
- From the original authors: We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research.
- From VSP team :  We gratefully acknowledge the support of Sakura Internet Inc. for lending us their cloud computing service equipped with one V100 free of charge for most of the development period of v1.

## Reference
[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.

[Dai15]: Dai, W., & Sunar, B. (2015, September). cuHE: A homomorphic encryption accelerator library. In International Conference on Cryptography and Information Security in the Balkans (pp. 169-186). Springer, Cham.
