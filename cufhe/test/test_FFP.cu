#include <cassert>
#include <iostream>
#include <random>
// #include <boost/multiprecision/cpp_int.hpp>
#include <gmp.h>
#include <gmpxx.h>

#include <include/ntt_gpu/ntt_ffp.cuh>

using namespace std;
using namespace cufhe;
// namespace mp = boost::multiprecision;
__global__ void __Lsh__(uint64_t* d_res, uint64_t* d_a, uint32_t* d_l)
{
    FFP temp = FFP(*d_a);
    temp.Lsh96(*d_l);
    *d_res = temp.val();
}

int main()
{
    constexpr uint64_t P = (((1ULL << 32) - 1) << 32) + 1;

    constexpr int numTest = 10000;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint64_t> dist(0, P);
    uint64_t* d_a;
    uint64_t* d_res;
    uint32_t* d_l;
    cudaMalloc((void**)&d_a, sizeof(uint64_t));
    cudaMalloc((void**)&d_res, sizeof(uint64_t));
    cudaMalloc((void**)&d_l, sizeof(uint32_t));

    // Lsh Test
    int upper = 3;
    for (uint32_t l = 32 * (upper - 1); l < 32 * upper; l++) {
        // cout<<l<<endl;
        for (int i = 0; i < numTest; i++) {
            uint64_t temp = dist(engine);
            uint64_t h_res;
            cudaMemcpy(d_a, &temp, sizeof(temp), cudaMemcpyHostToDevice);
            cudaMemcpy(d_l, &l, sizeof(l), cudaMemcpyHostToDevice);
            __Lsh__<<<1, 1>>>(d_res, d_a, d_l);
            cudaMemcpy(&h_res, d_res, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            // mp::cpp_int a = temp;
            mpz_class a = temp;
            // __uint128_t a = temp;
            // cout<<static_cast<uint64_t>(a)<<":";
            // if((A<<l).value!=(static_cast<uint64_t>((a<<l)%P))){
            if (h_res != (a << l) % P) {
                mpz_class res = (a << l) % P;
                // __uint128_t res = (a<<l)%P;
                cout << h_res << ":" << res.get_str() << endl;
                // cout<<h_res - static_cast<uint64_t>((a<<l)%P)<<endl;
                // cout<<h_res<<":"<<static_cast<uint64_t>((a<<l)%P)<<endl;
            }
            // assert(h_res==(static_cast<uint64_t>((a<<l)%P)));
            assert(h_res == (a << l) % P);
        }
        // __uint128_t temp = ((1UL<<32)-1)<<32;
        // INTorus A(temp);
        // mp::cpp_int a = temp;
        // if((A<<l).value!=(static_cast<uint64_t>((a<<l)%P))){
        //     cout<<(A<<l).value<<":"<<static_cast<uint64_t>((a<<l)%P)<<endl;
        //     cout<<"Here"<<endl;
        //  }
        // assert((A<<l).value==(static_cast<uint64_t>((a<<l)%P)));
    }
    cout << "Lsh" << upper * 32 << " PASS" << endl;
}