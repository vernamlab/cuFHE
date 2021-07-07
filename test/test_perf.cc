// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
#include <include/plain.h>
#include <test/test_util.h>
using namespace cufhe;

#include <iostream>
#include <random>
#include <vector>
using namespace std;

int main()
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const uint32_t kNumSMs = 800;
    const uint32_t kNumTests = 1024;

    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);

    cout << "n:" << sk->params.lvl0.n << endl;

    vector<uint8_t> pt(kNumTests);
    vector<Ctxt> ct(kNumTests);
    vector<cuFHETRLWElvl1> trlweLv1(kNumTests);
    vector<Ctxt> ctTemp(kNumTests);
    vector<cuFHETRLWElvl1> trlweLv1Temp(kNumTests);

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    for (int i = 0; i < kNumTests; i++) {
        auto tmp = binary(engine) > 0;
        ct[i].tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(
            tmp ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
            TFHEpp::lvl0param::α, sk->key.lvl0);
    }
    Synchronize();
    bool correct;

    cout << "Number of tests:\t" << kNumTests << endl;

    cout << "------ Initilizating Data on GPU(s) ------" << endl;
    Initialize(*gk);  // essential for GPU computing

    Stream* st = new Stream[kNumSMs];
    for (int i = 0; i < kNumSMs; i++) st[i].Create();

    for (int i = 0; i < kNumTests; i++) {
        GateBootstrappingTLWE2TRLWElvl01NTT(trlweLv1[i], ct[i], st[kNumTests % kNumSMs]);
    }
    Synchronize();

    cout << "Done." << endl;
    cout << "------ Starting Benchmark ------" << endl;
    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < kNumTests; i++) {
        SampleExtractAndKeySwitch(ctTemp[i], trlweLv1[i], st[kNumTests % kNumSMs]);
        GateBootstrappingTLWE2TRLWElvl01NTT(trlweLv1Temp[i], ctTemp[i], st[kNumTests % kNumSMs]);
    }
    Synchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cout << "Total:" << et << "ms" << endl;
    cout << et / kNumTests << " ms / operation" << endl;
    cout << et / kNumSMs << " ms / stream" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < kNumSMs; i++) st[i].Destroy();
    delete[] st;

    cout << "------ Cleaning Data on GPU(s) ------" << endl;
    CleanUp();  // essential to clean and deallocate data
    return 0;
}
