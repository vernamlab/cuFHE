
#include <include/cufhe.h>
#include <include/cufhe_gpu.cuh>
using namespace cufhe;
#include <iostream>
using namespace std;

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

int main() {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t kNumSMs = prop.multiProcessorCount;
  uint32_t kNumTests = kNumSMs * 64;

  SetSeed();
  bool correct;
  PriKey pri_key;
  PubKey pub_key;
  Ptxt* pt = new Ptxt[2 * kNumTests];
  Ctxt* ct = new Ctxt[2 * kNumTests];
  pri_key.New<AllocatorCPU>();
  pub_key.New<AllocatorCPU>();
  for (int i = 0; i < 2 * kNumTests; i ++)
    ct[i].New<AllocatorBoth>();
  cudaDeviceSynchronize();

  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key, pri_key);

  cout<< "------ Test Encryption/Decryption ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[i].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
    Decrypt(pt[kNumTests + i], ct[i], pri_key);
    if (pt[kNumTests + i].message_ != pt[i].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  cout<< "------ Initilizating Data on GPU(s) ------" <<endl;
  Initialize(pub_key);

  cout<< "------ Test NAND Gate ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  cudaStream_t* st = new cudaStream_t[kNumSMs];
  for (int i = 0; i < kNumSMs; i ++)
    cudaStreamCreateWithFlags(&st[i], cudaStreamDefault);//NonBlocking

  correct = true;
  for (int i = 0; i < 2 * kNumTests; i ++) {
    pt[i].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
  }
  cudaDeviceSynchronize();

  float et;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < kNumTests; i ++)
    Nand(ct[i], ct[i], ct[i + kNumTests], st[i % kNumSMs]);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&et, start, stop);
  cout<< et / kNumTests << " ms / gate" <<endl;

  for (int i = 0; i < kNumTests; i ++) {
    NandCheck(pt[i + kNumTests], pt[i], pt[i + kNumTests]);
    Decrypt(pt[i], ct[i], pri_key);
    if (pt[i + kNumTests].message_ != pt[i].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  cout<< "------ Cleaning Data on GPU(s) ------" <<endl;
  pri_key.Delete();
  pub_key.Delete();
  for (int i = 0; i < 2 * kNumTests; i ++)
    ct[i].Delete();
  delete [] ct;
  delete [] pt;
  CleanUp();
  for (int i = 0; i < kNumSMs; i ++)
    cudaStreamDestroy(st[i]);
  delete [] st;
  return 0;
}
