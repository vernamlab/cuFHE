#include <include/cufhe_gpu.cuh>

const size_t N = 20, M = 1000;

template <class Launcher, class Verifier>
void runAndVerify(const char* name, Launcher&& launcher, Verifier&& verifier)
{
    cufhe::Stream st[M];
    for (size_t i = 0; i < M; i++) st[i].Create();

    int workingIndex[M] = {};
    for (size_t i = 0; i < M; i++) launcher(0, i, st[i]);
    while (true) {
        bool cont = false;
        for (size_t i = 0; i < M; i++) {
            if (workingIndex[i] == N) continue;
            cont = true;

            if (cufhe::StreamQuery(st[i])) {
                int j = ++workingIndex[i];
                if (j == N) continue;
                launcher(j, i, st[i]);
            }
        }
        if (!cont) break;
    }
    size_t errcount = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            if (!verifier(i, j)) errcount++;
    if (errcount != 0)
        std::cerr << "TEST FAILED! " << name << " " << errcount << "/(" << N
                  << " * " << M << ")\n";
    assert(errcount == 0);

    for (size_t i = 0; i < M; i++) st[i].Destroy();
}

void testMux(cufhe::PriKey& sk, cufhe::PubKey& gk)
{
    cufhe::Ctxt ca, cb, cc, cres[N][M];
    cufhe::Ptxt pa, pb, pc;
    pa = true;
    pb = false;
    pc = true;
    uint32_t expected = pa.get() ? pb.get() : pc.get();
    Encrypt(ca, pa, sk);
    Encrypt(cb, pb, sk);
    Encrypt(cc, pc, sk);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            if (expected)
                cufhe::ConstantZero(cres[i][j]);
            else
                cufhe::ConstantOne(cres[i][j]);

    runAndVerify(
        "mux",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Mux(cres[i][j], ca, cb, cc, st);
        },
        [&](size_t i, size_t j) {
            cufhe::Ptxt decres;
            Decrypt(decres, cres[i][j], sk);
            return expected == decres.get();
        });
}

void testNand(cufhe::PriKey& sk, cufhe::PubKey& gk)
{
    cufhe::Ctxt ca, cb, cres[N][M];
    cufhe::Ptxt pa, pb;
    pa = true;
    pb = false;
    uint32_t expected = !(pa.get() && pb.get());
    Encrypt(ca, pa, sk);
    Encrypt(cb, pb, sk);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            if (expected)
                cufhe::ConstantZero(cres[i][j]);
            else
                cufhe::ConstantOne(cres[i][j]);

    runAndVerify(
        "nand",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Nand(cres[i][j], ca, cb, st);
        },
        [&](size_t i, size_t j) {
            cufhe::Ptxt decres;
            Decrypt(decres, cres[i][j], sk);
            return expected == decres.get();
        });
}

int main()
{
    cufhe::PriKey sk;
    cufhe::PubKey gk;
    cufhe::KeyGen(gk, sk);
    cufhe::Initialize(gk);

    testNand(sk, gk);
    testMux(sk, gk);

    cufhe::CleanUp();
}
