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
    cudaDeviceSynchronize();
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

void testMux(TFHEpp::SecretKey& sk)
{
    cufhe::Ctxt ca, cb, cc, cres[N][M];
    bool pa, pb, pc;
    pa = true;
    pb = false;
    pc = true;
    bool expected = pa ? pb : pc;
    ca.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(pa?TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ, TFHEpp::lvl0param::α, sk.key.lvl0);
    cb.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(pb?TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ, TFHEpp::lvl0param::α, sk.key.lvl0);
    cc.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(pc?TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ, TFHEpp::lvl0param::α, sk.key.lvl0);

    runAndVerify(
        "mux",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Mux(cres[i][j], ca, cb, cc, st);
        },
        [&](size_t i, size_t j) {
            bool decres;
            decres=TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(cres[i][j].tlwehost, sk.key.lvl0);
            return expected == decres;
        });
}

void testNand(TFHEpp::SecretKey& sk)
{
    cufhe::Ctxt ca, cb, cres[N][M];
    bool pa, pb;
    pa = true;
    pb = false;
    bool expected = !(pa && pb);
    ca.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(pa?TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ, TFHEpp::lvl0param::α, sk.key.lvl0);
    cb.tlwehost = TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(pb?TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ, TFHEpp::lvl0param::α, sk.key.lvl0);

    runAndVerify(
        "nand",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Nand(cres[i][j], ca, cb, st);
        },
        [&](size_t i, size_t j) {
            bool decres;
            decres=TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(cres[i][j].tlwehost, sk.key.lvl0);
            return expected == decres;
        });
}

int main()
{
    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::GateKeywoFFT* gk = new TFHEpp::GateKeywoFFT(*sk);
    cufhe::Initialize(*gk);

    testNand(*sk);
    testMux(*sk);

    cufhe::CleanUp();
}
