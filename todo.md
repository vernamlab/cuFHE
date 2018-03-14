# Future Updates

- [ ] Use configuration file to selectively compile CPU or GPU library.
  - [ ] Configuration file contains macros to be passed to source code.
  - [ ] A better CPU/GPU Context/Allocator design.
- [ ] Use CMake to detect GPU compute capability.
  - [ ] Ensure correct compilation.
  - [ ] Enable specific optimizations.
- [ ] Support multi-GPU computing.
  - [ ] A better GPU Context (device, stream, memory management).
  - [ ] Remove use of unified memory.
  - [ ] Use customized GPU memory allocator.
  - [ ] Depend on OpenMP to launch tasks.
  - [ ] Add alias, device transfer, assign methods to Ctxt class.
- [ ] Isolate CUDA NTT library.
- [ ] Use a fast CPU NTT library.
