# Future Updates

- [ ] Use configuration file to selectively compile CPU or GPU library.
  - [ ] Configuration file contains macros to be passed to source code.
  - [ ] A better CPU/GPU Context/Allocator design.
- [x] Use CMake to detect GPU compute capability.
  - [x] Ensure correct compilation.
  - [x] Enable specific optimizations.
- [x] Support multi-GPU computing.
  - [x] A better GPU Context (device, stream, memory management).
  - [x] Remove use of unified memory.
  - [ ] Use customized GPU memory allocator.
  - [ ] Add alias, device transfer, assign methods to Ctxt class.
- [ ] Isolate CUDA NTT library.
- [ ] Use a fast CPU NTT library.
