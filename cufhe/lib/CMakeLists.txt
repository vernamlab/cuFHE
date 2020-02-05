add_library(cufhe_gpu
    bootstrap_gpu.cu
        cufhe_gates_gpu.cu
    cufhe_gpu.cu
    details/allocator_gpu.cu
    ntt_gpu/ntt_1024_twiddle.cu
    cufhe.cc
    cufhe_io.cc
    )
target_include_directories(cufhe_gpu
    PUBLIC ../ ../include
    )
target_compile_options(cufhe_gpu
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -arch=sm_70>
    )
