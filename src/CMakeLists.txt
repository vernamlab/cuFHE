add_library(cufhe_gpu
    bootstrap_gpu.cu
    cufhe_gates_gpu.cu
    cufhe_gpu.cu
    details/allocator_gpu.cu
    ntt_gpu/ntt_1024_twiddle.cu
    )
target_link_libraries(cufhe_gpu INTERFACE tfhe++)
target_include_directories(cufhe_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/cereal/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/spqlios
    )
