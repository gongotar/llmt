# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Masoud Jami
if(LLMT_HAS_CUDA)
    add_executable(llmt_bench_matmul matmul_bench.cu)
    target_link_libraries(llmt_bench_matmul PRIVATE llmt)
    add_executable(llmt_bench_bandwidth bandwidth_bench.cu)
    target_link_libraries(llmt_bench_bandwidth PRIVATE llmt)
endif()
