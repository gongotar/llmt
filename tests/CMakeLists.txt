add_executable(llmt_tests
    test_main.cpp
    support/golden_io.cpp
    unit/test_smoke.cpp
    unit/test_core_host.cpp
    unit/test_golden_host.cpp
)
if(LLMT_HAS_CUDA)
    target_sources(llmt_tests PRIVATE
        unit/test_gpu_smoke.cu
        unit/test_core_gpu.cu
        unit/test_golden_gpu.cu
    )
endif()
target_link_libraries(llmt_tests PRIVATE llmt)
target_include_directories(llmt_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/support
)
# Golden files are read from the source tree (committed, tiny shapes).
target_compile_definitions(llmt_tests PRIVATE
    LLMT_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden"
)
add_test(NAME llmt_tests COMMAND llmt_tests)
