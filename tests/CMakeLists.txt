add_executable(llmt_tests
    test_main.cpp
    unit/test_smoke.cpp
    unit/test_core_host.cpp
)
if(LLMT_HAS_CUDA)
    target_sources(llmt_tests PRIVATE
        unit/test_gpu_smoke.cu
        unit/test_core_gpu.cu
    )
endif()
target_link_libraries(llmt_tests PRIVATE llmt)
target_include_directories(llmt_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
add_test(NAME llmt_tests COMMAND llmt_tests)
