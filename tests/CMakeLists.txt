cuda_add_executable(test_d2d src/test_d2d.cu)
target_link_libraries(test_d2d jusha)

cuda_add_executable(test_scan src/test_scan.cu)
target_link_libraries(test_scan jusha)


