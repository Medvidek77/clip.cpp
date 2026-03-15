add_executable(quantize quantize.cpp)
target_include_directories(quantize PRIVATE ../ggml/include)
target_link_libraries(quantize PRIVATE clip ggml)
