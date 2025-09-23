add_executable(test_tokenizer
   test_tokenizer.cpp
   ../src/base/test_base.cpp
   ../src/qwen3/tokenizer/tokenizer.cpp
)

add_executable(test_metal
   test_metal.cpp
)

# tokenizer
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/tests/include)
target_link_libraries(test_tokenizer minfer pthread)
apply_arch_flags(test_tokenizer)


# device transfer
target_include_directories(test_metal PRIVATE ${CMAKE_SOURCE_DIR}/tests/include)
target_link_libraries(test_metal minfer pthread)
apply_arch_flags(test_metal)