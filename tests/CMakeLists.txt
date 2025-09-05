add_executable(test_ops 
   test_ops.cpp
   src/base/test_base.cpp
   src/ops/test_matmul.cpp
   src/ops/test_silu.cpp
   src/ops/test_softmax.cpp
   src/ops/test_rmsnorm.cpp
   src/ops/test_rope.cpp
   src/ops/test_attn.cpp
   src/ops/test_swiglu.cpp
   src/ops/test_route.cpp
)

target_include_directories(test_ops PRIVATE include)
target_link_libraries(test_ops minfer pthread)

add_executable(test_tokenizer
   test_tokenizer.cpp
   src/base/test_base.cpp
   src/tokenizer/test_tokenizer.cpp
)

target_include_directories(test_tokenizer PRIVATE include)
target_link_libraries(test_tokenizer minfer pthread)
apply_arch_flags(test_ops)
apply_arch_flags(test_tokenizer)