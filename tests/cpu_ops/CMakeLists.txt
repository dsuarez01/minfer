add_executable(test_cpu_ops 
   test_cpu_ops.cpp
   ../src/base/test_base.cpp
   ../src/cpu_ops/test_matmul.cpp
   ../src/cpu_ops/test_silu.cpp
   ../src/cpu_ops/test_softmax.cpp
   ../src/cpu_ops/test_rmsnorm.cpp
   ../src/cpu_ops/test_rope.cpp
   ../src/cpu_ops/test_attn.cpp
   ../src/cpu_ops/test_swiglu.cpp
   ../src/cpu_ops/test_route.cpp
)

target_include_directories(test_cpu_ops PRIVATE ${CMAKE_SOURCE_DIR}/tests/include)
target_link_libraries(test_cpu_ops minfer pthread)
apply_arch_flags(test_cpu_ops)