add_subdirectory(cpu_ops)
add_subdirectory(qwen3)

add_custom_target(tests DEPENDS test_cpu_ops test_tokenizer)