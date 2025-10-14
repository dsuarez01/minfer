add_executable(generate generate.cpp)
target_link_libraries(generate minfer pthread)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark minfer pthread)

add_custom_target(apps DEPENDS generate benchmark)