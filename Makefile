CXX = g++
CXXFLAGS = -std=c++17 -I. -Iexternal -Isrc
BUILD_DIR = build
LOGS_DIR = logs
COMMON_SOURCES = src/common/config.cpp src/common/gguf.cpp src/common/model.cpp src/common/module.cpp

$(BUILD_DIR) $(LOGS_DIR):
	mkdir -p $@

# cross-platform compilation tests
compile-test: $(BUILD_DIR)
	@for src in $(COMMON_SOURCES); do \
		$(CXX) $(CXXFLAGS) -c $$src -o $(BUILD_DIR)/$$(basename $$src .cpp).o; \
	done

# unit tests for opn. correctness
TEST_SOURCES := $(wildcard src/tests/test_*.cpp)
UNIT_TESTS := $(patsubst src/tests/test_%.cpp,$(BUILD_DIR)/test_%,$(TEST_SOURCES))

$(BUILD_DIR)/test_%: src/tests/test_%.cpp $(COMMON_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

unit-tests: $(UNIT_TESTS)
	@if [ -n "$(UNIT_TESTS)" ]; then \
		passed=0; total=0; \
		for test in $(UNIT_TESTS); do \
			total=$$((total+1)); \
			if $$test; then \
				passed=$$((passed+1)); \
			fi; \
		done; \
		echo "$$passed of $$total tests passed"; \
		[ $$passed -eq $$total ] || exit 1; \
	fi

# dev tools (e.g. local testing w models)
loader_test: $(COMMON_SOURCES) src/tests/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/$@

run-loader: loader_test | $(LOGS_DIR)
	./$(BUILD_DIR)/loader_test $(FILE) > $(LOGS_DIR)/loader.txt 2>&1

clean:
	rm -rf $(BUILD_DIR) $(LOGS_DIR)

.PHONY: compile-test unit-tests clean