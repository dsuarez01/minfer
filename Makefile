CXX = g++
CXXFLAGS = -std=c++17 -I. -Iexternal -Isrc
BUILD_DIR = build
LOGS_DIR = logs
COMMON_SOURCES = src/common/config.cpp src/common/gguf.cpp

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LOGS_DIR):
	mkdir -p $(LOGS_DIR)

all:
	@echo "Available test programs:"
	@echo "  make loader_test"
	@echo "  make loader_test_debug"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-loader FILE=path/to/name-of-model.gguf"
	@echo "  make run-loader-debug FILE=path/to/name-of-model.gguf"

# test programs
loader_test: $(COMMON_SOURCES) src/tests/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/$@

# debug versions
loader_test_debug: CXXFLAGS += -g -O0 -DDEBUG
loader_test_debug: $(COMMON_SOURCES) src/tests/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/$@

# run targets
run-loader: loader_test | $(LOGS_DIR)
	@echo "Running loader test with $(FILE)..."
	./$(BUILD_DIR)/loader_test $(FILE) > $(LOGS_DIR)/loader.txt 2>&1
	@echo "Output saved to $(LOGS_DIR)/loader.txt"

run-loader-debug: loader_test_debug | $(LOGS_DIR)
	@echo "Running loader test (debug) with $(FILE)..."
	./$(BUILD_DIR)/loader_test_debug $(FILE) > $(LOGS_DIR)/loader_debug.txt 2>&1
	@echo "Debug output saved to $(LOGS_DIR)/loader_debug.txt"

clean:
	rm -rf $(BUILD_DIR) $(LOGS_DIR)

.PHONY: all clean run-loader run-loader-debug