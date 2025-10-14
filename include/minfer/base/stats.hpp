#include <chrono>

struct GenStats {
    int num_tokens_gen = 0; // number of tokens generated
    float ttft = 0.0f;         // time to first token, sec
    float throughput = 0.0f;   // tok/sec
    float prefill_time = 0.0f; // in secs
    float bandwidth = 0.0f;    // in GB/sec

    std::chrono::high_resolution_clock::time_point timer_start;

    void start_timer();
    float get_elapsed_sec() const;
    void print_stats() const;
};