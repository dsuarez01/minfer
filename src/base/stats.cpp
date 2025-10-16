#include "minfer/base/stats.hpp"

#include <iomanip>
#include <iostream>

void GenStats::start_timer() {
    timer_start = std::chrono::high_resolution_clock::now();
}

float GenStats::get_elapsed_sec() const {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-timer_start);
    return duration.count() / 1000000.0f;
}

void GenStats::print_stats() const {
    std::cout << "Number of tokens generated: " << this->num_tokens_gen << " toks\n"
              << "Prefill time: " << std::fixed << std::setprecision(2) << this->prefill_time << " sec(s)\n"
              << "Time to first token: " << std::fixed << std::setprecision(2) << this->ttft << " sec(s)\n"
              << "Generation throughput: " << std::setprecision(2) << this->throughput << " tok/sec\n"
              << "Mem. Bandwidth: " << std::setprecision(2) << this->bandwidth << " GB/sec" << std::endl;
}