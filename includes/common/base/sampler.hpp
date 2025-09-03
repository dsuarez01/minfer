#pragma once

#include "common/config/config.hpp"
#include <cstdint>
#include <random>
#include <vector>
#include <unordered_set>

class Sampler {
public:
    Sampler(const std::shared_ptr<Config> config);
    void add_token(uint32_t token);
    uint32_t sample(const float* logits);
    
private:
    size_t _vocab_size, _top_k;
    float _temperature, _penalty_pres, _top_p, _min_p;
    std::vector<float> _logits, _probs;
    std::vector<uint32_t> _indices;
    std::unordered_set<uint32_t> _seen_token_ids;
    std::mt19937_64 _rng;
    std::uniform_real_distribution<float> _dist;
};