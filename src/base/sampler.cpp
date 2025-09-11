#include "minfer/base/sampler.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

Sampler::Sampler(const std::shared_ptr<Config> config) : 
    _rng(config->seed),
    _dist(0.0f,1.0f),
    _vocab_size(config->vocab_size),
    _temperature(config->temperature),
    _top_k(config->top_k),
    _top_p(config->top_p),
    _min_p(config->min_p),
    _penalty_pres(config->penalty_pres),
    _logits(config->vocab_size),
    _indices(config->vocab_size),
    _probs(config->vocab_size),
    _seen_token_ids()
{    
    assert(_top_k > 0 && _top_k <= _vocab_size && "top_k > vocab_size or top_k <= 0");
    assert(_top_p >= 0.0f && _top_p <= 1.0f && "top_p not in [0.0f,1.0f]");
    assert(_min_p >= 0.0f && _min_p <= 1.0f && "min_p not in [0.0f,1.0f]");
    assert(_penalty_pres >= 0.0f && _penalty_pres <= 2.0f && "penalty_pres not in [0.0f,2.0f]");
}

void Sampler::add_token(uint32_t token) {
    _seen_token_ids.insert(token);
}

uint32_t Sampler::sample(const float* logits) {  
    // greedy selection, idx with highest logit
    if (_temperature == 0.0f) {  
        return std::max_element(logits, logits+_vocab_size)-logits;  
    }
    
    // apply presence penalty, then sort logits and get top-k
    for (size_t i=0; i<_vocab_size; ++i) {
        _logits[i] = logits[i] - float(_seen_token_ids.count(i)) * _penalty_pres;
    }  
    
    std::iota(_indices.begin(), _indices.end(), 0);  
    
    // top-k unsorted
    std::nth_element(
        _indices.begin(), 
        _indices.begin() + _top_k, 
        _indices.end(),
        [&](size_t a, size_t b) { return _logits[a] > _logits[b]; }
    );

    // sort top-20
    std::sort(
        _indices.begin(), 
        _indices.begin() + _top_k,
        [&](size_t a, size_t b) { return _logits[a] > _logits[b]; }
    );
    
    size_t cutoff = _top_k;
    
    // softmax over top-k to get probs
    float cumsum = 0.0f;
    for (size_t i=0; i<cutoff; ++i) {  
        _probs[_indices[i]] = std::expf(_logits[_indices[i]] - _logits[_indices[0]]);
        cumsum += _probs[_indices[i]];
    }
    
    for (size_t i=0; i<cutoff; ++i) {  
        _probs[_indices[i]] /= cumsum;  
    }

    // top-p filter on probs
    cumsum = 0.0f;
    for (size_t i=0; i<cutoff; ++i) {
        cumsum += _probs[_indices[i]];
        if (cumsum >= _top_p) {  
            cutoff = i+1;  
            break;  
        }
    }  

    if (_min_p > 0.0f) {
        float threshold = _logits[_indices[0]] + std::logf(_min_p);
        for (size_t i=0; i<cutoff; ++i) {  
            if (_logits[_indices[i]] < threshold) {
                cutoff = i;
                break;
            }  
        }
    }

    assert(cutoff > 0 && "Filtered out all tokens, check sampling parameters");
   
    // final softmax on remaining logits, temperature applied
    cumsum = 0.0f;
    for (size_t i=0; i<cutoff; ++i) {
        _probs[_indices[i]] = std::expf((_logits[_indices[i]] - _logits[_indices[0]]) / _temperature);
        cumsum += _probs[_indices[i]];
    }
    for (size_t i=0; i<cutoff; ++i) {
        _probs[_indices[i]] /= cumsum;  
    }

    // sampling from discrete distribution
    float r = _dist(_rng);
    cumsum = 0.0f;
    for (size_t i=0; i<cutoff; ++i) {
        cumsum += _probs[_indices[i]];
        if (cumsum >= r) {
            return _indices[i];
        }
    }
    return _indices[cutoff-1]; // fallback
}