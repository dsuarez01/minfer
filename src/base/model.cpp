#include "minfer/base/model.hpp"

#include <iostream>

BaseModel::BaseModel(const std::string& model_file, const RunParams& run_params) {
    model_data = std::make_unique<ModelData>();
    if (model_data->from_file(model_file) != 0) {
        throw std::runtime_error("Failed to load model file: " + model_file);
    }
    config = std::make_shared<Config>(*model_data, run_params);
    sampler = std::make_unique<Sampler>(config);
    run_state = std::make_shared<RunState>(config);
    stats = std::make_unique<GenStats>();
    // derived responsible for setting rest
}

void BaseModel::append_layer(std::shared_ptr<BaseLayer> layer) {
    _read_bytes += layer->get_read_bytes();
    // std::cout << "qdtype: " << dtype_to_str(layer->get_qdtype()) << std::endl;
    _layers.push_back(layer);
}

void BaseModel::set_device(Device target_device) {
    if (this->_device == target_device) return;
    for (auto& layer : _layers) {
        layer->set_device(target_device);
    }
    run_state->set_device(target_device); // move buffers to device
    this->_device = target_device;
}

Device BaseModel::get_device() const {
    return this->_device;
}

size_t BaseModel::get_read_bytes() const {
    return this->_read_bytes;
}

// TO-DO: multi-turn conversation?
void BaseModel::generate(std::string& input_text) {
    
    // create message for single user input
    std::vector<BaseTokenizer::Message> messages;
    messages.emplace_back("user", input_text);
    
    // apply chat template, tokenize
    std::string formatted = tokenizer->apply_chat_template(messages, {}, true);
    std::cout << "Formatted message:\n" << formatted << std::endl;
    std::vector<uint32_t> tokens = tokenizer->encode(formatted);
    std::cout << "Number of tokens: " << tokens.size() << std::endl;
    
    // TO-DO: remove once/if ring buffer implemented
    assert(tokens.size() <= config->user_max_seq_len && "# of encoded tokens should be at most passed in max seq len");

    stats->start_timer(); // PREFILL START
    
    size_t total_passes = 0;
    size_t kv_cache_bytes_read = 0;
    
    // PREFILL
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "Prefill progress: " << (i+1) << "/" << (tokens.size()) << "\r" << std::flush;
        run_state->cur_pos = i;
        run_state->token_id = tokens[i];
        run_state->compute_logits = (i == tokens.size()-1);
        forward(run_state);
        
        total_passes++;
        kv_cache_bytes_read += run_state->kv_bytes_per_pos * (run_state->cur_pos+1); // read from pos [0, 1, ..., i]
    }

    std::cout << std::endl;
    float prefill_time = stats->get_elapsed_sec(); // PREFILL END
    stats->prefill_time = prefill_time;

    for (uint32_t token : tokens) {
        sampler->add_token(token); // for presence penalty
    }

    size_t num_iters = std::min(config->num_iters, config->user_max_seq_len - tokens.size());

    stats->start_timer(); // GENERATE START
    size_t generated = 0;
    
    // GENERATION
    while (generated < num_iters) {

        uint32_t next_token = sampler->sample(run_state->logits.get());
        sampler->add_token(next_token); // for presence penalty
        tokens.push_back(next_token);
        std::cout << tokenizer->decode_token(next_token) << std::flush;
        generated++;
        
        if (generated == 1) {
            stats->ttft = prefill_time + stats->get_elapsed_sec(); // time-to-first-token
        }
        
        if (next_token == tokenizer->get_eos_id()) { 
            break;
        }
        
        run_state->cur_pos = tokens.size()-1;
        run_state->token_id = next_token;
        run_state->compute_logits = true;
        forward(run_state);

        total_passes++;
        kv_cache_bytes_read += run_state->kv_bytes_per_pos * (run_state->cur_pos + 1); // read from pos [0, 1, ..., i]
    }

    std::cout << "\n" << std::endl;

    float throughput_time = stats->get_elapsed_sec(); // GENERATE END
    stats->num_tokens_gen = generated;
    stats->throughput = generated/throughput_time;

    // total mem reads
    size_t total_bytes = total_passes * this->get_read_bytes() + kv_cache_bytes_read;
    
    stats->bandwidth = total_bytes / 1e9 / (throughput_time + prefill_time); // mem bandwidth in GB/sec
    stats->print_stats();
}

void BaseModel::benchmark() {
    std::mt19937 gen(config->seed);
    
    // 512 randomly generated token ids for prefill
    // intended to emulate pp512 in llama-bench (we won't win here, our impl. is not batched.)
    std::uniform_int_distribution<> distrib(0, config->vocab_size-1);

    stats->start_timer(); // PREFILL START

    size_t total_passes = 0;
    size_t kv_cache_bytes_read = 0;

    for(size_t i = 0; i<512; ++i) {
        run_state->cur_pos = i;
        run_state->token_id = distrib(gen);
        run_state->compute_logits = (i == 512-1);
        forward(run_state);
        total_passes++;
        kv_cache_bytes_read += run_state->kv_bytes_per_pos * (run_state->cur_pos+1);
    }

    float prefill_time = stats->get_elapsed_sec(); // PREFILL END
    stats->prefill_time = prefill_time;
    stats->ttft = prefill_time;

    // intended to emulate tg128 in llama-bench
    stats->start_timer(); // GENERATE START

    for (size_t i=0; i<128; ++i) {
        run_state->cur_pos = i;
        run_state->token_id = distrib(gen);
        run_state->compute_logits = true;
        forward(run_state);

        total_passes++;
        kv_cache_bytes_read += run_state->kv_bytes_per_pos * (run_state->cur_pos + 1); // read from pos [0, 1, ..., i]
    }

    float throughput_time = stats->get_elapsed_sec(); // GENERATE END
    stats->num_tokens_gen = 128;
    stats->throughput = 128/throughput_time;

    // total mem reads (an overestimate because lm_head weights aren't read when compute_logits is false)
    size_t total_bytes = total_passes * this->get_read_bytes() + kv_cache_bytes_read;
    
    stats->bandwidth = total_bytes / 1e9 / (throughput_time+prefill_time); // mem bandwidth in GB/sec
    stats->print_stats();
}