#include <metal_stdlib>

using namespace metal;

struct RMSNormArgs {
    uint dim;
    float eps;
    uint stride;
};

struct SoftmaxArgs {
    uint dim;
    uint stride;
};

struct RopeArgs {
    uint d_rotary;
    uint d_head;
    float freq_base;
    uint pos;
};

struct CacheArgs {
    uint layer_idx;
    uint cur_pos;
    uint seq_len;
    uint n_kv_heads;
    uint d_head;
};

struct AttnScoreArgs {
    uint d_head;
    uint kv_dim;
    uint kv_mul;
    uint kv_len;
    uint seq_len;
    size_t loff;
};

struct AttnOutArgs {
    uint seq_len;
    uint kv_len;
    uint d_head;
    uint kv_dim;
    uint kv_mul;
    size_t loff;
};

struct MoeTopkArgs {
    uint n_experts;
    uint k;
};

// much of this is inspired by calm (see repo README acknowledgements for more info)
// blockreduce algs used when we need to compute sum or max across
// all threadgroups, usually 1024 threads i.e. 32 thrgps of 32 threads each

// reduce w/in each SIMD gp and assign, then reduce across SIMD gps
inline float blockreduce_max(threadgroup float* vs, float val, uint id) {

    val = simd_max(val);

    vs[id/32] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return simd_max(vs[id%32]);
}

// reduce w/in each SIMD gp and assign, then reduce across SIMD gps
inline float blockreduce_sum(threadgroup float* vs, float val, uint id) {
    
    val = simd_sum(val);
    
    vs[id/32] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return simd_sum(vs[id%32]);
}

// if weight is 1D i.e. vec-vec dotprod, always pass in row_idx = 0
inline float matmul_row(
    const device float* weight,
    const device float* x_in,
    uint row_idx,
    uint d_in,
    uint tid
) {
    int lane = tid%32;
    float val = 0.0f;
    for (uint j=lane*2; j<d_in; j+=32*2) {
        float2 w = *(device float2*)&weight[row_idx*d_in + j];
        float2 x = *(device float2*)&x_in[j];
        val += w.x*x.x + w.y*x.y;
    }
    return simd_sum(val);
}

inline float attn_dot(
    const device float* att_scores,
    const device float* vh,
    uint kv_len,
    uint kv_dim,
    uint dim_idx,
    uint tid
) {
    int lane = tid % 32;
    float out = 0.0f;
    
    for (uint pos=lane; pos<kv_len; pos+=32) {
        out += att_scores[pos]*vh[pos*kv_dim+dim_idx];
    }
    
    return simd_sum(out);
}

// how many threads to spawn depends on where this is used
// usually d_model threads
kernel void resadd(
    device float* x [[ buffer(0) ]],
    const device float* res [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    x[tid] += res[tid];
}

// how many threads to spawn depends on where this is used
// usually d_model threads
kernel void weight_resadd(
    constant float& weight [[ buffer(0) ]],
    device float* x [[ buffer(1) ]],
    const device float* res [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    x[tid] += res[tid]*weight;
}

// d_out threadgroups, 32 threads each
kernel void linear_proj(
    constant int& d_in [[ buffer(0) ]],
    const device float* weight [[ buffer(1) ]],
    const device float* x_in [[ buffer(2) ]],
    device float* x_out [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint row_idx [[ threadgroup_position_in_grid ]]
) {
    x_out[row_idx] = matmul_row(weight, x_in, row_idx, d_in, tid);
}

// spawn d_model threads for this
kernel void embed(
    constant int& token_offset [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    const device float* weight [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    out[tid] = weight[token_offset + tid];
}

// n_heads thrgp of 1024 threads each for per-head norms
// 1 thrgp of 1024 threads for single norm (stride=0)
kernel void rmsnorm(
    constant RMSNormArgs& args [[ buffer(0) ]],
    const device float* weight [[ buffer(1) ]],
    const device float* in [[ buffer(2) ]],
    device float* out [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint thrs_per_thrgp [[ threads_per_threadgroup ]],
    uint head_idx [[ threadgroup_position_in_grid ]]
) {
    const device float* head_in = in + head_idx*args.stride;
    device float* head_out = out + head_idx*args.stride;
    
    threadgroup float shared_sum[32];
    
    float sum_sq = 0.0f;
    for (uint i=tid; i<args.dim; i+=thrs_per_thrgp) {
        float val = head_in[i];
        sum_sq += val*val;
    }
    
    sum_sq = blockreduce_sum(shared_sum, sum_sq, tid);
    float scale = rsqrt(sum_sq/float(args.dim) + args.eps);
    
    for (uint i=tid; i<args.dim; i+=thrs_per_thrgp) {
        head_out[i] = head_in[i]*weight[i]*scale;
    }
}

// n_heads thrgp of 1024 threads each for per-head norms
// 1 thrgp of 1024 threads for single norm (stride=0)
kernel void softmax(
    constant SoftmaxArgs& args [[ buffer(0) ]],
    device float* buf [[ buffer(1) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint thrs_per_thrgp [[ threads_per_threadgroup ]],
    uint head_idx [[ threadgroup_position_in_grid ]]
) {
    device float* head_buf = buf + head_idx*args.stride;
    threadgroup float shared[32];

    float max_val = -FLT_MAX;
    for (uint i=tid; i<args.dim; i+=thrs_per_thrgp) {
        // (each thr finds max across all its elements)
        max_val = max(max_val, head_buf[i]);
    }

    // max across all thrs regardless of thrgp
    max_val = blockreduce_max(shared, max_val, tid); 

    float sum = 0.0f;
    for (uint i=tid; i<args.dim; i+=thrs_per_thrgp) {
        // (each thr performs opn over all its elements)
        head_buf[i] = exp(head_buf[i]-max_val);
        sum += head_buf[i];
    }

    // sum across all thrs regardless of thrgp
    sum = blockreduce_sum(shared, sum, tid);

    float scale = 1.0f/sum;
    for (uint i=tid; i<args.dim; i+=thrs_per_thrgp) {
        head_buf[i] *= scale;
    }
}

// dispatch2d w [d_rotary/2, n_heads] thrgp x,y; 32 thrs per gp
kernel void il_rope(
    constant RopeArgs& args [[ buffer(0) ]],
    device float* buf [[ buffer(1) ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 gid [[ threadgroup_position_in_grid ]]
) {
    uint pair_idx = gid.x*32 + tid.x;
    uint head_idx = gid.y;
    
    device float* head_buf = buf + head_idx*args.d_head;
    
    float freq = 1.0f / pow(args.freq_base, 2.0f*pair_idx/args.d_rotary);
    float angle = args.pos*freq;

    // pair_idx, pair_idx+1 rotated as pair
    uint idx = 2*pair_idx;
    
    float x_0 = head_buf[idx];
    float x_1 = head_buf[idx+1];

    head_buf[idx] = cos(angle)*x_0 - sin(angle)*x_1;
    head_buf[idx+1] = sin(angle)*x_0 + cos(angle)*x_1;
}

// dispatch2d w [d_rotary/2, n_heads or n_kv_heads] as thrgp x,y; 32 thrs per gp
kernel void neox_rope(
    constant RopeArgs& args [[ buffer(0) ]],
    device float* buf [[ buffer(1) ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 gid [[ threadgroup_position_in_grid ]]
) {

    uint pair_idx = gid.x*32 + tid.x;
    uint head_idx = gid.y;

    device float* head_buf = buf + head_idx*args.d_head;
    
    float freq = 1.0f / pow(args.freq_base, 2.0f*pair_idx/args.d_rotary);
    float angle = args.pos*freq;
    
    float x_0 = head_buf[pair_idx];
    float x_1 = head_buf[pair_idx + args.d_rotary/2];
    
    head_buf[pair_idx] = cos(angle)*x_0 - sin(angle)*x_1;
    head_buf[pair_idx + args.d_rotary/2] = sin(angle)*x_0 + cos(angle)*x_1;
}

// n_kv_heads * d_head (kv_dim) threads
kernel void write_kv_cache(
    constant CacheArgs& args [[ buffer(0) ]],
    const device float* k_in [[ buffer(1) ]],
    const device float* v_in [[ buffer(2) ]],
    device float* k_cache [[ buffer(3) ]],
    device float* v_cache [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint kv_dim = args.n_kv_heads*args.d_head;

    ulong loff = ulong(args.layer_idx) * args.seq_len * kv_dim;
    ulong poff = ulong(args.cur_pos) * kv_dim;
    ulong cache_idx = loff + poff + tid;

    k_cache[cache_idx] = k_in[tid];
    v_cache[cache_idx] = v_in[tid];
}


// dispatch2d w [kv_len, n_heads] as thrgp x,y; 32 thrs per gp
kernel void attn_score(
    constant AttnScoreArgs& args [[ buffer(0) ]],
    const device float* q [[ buffer(1) ]],
    const device float* k_cache [[ buffer(2) ]],
    device float* att_scores [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 gid [[ threadgroup_position_in_grid ]]
) {
    uint pos = gid.x;
    uint head_idx = gid.y;
    
    int kv_head = head_idx / args.kv_mul;
    const device float* qh = q + head_idx*args.d_head;
    const device float* kh = k_cache + args.loff + pos*args.kv_dim + kv_head*args.d_head;
    device float* atth = att_scores + head_idx*args.seq_len;
    
    float score = matmul_row(qh, kh, 0, args.d_head, tid.x);
    float scale = 1.0f / sqrt(float(args.d_head));
    
    if (tid.x == 0) {
        atth[pos] = score*scale;
    }
}

// dispatch2d w [d_head, n_heads] as thrgp x,y; 32 thrs per gp
kernel void attn_out(
    constant AttnOutArgs& args [[ buffer(0) ]],
    const device float* att_scores [[ buffer(1) ]],
    const device float* v_cache [[ buffer(2) ]],
    device float* att_out [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 gid [[ threadgroup_position_in_grid ]]
) {
    uint head_idx = gid.y;
    uint dim_idx = gid.x;
    
    if (dim_idx >= args.d_head) return;
    
    int kv_head = head_idx / args.kv_mul;
    const device float* atth = att_scores + head_idx*args.seq_len;
    const device float* vh = v_cache + args.loff + kv_head*args.d_head;
    device float* out = att_out + head_idx*args.d_head;
    
    float result = attn_dot(atth, vh, args.kv_len, args.kv_dim, dim_idx, tid.x);
    
    if (tid.x == 0) {
        out[dim_idx] = result;
    }
}

// d_ff threads
kernel void silu_mul(
    device float* gate_buf [[ buffer(0) ]],
    const device float* up_buf [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    float x = gate_buf[tid];
    gate_buf[tid] = (x / (1.0f + exp(-x))) * up_buf[tid];
}

// one threadgroup, 32 threads
// supports at most 32 experts for now
// (8 threadgroups would allow for 256 experts max)
// credit: calm
kernel void moe_topk(
    constant MoeTopkArgs& args [[ buffer(0) ]],
    const device float* scores [[ buffer(1) ]],
    device int* top_experts [[ buffer(2) ]],
    device float* top_scores [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint lane = tid % 32;
    
    float score = (lane < args.n_experts) ? scores[lane] : -FLT_MAX;
    uint packed = (as_type<uint>(score) & 0xFFFFFF00) | lane;
    
    for (uint i=0; i<args.k; ++i) {
        uint max_packed = simd_max(packed);
        
        float max_score = as_type<float>(max_packed & 0xFFFFFF00);
        uint max_expert = max_packed & 0xFF;
        
        top_experts[i] = max_expert;
        top_scores[i] = max_score;
        
        if (lane == max_expert) {
            packed = 0;
        }
    }
}