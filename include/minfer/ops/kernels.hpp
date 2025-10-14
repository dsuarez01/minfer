#pragma once

#include <cstdint>
#include <vector>

// forward decls.
struct fp32_t;
struct fp16_t;
struct bf16_t;
struct Tensor;
using TPtr = std::shared_ptr<Tensor>;

float silu(float);
void softmax(float*, const float*, int);
void il_rope(float*, const float*, int, int, int, float, int);
void neox_rope(float*, const float*, int, int, int, float, int);
void attn(float*, float*, const float*, const float*, const float*, int, int, int);

void rmsnorm(float*, const float*, const fp32_t&, int, float);
void route(const float*, int*, float*, float*, const fp32_t&, int, int, int);

void embed(float*, const TPtr, uint32_t, size_t);

void embed_fp32(float*, const fp32_t&, uint32_t, size_t);
void embed_fp16(float*, const fp16_t&, uint32_t, size_t);
void embed_bf16(float*, const bf16_t&, uint32_t, size_t);

void matmul(float*, const float*, const TPtr, size_t, int, int);

void matmul_fp32(float*, const float*, const fp32_t&, size_t, int, int);
void matmul_fp16(float*, const float*, const fp16_t&, size_t, int, int);
void matmul_bf16(float*, const float*, const bf16_t&, size_t, int, int);
