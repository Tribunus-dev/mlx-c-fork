#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// block_q4_0: 16 x int4 weights (packed) + fp16 scale = 18 bytes per 16 weights
typedef struct {
    uint16_t scale; // fp16
    uint8_t qs[8];  // 16 int4 weights packed into 8 bytes
} block_q4_0;

// block_q4_1: 16 x int4 weights (packed) + fp16 scale + fp16 min = 20 bytes per 16 weights
typedef struct {
    uint16_t scale; // fp16
    uint16_t min;   // fp16
    uint8_t qs[8];  // 16 int4 weights packed into 8 bytes
} block_q4_1;

void quantize_row_q4_0(const float* src, void* dst, int64_t n);
void dequantize_row_q4_0(const void* src, float* dst, int64_t n);

void quantize_row_q4_1(const float* src, void* dst, int64_t n);
void dequantize_row_q4_1(const void* src, float* dst, int64_t n);

void quantize_fp32_to_q4_0(int64_t n, const float* src, char* dst);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZE_H
