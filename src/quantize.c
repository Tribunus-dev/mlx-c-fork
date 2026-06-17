#include "../include/quantize.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            return (sign ? -1.0f : 1.0f) * ldexpf((float)frac, -24);
        }
    } else if (exp == 0x1f) {
        if (frac == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        uint32_t fp32_val = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        float result;
        memcpy(&result, &fp32_val, sizeof(result));
        return result;
    }
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t fp32_val;
    memcpy(&fp32_val, &f, sizeof(fp32_val));

    uint32_t sign = (fp32_val >> 16) & 0x8000;
    int32_t exp = ((fp32_val >> 23) & 0xff) - 127;
    uint32_t frac = fp32_val & 0x7fffff;

    if (exp > 15) {
        return sign | 0x7c00;
    } else if (exp < -14) {
        if (exp < -24) {
            return sign;
        }
        uint32_t shifted_frac = (frac | 0x800000) >> (-exp - 14);
        if ((shifted_frac & 0x1fff) > 0x1000 || ((shifted_frac & 0x1fff) == 0x1000 && (shifted_frac & 0x2000))) {
            shifted_frac += 0x2000;
        }
        return sign | (shifted_frac >> 13);
    } else {
        uint32_t shifted_frac = frac;
        if ((shifted_frac & 0x1fff) > 0x1000 || ((shifted_frac & 0x1fff) == 0x1000 && (shifted_frac & 0x2000))) {
            shifted_frac += 0x2000;
            if (shifted_frac & 0x800000) {
                shifted_frac &= 0x7fffff;
                exp++;
                if (exp > 15) {
                    return sign | 0x7c00; 
                }
            }
        }
        return sign | ((exp + 15) << 10) | (shifted_frac >> 13);
    }
}

void quantize_row_q4_0(const float* src, void* dst, int64_t n) {
    block_q4_0* blocks = (block_q4_0*)dst;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        const float* block_src = src + i * 16;
        float max_val = 0.0f;
        
        for (int j = 0; j < 16; j++) {
            float abs_val = fabsf(block_src[j]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
        
        float scale = max_val / -8.0f;
        blocks[i].scale = fp32_to_fp16(scale);
        
        float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        
        for (int j = 0; j < 8; j++) {
            float v0 = block_src[j] * inv_scale;
            float v1 = block_src[j + 8] * inv_scale;
            
            int8_t i0 = (int8_t)roundf(v0);
            int8_t i1 = (int8_t)roundf(v1);
            
            if (i0 < -8) i0 = -8;
            if (i0 > 7) i0 = 7;
            if (i1 < -8) i1 = -8;
            if (i1 > 7) i1 = 7;
            
            blocks[i].qs[j] = ((i0 + 8) & 0x0f) | (((i1 + 8) & 0x0f) << 4);
        }
    }
}

void dequantize_row_q4_0(const void* src, float* dst, int64_t n) {
    const block_q4_0* blocks = (const block_q4_0*)src;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        float scale = fp16_to_fp32(blocks[i].scale);
        float* block_dst = dst + i * 16;
        
        for (int j = 0; j < 8; j++) {
            uint8_t q = blocks[i].qs[j];
            int8_t i0 = (q & 0x0f) - 8;
            int8_t i1 = ((q >> 4) & 0x0f) - 8;
            
            block_dst[j] = i0 * scale;
            block_dst[j + 8] = i1 * scale;
        }
    }
}

void quantize_row_q4_1(const float* src, void* dst, int64_t n) {
    block_q4_1* blocks = (block_q4_1*)dst;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        const float* block_src = src + i * 16;
        float min_val = block_src[0];
        float max_val = block_src[0];
        
        for (int j = 1; j < 16; j++) {
            if (block_src[j] < min_val) min_val = block_src[j];
            if (block_src[j] > max_val) max_val = block_src[j];
        }
        
        float scale = (max_val - min_val) / 15.0f;
        blocks[i].scale = fp32_to_fp16(scale);
        blocks[i].min = fp32_to_fp16(min_val);
        
        float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        
        for (int j = 0; j < 8; j++) {
            float v0 = (block_src[j] - min_val) * inv_scale;
            float v1 = (block_src[j + 8] - min_val) * inv_scale;
            
            int8_t i0 = (int8_t)roundf(v0);
            int8_t i1 = (int8_t)roundf(v1);
            
            if (i0 < 0) i0 = 0;
            if (i0 > 15) i0 = 15;
            if (i1 < 0) i1 = 0;
            if (i1 > 15) i1 = 15;
            
            blocks[i].qs[j] = (i0 & 0x0f) | ((i1 & 0x0f) << 4);
        }
    }
}

void dequantize_row_q4_1(const void* src, float* dst, int64_t n) {
    const block_q4_1* blocks = (const block_q4_1*)src;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        float scale = fp16_to_fp32(blocks[i].scale);
        float min_val = fp16_to_fp32(blocks[i].min);
        float* block_dst = dst + i * 16;
        
        for (int j = 0; j < 8; j++) {
            uint8_t q = blocks[i].qs[j];
            int8_t i0 = q & 0x0f;
            int8_t i1 = (q >> 4) & 0x0f;
            
            block_dst[j] = i0 * scale + min_val;
            block_dst[j + 8] = i1 * scale + min_val;
        }
    }
}

void quantize_fp32_to_q4_0(int64_t n, const float* src, char* dst) {
    quantize_row_q4_0(src, dst, n);
}
