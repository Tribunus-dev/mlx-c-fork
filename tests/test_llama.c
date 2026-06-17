#include "mlx_c/llama.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void create_mock_gguf(const char* filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    // GGUF Magic and version
    uint32_t magic = GGUF_MAGIC;
    fwrite(&magic, 4, 1, f);
    uint32_t version = 1;
    fwrite(&version, 4, 1, f);
    
    uint64_t tensor_count = 2; // tok_embeddings, blk.0.attn_q.weight
    fwrite(&tensor_count, 8, 1, f);
    uint64_t kv_count = 2; // block_count, alignment
    fwrite(&kv_count, 8, 1, f);
    
    // KV 1: llama.block_count = 1
    uint64_t len1 = strlen("llama.block_count");
    fwrite(&len1, 8, 1, f);
    fwrite("llama.block_count", 1, len1, f);
    uint32_t t_kv1 = GGUF_TYPE_UINT32;
    fwrite(&t_kv1, 4, 1, f);
    uint32_t block_count = 1;
    fwrite(&block_count, 4, 1, f);
    
    // KV 2: general.alignment = 32
    uint64_t len2 = strlen("general.alignment");
    fwrite(&len2, 8, 1, f);
    fwrite("general.alignment", 1, len2, f);
    uint32_t t_kv2 = GGUF_TYPE_UINT32;
    fwrite(&t_kv2, 4, 1, f);
    uint32_t alignment = 32;
    fwrite(&alignment, 4, 1, f);
    
    // Tensor 1: token_embd.weight
    uint64_t len3 = strlen("token_embd.weight");
    fwrite(&len3, 8, 1, f);
    fwrite("token_embd.weight", 1, len3, f);
    uint32_t ndim1 = 2;
    fwrite(&ndim1, 4, 1, f);
    uint64_t shape1[2] = {128, 128}; // [128, 128]
    fwrite(shape1, 8, 2, f);
    uint32_t t_t1 = GGUF_TYPE_FLOAT32;
    fwrite(&t_t1, 4, 1, f);
    uint64_t off1 = 0; // offset 0 from data start
    fwrite(&off1, 8, 1, f);
    
    // Tensor 2: blk.0.attn_q.weight
    uint64_t len4 = strlen("blk.0.attn_q.weight");
    fwrite(&len4, 8, 1, f);
    fwrite("blk.0.attn_q.weight", 1, len4, f);
    uint32_t ndim2 = 2;
    fwrite(&ndim2, 4, 1, f);
    uint64_t shape2[2] = {64, 64};
    fwrite(shape2, 8, 2, f);
    uint32_t t_t2 = GGUF_TYPE_FLOAT32;
    fwrite(&t_t2, 4, 1, f);
    uint64_t off2 = 128 * 128 * 4; // offset after first tensor
    fwrite(&off2, 8, 1, f);
    
    // Align data offset to 32
    long curr_pos = ftell(f);
    long padding = (32 - (curr_pos % 32)) % 32;
    for(long i = 0; i < padding; i++) {
        fputc(0, f);
    }
    
    // Write tensor data (dummy zeros)
    long data_size = off2 + 64 * 64 * 4;
    for(long i = 0; i < data_size; i++) {
        fputc(0, f);
    }
    
    fclose(f);
}

int main() {
    const char *test_file = "test_llama.gguf";
    create_mock_gguf(test_file);
    
    llama_model *model = llama_model_load(test_file);
    if (!model) {
        printf("Failed to load model\n");
        return 1;
    }
    
    if (model->n_layers != 1) {
        printf("Expected n_layers = 1, got %d\n", model->n_layers);
        return 1;
    }
    
    if (model->tok_embeddings.ctx == NULL) {
        printf("Failed to load tok_embeddings array\n");
        return 1;
    }
    
    if (model->layers[0].wq.ctx == NULL) {
        printf("Failed to load wq array\n");
        return 1;
    }
    
    llama_model_free(model);
    remove(test_file);
    
    printf("Tests passed.\n");
    return 0;
}
