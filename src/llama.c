#include "mlx_c/llama.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MLX_BACKEND
static mlx_array mock_mlx_array_new(void) {
    mlx_array a;
    a.ctx = NULL;
    return a;
}

static mlx_array mock_mlx_array_new_data(void *data, const int *shape, int ndim) {
    mlx_array a;
    a.ctx = data ? data : (void*)1; // Just non-null for test check
    return a;
}

static void mock_mlx_array_free(mlx_array a) {
    // nothing
}
#endif

static mlx_array load_tensor_array(gguf_context *ctx, const char *name) {
    gguf_tensor_info *t = gguf_get_tensor(ctx, name);
    if (!t) {
#ifdef MLX_BACKEND
        return mlx_array_new();
#else
        return mock_mlx_array_new();
#endif
    }
    
    int shape[4] = {1, 1, 1, 1};
    for (uint32_t i = 0; i < t->ndim && i < 4; i++) {
        shape[i] = (int)t->shape[i];
    }
    
#ifdef MLX_BACKEND
    // In real MLX backend, we map to mlx_dtype. Here we assume FLOAT32 for simplicity.
    // In practice, we'd map GGUF types like F16 to MLX_FLOAT16.
    return mlx_array_new_data(t->data, shape, t->ndim, MLX_FLOAT32);
#else
    return mock_mlx_array_new_data(t->data, shape, t->ndim);
#endif
}

llama_model* llama_model_load(const char *gguf_path) {
    gguf_context *ctx = gguf_load(gguf_path);
    if (!ctx) return NULL;
    
    llama_model *model = calloc(1, sizeof(llama_model));
    
    // Parse some basic metadata
    for (uint64_t i = 0; i < ctx->header.kv_count; i++) {
                if (strcmp(ctx->kvs[i].name, "llama.block_count") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->n_layers = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->n_layers = *(uint64_t*)ctx->kvs[i].value;
            }
        } else         if (strcmp(ctx->kvs[i].name, "llama.attention.head_count") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->n_heads = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->n_heads = *(uint64_t*)ctx->kvs[i].value;
            }
        } else         if (strcmp(ctx->kvs[i].name, "llama.attention.head_count_kv") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->n_kv_heads = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->n_kv_heads = *(uint64_t*)ctx->kvs[i].value;
            }
        } else         if (strcmp(ctx->kvs[i].name, "llama.embedding_length") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->n_embd = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->n_embd = *(uint64_t*)ctx->kvs[i].value;
            }
        } else         if (strcmp(ctx->kvs[i].name, "llama.feed_forward_length") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->n_ff = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->n_ff = *(uint64_t*)ctx->kvs[i].value;
            }
        } else if (strcmp(ctx->kvs[i].name, "llama.context_length") == 0) {
            if (ctx->kvs[i].type == GGUF_TYPE_UINT32) {
                model->params.n_ctx = *(uint32_t*)ctx->kvs[i].value;
            } else if (ctx->kvs[i].type == GGUF_TYPE_UINT64) {
                model->params.n_ctx = *(uint64_t*)ctx->kvs[i].value;
            }
        }
    }
    
    model->tok_embeddings = load_tensor_array(ctx, "token_embd.weight");
    model->norm = load_tensor_array(ctx, "output_norm.weight");
    model->output = load_tensor_array(ctx, "output.weight");
    
    if (model->n_layers > 0) {
        model->layers = calloc(model->n_layers, sizeof(llama_layer));
        for (int i = 0; i < model->n_layers; i++) {
            char name[256];
            
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            model->layers[i].wq = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            model->layers[i].wk = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            model->layers[i].wv = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            model->layers[i].wo = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
            model->layers[i].w1 = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
            model->layers[i].w2 = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
            model->layers[i].w3 = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
            model->layers[i].rms_att_w = load_tensor_array(ctx, name);
            
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
            model->layers[i].rms_ffn_w = load_tensor_array(ctx, name);
        }
    }
    
    gguf_free(ctx);
    return model;
}

static void free_array(mlx_array a) {
#ifdef MLX_BACKEND
    if (a.ctx) mlx_array_free(a);
#else
    mock_mlx_array_free(a);
#endif
}

void llama_model_free(llama_model *model) {
    if (!model) return;
    
    free_array(model->tok_embeddings);
    free_array(model->norm);
    free_array(model->output);
    
    if (model->layers) {
        for (int i = 0; i < model->n_layers; i++) {
            free_array(model->layers[i].wq);
            free_array(model->layers[i].wk);
            free_array(model->layers[i].wv);
            free_array(model->layers[i].wo);
            free_array(model->layers[i].w1);
            free_array(model->layers[i].w2);
            free_array(model->layers[i].w3);
            free_array(model->layers[i].rms_att_w);
            free_array(model->layers[i].rms_ffn_w);
        }
        free(model->layers);
    }
    
    free(model);
}
