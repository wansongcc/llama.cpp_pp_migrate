#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-adapter.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"
#include "../src/llama-context.h"
#include "../src/llama-memory-hybrid.h"

#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [--pp-replicas] [--pp-rpc endpoint-a endpoint-b] [--pp-model path] [--pp-low-vram] [--pp-replay-acceptance] [--pp-replay-costs] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_QWEN35MOE) {
        n_layer = 4; // cover a full layer followed by a recurrent layer at a PP boundary
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

static std::vector<float> pp_decode_tokens(
        llama_context & ctx, const std::vector<llama_token> & tokens, llama_pos pos = 0,
        std::vector<int64_t> * elapsed_us = nullptr) {
    std::vector<float> result;
    auto batch = llama_batch_init(1, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.n_tokens = 1;
        batch.token[0] = tokens[i];
        batch.pos[0] = pos + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        const int64_t start_us = ggml_time_us();
        if (llama_decode(&ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("PP fixture decode failed");
        }
        const auto * logits = llama_get_logits_ith(&ctx, 0);
        if (elapsed_us) {
            elapsed_us->push_back(ggml_time_us() - start_us);
        }
        const auto n_vocab = ctx.get_model().vocab.n_tokens();
        for (uint32_t j = 0; j < n_vocab; ++j) {
            if (!std::isfinite(logits[j])) {
                llama_batch_free(batch);
                throw std::runtime_error("PP fixture produced nonfinite logits");
            }
        }
        result.insert(result.end(), logits, logits + n_vocab);
    }
    llama_batch_free(batch);
    return result;
}

static void pp_append(std::vector<float> & dst, std::vector<float> src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

static std::pair<double, double> pp_compare_logits(
        const std::vector<float> & expected, const std::vector<float> & actual,
        double abs_tol = 1e-4, double rel_tol = 1e-4) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("PP logit result size changed");
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::abs(double(expected[i]) - double(actual[i]));
        const double rel = diff/std::max(std::abs(double(expected[i])), 1e-12);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        if (!std::isfinite(actual[i]) || diff > abs_tol + rel_tol*std::abs(double(expected[i]))) {
            throw std::runtime_error("PP layer migration changed logits beyond tolerance");
        }
    }
    return {max_abs, max_rel};
}

struct pp_timing_summary {
    int64_t minimum;
    int64_t median;
    int64_t p95;
};

static pp_timing_summary pp_summarize(std::vector<int64_t> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot summarize empty timings");
    }
    std::sort(values.begin(), values.end());
    const size_t p95 = std::min(values.size() - 1, size_t(std::ceil(0.95*values.size())) - 1);
    return {values.front(), values[values.size()/2], values[p95]};
}

static std::vector<std::vector<uint8_t>> pp_layer_state_bytes(
        const llama_memory_hybrid & memory, int32_t first, int32_t end) {
    std::vector<std::vector<uint8_t>> result;
    for (int32_t il = first; il < end; ++il) {
        for (auto * tensor : memory.layer_state_tensors(il)) {
            result.emplace_back(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, result.back().data(), 0, result.back().size());
        }
    }
    return result;
}

static std::vector<const void *> pp_layer_state_addresses(
        const llama_memory_hybrid & memory, int32_t first, int32_t end) {
    std::vector<const void *> result;
    for (int32_t il = first; il < end; ++il) {
        for (auto * tensor : memory.layer_state_tensors(il)) {
            result.push_back(tensor);
            result.push_back(tensor->buffer);
            result.push_back(tensor->data);
        }
    }
    return result;
}

static size_t pp_benchmark_valid_copy(
        llama_context & ctx, int first, int end, ggml_backend_dev_t destination,
        uint32_t checkpoint, uint32_t frontier, std::vector<int64_t> & timings) {
    auto * mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
    auto devices = mem->get_active_view()->layer_devices;
    for (int il = first; il < end; ++il) {
        devices[il] = destination;
    }
    const auto target = mem->clone_active_view(devices);
    struct copy_span {
        ggml_tensor * src;
        ggml_tensor * dst;
        size_t offset;
        size_t elements;
    };
    std::vector<copy_span> spans;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> saved;
    const auto & hp = ctx.get_model().hparams;
    size_t payload = 0;
    for (int il = first; il < end; ++il) {
        const auto source = mem->layer_state_tensors(il);
        const auto dest = mem->layer_state_tensors(target, il);
        for (size_t it = 0; it < source.size(); ++it) {
            auto * src = source[it];
            auto * dst = dest[it];
            saved.push_back({dst, std::vector<uint8_t>(ggml_nbytes(dst))});
            ggml_backend_tensor_get(dst, saved.back().second.data(), 0, saved.back().second.size());
            if (hp.is_recr(il)) {
                spans.push_back({src, dst, 0, size_t(ggml_nelements(src))});
            } else if (it == 0 || ctx.get_cparams().flash_attn) {
                const size_t width = it == 0 ? hp.n_embd_k_gqa(il) : hp.n_embd_v_gqa(il);
                spans.push_back({src, dst, checkpoint * ggml_row_size(src->type, width), (frontier - checkpoint) * width});
            } else {
                if (ggml_blck_size(src->type) != 1) {
                    throw std::runtime_error("transposed quantized V copy benchmark is unsupported");
                }
                for (uint32_t row = 0; row < hp.n_embd_v_gqa(il); ++row) {
                    spans.push_back({src, dst, (row * mem->get_mem_attn()->get_size() + checkpoint) * ggml_type_size(src->type), frontier - checkpoint});
                }
            }
        }
    }
    ggml_init_params params = {2 * spans.size() * ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr views(ggml_init(params));
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
    for (const auto & span : spans) {
        auto * src = ggml_view_1d(views.get(), span.src, span.elements, span.offset);
        auto * dst = ggml_view_1d(views.get(), span.dst, span.elements, span.offset);
        if (ggml_backend_view_init(src) != GGML_STATUS_SUCCESS || ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("valid copy view initialization failed");
        }
        copies.emplace_back(src, dst);
        payload += ggml_nbytes(src);
    }
    for (int repeat = 0; repeat < 110; ++repeat) {
        const auto start = ggml_time_us();
        for (const auto & copy : copies) {
            ggml_backend_tensor_copy(copy.first, copy.second);
        }
        if (repeat >= 10) {
            timings.push_back(ggml_time_us() - start);
        }
    }
    for (const auto & copy : copies) {
        std::vector<uint8_t> source(ggml_nbytes(copy.first));
        std::vector<uint8_t> destination(source.size());
        ggml_backend_tensor_get(copy.first, source.data(), 0, source.size());
        ggml_backend_tensor_get(copy.second, destination.data(), 0, destination.size());
        if (source != destination) {
            throw std::runtime_error("valid copy benchmark changed state bytes");
        }
    }
    for (const auto & item : saved) {
        ggml_backend_tensor_set(item.first, item.second.data(), 0, item.second.size());
    }
    return payload;
}

static bool pp_cuda_graph_stats(llama_context & ctx, ggml_backend_cuda_graph_stats & result) {
    using get_stats_t = bool (*)(ggml_backend_t, ggml_backend_cuda_graph_stats *);
    result = {};
    bool found = false;
    auto * sched = ctx.get_sched();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto * backend = ggml_backend_sched_get_backend(sched, i);
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto get_stats = (get_stats_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_graph_stats");
        ggml_backend_cuda_graph_stats current = {};
        if (!get_stats || !get_stats(backend, &current)) {
            continue;
        }
        found = true;
        result.graph_support += current.graph_support;
        result.graph_count += current.graph_count;
        result.enabled_graph_count += current.enabled_graph_count;
        result.warmup_complete_graph_count += current.warmup_complete_graph_count;
        result.executable_graph_count += current.executable_graph_count;
        result.direct_evaluations += current.direct_evaluations;
        result.warmup_completions += current.warmup_completions;
        result.warmup_resets += current.warmup_resets;
        result.captures += current.captures;
        result.instantiations += current.instantiations;
        result.updates += current.updates;
        result.launches += current.launches;
        result.evictions += current.evictions;
    }
    return found;
}

static ggml_backend_cuda_graph_stats pp_cuda_graph_stats_delta(
        const ggml_backend_cuda_graph_stats & end, const ggml_backend_cuda_graph_stats & begin) {
    ggml_backend_cuda_graph_stats result = {};
    result.direct_evaluations = end.direct_evaluations - begin.direct_evaluations;
    result.warmup_completions = end.warmup_completions - begin.warmup_completions;
    result.warmup_resets = end.warmup_resets - begin.warmup_resets;
    result.captures = end.captures - begin.captures;
    result.instantiations = end.instantiations - begin.instantiations;
    result.updates = end.updates - begin.updates;
    result.launches = end.launches - begin.launches;
    result.evictions = end.evictions - begin.evictions;
    return result;
}

struct pp_layer_compute_probe {
    int32_t first;
    int32_t end;
    bool active = false;
    int64_t start_us = 0;
    std::map<ggml_backend_dev_t, int64_t> layer_us;
    std::map<ggml_backend_dev_t, int64_t> moved_layer_us;

    void begin_token() {
        active = false;
        start_us = 0;
        layer_us.clear();
        moved_layer_us.clear();
    }

    int64_t get_layer_us(ggml_backend_dev_t dev) const {
        const auto it = layer_us.find(dev);
        return it == layer_us.end() ? 0 : it->second;
    }

    int64_t get_moved_layer_us(ggml_backend_dev_t dev) const {
        const auto it = moved_layer_us.find(dev);
        return it == moved_layer_us.end() ? 0 : it->second;
    }
};

static bool pp_layer_compute_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & probe = *static_cast<pp_layer_compute_probe *>(user_data);
    int32_t il = -1;
    const bool is_layer_output = sscanf(ggml_get_name(tensor), "l_out-%d", &il) == 1;
    if (ask) {
        if (!probe.active) {
            probe.active = true;
            probe.start_us = ggml_time_us();
        }
        return is_layer_output;
    }

    if (!is_layer_output || !tensor->buffer) {
        throw std::runtime_error("PP compute probe received an invalid layer output");
    }
    const int64_t elapsed_us = ggml_time_us() - probe.start_us;
    const auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    probe.layer_us[dev] += elapsed_us;
    if (il >= probe.first && il < probe.end) {
        probe.moved_layer_us[dev] += elapsed_us;
    }
    probe.active = false;
    return true;
}

static void pp_replay_matrix(llama_model & model, const std::vector<ggml_backend_dev_t> & devices, size_t seed) {
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    const auto tokens = get_tokens(511, model.vocab.n_tokens(), seed);
    std::vector<float> expected;
    {
        llama_context ctx(model, params);
        expected = pp_decode_tokens(ctx, tokens);
    }
    size_t cases = 0;
    for (const auto & interval : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {0, 2}}) {
        for (const bool reverse : {false, true}) {
            std::vector<ggml_backend_dev_t> placement;
            for (int il = 0; il < 4; ++il) {
                placement.push_back(il < (reverse ? interval.first : interval.second) ? devices[0] : devices[1]);
            }
            for (const int checkpoint : {0, 6, 255}) {
                for (const int gap : {0, 1, 8, 32, 128}) {
                    const int frontier = checkpoint + gap;
                    llama_context ctx(model, params, placement);
                    auto actual = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + checkpoint});
                    const auto id = ctx.migrate_layers_async_begin(interval.first, interval.second, reverse ? devices[0] : devices[1]);
                    if (!ctx.enable_layer_migration_replay(id, std::max(1, gap))) {
                        throw std::runtime_error("matrix replay enable failed");
                    }
                    pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + checkpoint, tokens.begin() + frontier}, checkpoint));
                    std::vector<uint8_t> state(llama_state_seq_get_size(&ctx, 0));
                    llama_state_seq_get_data(&ctx, state.data(), state.size(), 0);
                    if (!ctx.replay_layer_migration(id)) {
                        throw std::runtime_error("matrix replay failed");
                    }
                    std::vector<uint8_t> after(state.size());
                    llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
                    if (state != after || !ctx.migrate_layers_async_commit(id)) {
                        throw std::runtime_error("matrix replay changed active state or failed commit");
                    }
                    if (llama_state_seq_get_size(&ctx, 0) != state.size()) {
                        throw std::runtime_error("matrix replay changed state size");
                    }
                    llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
                    if (state != after) {
                        throw std::runtime_error("matrix reconstructed state differs");
                    }
                    pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + frontier, tokens.begin() + frontier + 128}, frontier));
                    if (!std::equal(actual.begin(), actual.end(), expected.begin())) {
                        throw std::runtime_error("matrix replay continuation differs");
                    }
                    printf("PP replay matrix [%d,%d) reverse=%d P=%d D=%d: state exact, 128 continuation logits exact\n",
                            interval.first, interval.second, reverse, checkpoint, gap);
                    ++cases;
                }
            }
        }
    }
    printf("PP replay matrix: %zu cases passed\n", cases);

    std::vector<int64_t> replay_us;
    std::vector<int64_t> switch_us;
    const std::vector<ggml_backend_dev_t> placement = {devices[0], devices[0], devices[1], devices[1]};
    llama_context ctx(model, params, placement);
    auto * mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
    const auto bank_sizes = mem->memory_breakdown();
    const auto source_addresses = pp_layer_state_addresses(*mem, 0, 2);
    std::vector<const void *> destination_addresses;
    std::vector<float> actual;
    size_t journal_peak = 0;
    size_t compute_peak = 0;
    for (int step = 0; step < 200; ++step) {
        const auto id = ctx.migrate_layers_async_begin(0, 2, devices[step % 2 == 0 ? 1 : 0]);
        if (!ctx.enable_layer_migration_replay(id, 1)) {
            throw std::runtime_error("roundtrip replay enable failed");
        }
        pp_append(actual, pp_decode_tokens(ctx, {tokens[step]}, step));
        if (!ctx.replay_layer_migration(id) || !ctx.migrate_layers_async_commit(id)) {
            throw std::runtime_error("roundtrip replay failed");
        }
        if (mem->memory_breakdown() != bank_sizes) {
            throw std::runtime_error("roundtrip replay allocated more state banks");
        }
        const auto addresses = pp_layer_state_addresses(*mem, 0, 2);
        if (step == 0) {
            destination_addresses = addresses;
        }
        if (addresses != (step % 2 == 0 ? destination_addresses : source_addresses)) {
            throw std::runtime_error("roundtrip replay changed state bank addresses");
        }
        const auto & stats = ctx.get_last_layer_migration_stats();
        if (step >= 10) {
            replay_us.push_back(stats.replay_us);
            switch_us.push_back(stats.switch_us);
        }
        journal_peak = std::max(journal_peak, stats.journal_peak_bytes);
        compute_peak = std::max(compute_peak, stats.replay_compute_bytes);
    }
    if (!std::equal(actual.begin(), actual.end(), expected.begin())) {
        throw std::runtime_error("roundtrip replay changed logits");
    }
    const auto replay = pp_summarize(replay_us);
    const auto promote = pp_summarize(switch_us);
    printf("PP replay stability: 100 roundtrips, logits exact, state bank addresses/sizes stable, samples=%zu, replay p50/p95=%.3f/%.3f ms, switch p50/p95=%.3f/%.3f ms, journal payload peak=%zu bytes, replay compute peak=%zu bytes\n",
            replay_us.size(), replay.median / 1000.0, replay.p95 / 1000.0, promote.median / 1000.0, promote.p95 / 1000.0, journal_peak, compute_peak);
}

static int test_layer_replicas(size_t seed, const std::vector<std::string> & rpc_endpoints) {
    std::vector<ggml_backend_dev_t> devices;
    if (!rpc_endpoints.empty()) {
        auto * rpc = ggml_backend_reg_by_name("RPC");
        using add_server_t = ggml_backend_reg_t (*)(const char *);
        auto add_server = rpc ? (add_server_t) ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server") : nullptr;
        if (!add_server) {
            fprintf(stderr, "SKIP: RPC backend is unavailable\n");
            return 77;
        }
        for (const auto & endpoint : rpc_endpoints) {
            auto * reg = add_server(endpoint.c_str());
            if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
                throw std::runtime_error("RPC endpoint has no device");
            }
            devices.push_back(ggml_backend_reg_dev_get(reg, 0));
        }
    }
    for (size_t i = 0; rpc_endpoints.empty() && i < ggml_backend_dev_count() && devices.size() < 2; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        fprintf(stderr, "SKIP: PP replica test requires two GPU devices\n");
        return 77;
    }

    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN3NEXT, true);
    gguf_set_val_u32(metadata.get(), "qwen3next.block_count", 4);
    auto params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> terminated_devices = devices;
    terminated_devices.push_back(nullptr);
    params.devices = terminated_devices.data();
    const float tensor_split[] = {1.0f, 0.0f};
    params.tensor_split = tensor_split;
    params.n_gpu_layers = 999;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, params));
    if (!model) {
        throw std::runtime_error("could not create PP replica fixture");
    }

    for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
        auto * source = model->dev_layer(il);
        auto * target = source == devices[0] ? devices[1] : devices[0];
        model->prepare_layer_replica(il, target);
        const auto & replica = model->layer_on_device(il, target);
        const std::string prefix = "blk." + std::to_string(il) + ".";
        size_t n_checked = 0;
        bool checked_experts = false;
        for (const auto & entry : model->tensors_by_name) {
            if (entry.first.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            auto * original = entry.second;
            auto * copied = model->layer_tensor_on_device(il, target, original);
            if (copied == original || copied->type != original->type ||
                    !ggml_are_same_shape(original, copied) || !ggml_are_same_stride(original, copied)) {
                throw std::runtime_error("replica must have separate storage and the original layout");
            }
            auto * buft = ggml_backend_buffer_get_type(copied->buffer);
            if (ggml_backend_buft_get_device(buft) != target) {
                throw std::runtime_error("replica is on the wrong device");
            }
            std::vector<uint8_t> expected(ggml_nbytes(original));
            std::vector<uint8_t> actual(expected.size());
            ggml_backend_tensor_get(original, expected.data(), 0, expected.size());
            ggml_backend_tensor_get(copied, actual.data(), 0, actual.size());
            if (actual != expected) {
                throw std::runtime_error("replica changed weight bytes");
            }
            if (original == model->layers[il].ffn_down_exps) {
                if (replica.ffn_down_exps != copied) {
                    throw std::runtime_error("expert field still references the source weights");
                }
                checked_experts = true;
            }
            ++n_checked;
        }
        if (n_checked == 0 || !checked_experts) {
            throw std::runtime_error("PP replica fixture did not exercise expert weights");
        }
        const auto sizes = model->memory_breakdown();
        model->prepare_layer_replica(il, target);
        if (&replica != &model->layer_on_device(il, target) || sizes != model->memory_breakdown()) {
            throw std::runtime_error("preparing an existing replica allocated another copy");
        }
        if (model->dev_layer(il) != source) {
            throw std::runtime_error("preparing a replica changed the model's default placement");
        }
        printf("PP replica layer %u: %zu tensors checked on %s\n", il, n_checked, ggml_backend_dev_name(target));
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.no_perf = false;
    const auto tokens = get_tokens(16, model->vocab.n_tokens(), seed);
    std::vector<float> expected;
    {
        llama_context ctx(*model, cparams);
        expected = pp_decode_tokens(ctx, tokens);
    }
    std::vector<ggml_backend_dev_t> placement;
    for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
        placement.push_back(model->dev_layer(il) == devices[0] ? devices[1] : devices[0]);
    }
    auto run_placement = [&]() {
        llama_context ctx(*model, cparams, placement);
        llama_adapter_lora adapter(model.get());
        llama_adapter_lora * adapters[] = {&adapter};
        float scales[] = {1.0f};
        if (ctx.set_adapters_lora(adapters, 1, scales)) {
            throw std::runtime_error("experimental placement allowed a runtime LoRA adapter");
        }
        std::vector<float> cvec((model->hparams.n_layer() - 1)*model->hparams.n_embd, 0.25f);
        if (ctx.set_adapter_cvec(cvec.data(), cvec.size(), model->hparams.n_embd, 1, model->hparams.n_layer())) {
            throw std::runtime_error("experimental placement allowed a runtime control vector");
        }
        if (llama_memory_can_shift(ctx.get_memory())) {
            throw std::runtime_error("experimental placement allowed context shift");
        }
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        if (ctx.set_sampler(0, sampler.get())) {
            throw std::runtime_error("experimental placement allowed backend sampling");
        }
        bool context_rejected = false;
        try {
            llama_context other(*model, cparams);
        } catch (const std::runtime_error &) {
            context_rejected = true;
        }
        if (!context_rejected) {
            throw std::runtime_error("experimental placement allowed a second active context");
        }
        bool preparation_rejected = false;
        try {
            model->prepare_layer_replica(0, placement[0]);
        } catch (const std::runtime_error &) {
            preparation_rejected = true;
        }
        if (!preparation_rejected) {
            throw std::runtime_error("weight replica preparation was allowed with an active context");
        }
        auto * mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
        if (!mem) {
            throw std::runtime_error("PP fixture requires hybrid memory");
        }
        for (size_t il = 0; il < placement.size(); ++il) {
            for (auto * tensor : mem->layer_state_tensors(il)) {
                if (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != placement[il]) {
                    throw std::runtime_error("state does not follow context placement");
                }
            }
        }
        return pp_decode_tokens(ctx, tokens);
    };
    if (run_placement() != expected) {
        throw std::runtime_error("initial placement changed inference results");
    }
    const int32_t intervals[][3] = {{1, 2, 1}, {2, 3, 0}, {0, 2, 1}, {2, 4, 0}};
    for (const auto & interval : intervals) {
        const int32_t first = interval[0];
        const int32_t end = interval[1];
        const auto target = devices[interval[2]];
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        auto * mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
        std::vector<std::vector<ggml_tensor *>> state_tensors_before;
        for (size_t il = 0; il < original.size(); ++il) {
            state_tensors_before.push_back(mem->layer_state_tensors(il));
        }
        const auto state_allocation_size = [](const std::map<ggml_backend_buffer_type_t, size_t> & sizes) {
            size_t result = 0;
            for (const auto & entry : sizes) {
                result += entry.second;
            }
            return result;
        };
        const size_t allocation_size_before = state_allocation_size(mem->memory_breakdown());

        bool invalid_rejected = false;
        try {
            ctx.migrate_layers_sync(first, end, original[first]);
        } catch (const std::invalid_argument &) {
            invalid_rejected = true;
        }
        if (!invalid_rejected) {
            throw std::runtime_error("same-device migration was not rejected");
        }
        invalid_rejected = false;
        try {
            ctx.migrate_layers_sync(1, 3, devices[1]);
        } catch (const std::invalid_argument &) {
            invalid_rejected = true;
        }
        if (!invalid_rejected) {
            throw std::runtime_error("multi-stage migration interval was not rejected");
        }
        for (size_t il = 0; il < original.size(); ++il) {
            if (ctx.get_layer_device(il) != original[il] || mem->layer_state_tensors(il) != state_tensors_before[il]) {
                throw std::runtime_error("rejected migration changed active placement or state");
            }
        }

        std::vector<uint8_t> state_before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_before.data(), state_before.size(), 0);
        std::vector<std::vector<uint8_t>> state_bytes;
        for (int32_t il = first; il < end; ++il) {
            for (auto * tensor : mem->layer_state_tensors(il)) {
                state_bytes.emplace_back(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor, state_bytes.back().data(), 0, state_bytes.back().size());
            }
        }
        ctx.migrate_layers_sync(first, end, target);
        if (state_allocation_size(mem->memory_breakdown()) != allocation_size_before) {
            throw std::runtime_error("synchronous move retained an active source state allocation");
        }
        size_t ti = 0;
        for (size_t il = 0; il < original.size(); ++il) {
            const auto state_tensors_after = mem->layer_state_tensors(il);
            if ((il >= (size_t) first && il < (size_t) end) == (state_tensors_after == state_tensors_before[il])) {
                throw std::runtime_error("synchronous move replaced the wrong layer state");
            }
            for (auto * tensor : state_tensors_after) {
                if (il >= (size_t) first && il < (size_t) end) {
                    std::vector<uint8_t> actual(ggml_nbytes(tensor));
                    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
                    if (actual != state_bytes.at(ti++)) {
                        throw std::runtime_error("synchronous move changed state bytes");
                    }
                }
                const auto expected_device = il >= (size_t) first && il < (size_t) end ? target : original[il];
                if (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != expected_device) {
                    throw std::runtime_error("synchronous move used the wrong state device");
                }
            }
        }
        std::vector<uint8_t> state_after(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_after.data(), state_after.size(), 0);
        if (state_before != state_after) {
            throw std::runtime_error("synchronous move changed valid sequence state or metadata");
        }
        auto middle = pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.begin() + 11}, 6);
        result.insert(result.end(), middle.begin(), middle.end());
        ctx.migrate_layers_sync(first, end, original[first]);
        if (state_allocation_size(mem->memory_breakdown()) != allocation_size_before) {
            throw std::runtime_error("reverse move retained an active destination state allocation");
        }
        auto suffix = pp_decode_tokens(ctx, {tokens.begin() + 11, tokens.end()}, 11);
        result.insert(result.end(), suffix.begin(), suffix.end());
        if (result != expected) {
            throw std::runtime_error("forward/reverse layer migration changed logits");
        }
        printf("PP synchronous migration [%d, %d): state and logits verified\n", first, end);
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        auto * mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
        std::vector<std::vector<ggml_tensor *>> state_tensors_before;
        for (size_t il = 0; il < original.size(); ++il) {
            state_tensors_before.push_back(mem->layer_state_tensors(il));
        }
        std::vector<uint8_t> state_before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_before.data(), state_before.size(), 0);

        bool failed = false;
        try {
            ctx.migrate_layers_sync(1, 2, devices[1], true);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        if (!failed) {
            throw std::runtime_error("injected post-switch migration failure was not reported");
        }
        for (size_t il = 0; il < original.size(); ++il) {
            if (ctx.get_layer_device(il) != original[il] || mem->layer_state_tensors(il) != state_tensors_before[il]) {
                throw std::runtime_error("failed migration did not restore placement and state bindings");
            }
        }
        std::vector<uint8_t> state_after(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_after.data(), state_after.size(), 0);
        if (state_before != state_after) {
            throw std::runtime_error("failed migration changed sequence state");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.end()}, 6));
        if (result != expected) {
            throw std::runtime_error("failed migration changed continued inference");
        }
        printf("PP synchronous migration rollback: placement, state, and logits verified\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        std::vector<uint8_t> state_before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_before.data(), state_before.size(), 0);
        const auto request_id = ctx.migrate_layers_async_begin(1, 2, devices[1]);
        if (ctx.get_layer_device(1) != devices[0]) {
            throw std::runtime_error("async migration begin changed active placement");
        }
        if (!ctx.warmup_layer_migration_candidate(request_id)) {
            throw std::runtime_error("candidate CUDA Graph warmup failed");
        }
        std::vector<uint8_t> state_after_warmup(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, state_after_warmup.data(), state_after_warmup.size(), 0);
        if (state_after_warmup != state_before) {
            throw std::runtime_error("candidate CUDA Graph warmup changed active state");
        }
        if (!ctx.migrate_layers_async_commit(request_id)) {
            throw std::runtime_error("async migration commit was unexpectedly stale");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.end()}, 6));
        if (result != expected) {
            throw std::runtime_error("async migration changed logits");
        }
        printf("PP async migration: deferred commit, candidate warmup, and logits verified\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto request_id = ctx.migrate_layers_async_begin(0, 2, devices[1]);
        if (!ctx.enable_layer_migration_shadow(request_id)) {
            throw std::runtime_error("failed to enable candidate shadow compute");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.begin() + 11}, 6));
        if (!ctx.migrate_layers_async_commit(request_id)) {
            throw std::runtime_error("shadow candidate did not catch the active frontier");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 11, tokens.end()}, 11));
        if (result != expected) {
            throw std::runtime_error("shadow compute migration changed logits");
        }
        printf("PP shadow compute: mixed state tracked to hot commit\n");
    }
    for (const auto & interval : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {0, 2}}) {
        for (const bool reverse : {false, true}) {
            std::vector<ggml_backend_dev_t> original;
            for (int il = 0; il < 4; ++il) {
                original.push_back(il < (reverse ? interval.first : interval.second) ? devices[0] : devices[1]);
            }
            auto target_placement = original;
            for (int il = interval.first; il < interval.second; ++il) {
                target_placement[il] = reverse ? devices[0] : devices[1];
            }
            std::vector<uint8_t> direct_target_state;
            {
                llama_context reference(*model, cparams, target_placement);
                pp_decode_tokens(reference, {tokens.begin(), tokens.begin() + 11});
                direct_target_state.resize(llama_state_seq_get_size(&reference, 0));
                llama_state_seq_get_data(&reference, direct_target_state.data(), direct_target_state.size(), 0);
            }
            llama_context ctx(*model, cparams, original);
            auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
            const auto id = ctx.migrate_layers_async_begin(interval.first, interval.second, reverse ? devices[0] : devices[1]);
            if (!ctx.enable_layer_migration_replay(id, 8)) {
                throw std::runtime_error("failed to enable interval replay");
            }
            pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.begin() + 11}, 6));
            std::vector<uint8_t> before(llama_state_seq_get_size(&ctx, 0));
            llama_state_seq_get_data(&ctx, before.data(), before.size(), 0);
            auto * replay_mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
            const auto reference_tensors = pp_layer_state_bytes(*replay_mem, interval.first, interval.second);
            if (!ctx.replay_layer_migration(id)) {
                throw std::runtime_error("interval replay failed");
            }
            std::vector<uint8_t> after(before.size());
            llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
            if (before != after) {
                throw std::runtime_error("interval replay modified active sequence");
            }
            if (!ctx.migrate_layers_async_commit(id)) {
                throw std::runtime_error("interval replay commit failed");
            }
            llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
            if (before != after) {
                fprintf(stderr, "replay direct-target match=%d source-target match=%d\n", after == direct_target_state, before == direct_target_state);
                const auto reconstructed = pp_layer_state_bytes(*replay_mem, interval.first, interval.second);
                for (size_t it = 0; it < reconstructed.size(); ++it) {
                    size_t count = 0;
                    size_t first_diff = 0;
                    for (size_t j = 0; j < reconstructed[it].size(); ++j) {
                        if (reference_tensors[it][j] != reconstructed[it][j]) {
                            if (count == 0) { first_diff = j; }
                            ++count;
                        }
                    }
                    fprintf(stderr, "replay tensor %zu: differences=%zu first=%zu size=%zu\n", it, count, first_diff, reconstructed[it].size());
                }
                throw std::runtime_error("interval replay reconstructed different state");
            }
            pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 11, tokens.end()}, 11));
            if (result != expected) {
                throw std::runtime_error("interval replay changed logits");
            }
            const auto & stats = ctx.get_last_layer_migration_stats();
            if (stats.replay_tokens != 5 || stats.replay_layer_tokens != size_t(5 * (interval.second - interval.first))) {
                throw std::runtime_error("wrong replay interval accounting");
            }
            printf("PP interval replay [%d,%d) reverse=%d: state exact, active isolated, logits exact, tokens=%zu, hidden=%zu bytes, replay=%.3f ms\n",
                    interval.first, interval.second, reverse, stats.replay_tokens, stats.journal_bytes, stats.replay_us / 1000.0);
        }
    }
    for (const int failure : {0, 1, 2, 3, 4}) {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto id = ctx.migrate_layers_async_begin(0, 2, devices[1]);
        if (!ctx.enable_layer_migration_replay(id, failure == 0 ? 1 : 8)) {
            throw std::runtime_error("failed to enable replay failure test");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.begin() + 8}, 6));
        std::vector<uint8_t> before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, before.data(), before.size(), 0);
        if (failure == 0) {
            if (ctx.replay_layer_migration(id) || ctx.migrate_layers_async_commit(id)) {
                throw std::runtime_error("overflowed journal was accepted");
            }
        } else if (failure == 1 || failure == 4) {
            bool failed = false;
            try {
                ctx.replay_layer_migration(id, failure == 1 ? 1 : 0);
            } catch (const std::runtime_error &) {
                failed = true;
            }
            if (!failed || ctx.migrate_layers_async_commit(id)) {
                throw std::runtime_error("failed replay was accepted");
            }
        } else if (failure == 2) {
            ctx.migrate_layers_async_cancel(id);
        } else {
            if (!ctx.replay_layer_migration(id)) {
                throw std::runtime_error("replay for rollback failed");
            }
            bool failed = false;
            try {
                ctx.migrate_layers_async_commit(id, true);
            } catch (const std::runtime_error &) {
                failed = true;
            }
            if (!failed) {
                throw std::runtime_error("replay switch failure was ignored");
            }
        }
        std::vector<uint8_t> after(before.size());
        llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
        if (before != after || ctx.get_layer_device(0) != original[0]) {
            throw std::runtime_error("replay failure changed active state or placement");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 8, tokens.end()}, 8));
        if (result != expected) {
            throw std::runtime_error("replay failure changed continued logits");
        }
        const auto & stats = ctx.get_last_layer_migration_stats();
        printf("PP replay failure %d: active state, placement, continued logits exact; discarded checkpoint=%zu hidden=%zu bytes, replayed=%zu tokens\n",
                failure, stats.kv_bytes + stats.recurrent_bytes, stats.journal_bytes, stats.replay_tokens);
    }
    for (const bool reset_epoch : {false, true}) {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto actual = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto id = ctx.migrate_layers_async_begin(0, 2, devices[1]);
        if (reset_epoch) {
            if (!ctx.enable_layer_migration_replay(id, 8)) {
                throw std::runtime_error("epoch test could not enable replay");
            }
            llama_memory_clear(ctx.get_memory(), true);
            actual = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
            if (ctx.replay_layer_migration(id)) {
                throw std::runtime_error("old epoch replay was accepted");
            }
        } else {
            pp_append(actual, pp_decode_tokens(ctx, {tokens[6]}, 6));
            if (ctx.enable_layer_migration_replay(id, 8)) {
                throw std::runtime_error("missing hidden prefix was accepted");
            }
        }
        if (ctx.migrate_layers_async_commit(id)) {
            throw std::runtime_error("invalid replay request was committed");
        }
        const int pos = reset_epoch ? 6 : 7;
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + pos, tokens.end()}, pos));
        if (actual != expected) {
            throw std::runtime_error("rejected replay changed continued logits");
        }
        printf("PP replay %s: rejected, continued logits exact\n", reset_epoch ? "old sequence epoch" : "missing hidden prefix");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto actual = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto id = ctx.migrate_layers_async_begin(0, 2, devices[1]);
        if (!ctx.enable_layer_migration_replay(id, 1)) {
            throw std::runtime_error("incremental replay enable failed");
        }
        for (int pos = 6; pos < 11; ++pos) {
            pp_append(actual, pp_decode_tokens(ctx, {tokens[pos]}, pos));
            if (!ctx.replay_layer_migration(id)) {
                throw std::runtime_error("incremental replay failed to release journal entries");
            }
        }
        if (!ctx.migrate_layers_async_commit(id)) {
            throw std::runtime_error("incremental replay commit failed");
        }
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + 11, tokens.end()}, 11));
        if (actual != expected) {
            throw std::runtime_error("incremental replay changed logits");
        }
        printf("PP incremental replay: 5 segments with capacity 1, continued logits exact\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        const auto decode_chunk = [&](llama_context & ctx) {
            auto batch = llama_batch_init(2, 0, 1);
            batch.n_tokens = 2;
            for (int i = 0; i < 2; ++i) {
                batch.token[i] = tokens[6 + i];
                batch.pos[i] = 6 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            const int status = llama_decode(&ctx, batch);
            llama_batch_free(batch);
            if (status != 0) {
                throw std::runtime_error("chunk rejection foreground decode failed");
            }
        };
        std::vector<float> reference;
        {
            llama_context ctx(*model, cparams, original);
            pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
            decode_chunk(ctx);
            reference = pp_decode_tokens(ctx, {tokens.begin() + 8, tokens.end()}, 8);
        }
        llama_context ctx(*model, cparams, original);
        pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto id = ctx.migrate_layers_async_begin(0, 2, devices[1]);
        if (!ctx.enable_layer_migration_replay(id, 8)) {
            throw std::runtime_error("chunk rejection replay enable failed");
        }
        decode_chunk(ctx);
        if (ctx.replay_layer_migration(id) || ctx.migrate_layers_async_commit(id)) {
            throw std::runtime_error("chunk-generated hidden was accepted by token replay");
        }
        if (pp_decode_tokens(ctx, {tokens.begin() + 8, tokens.end()}, 8) != reference) {
            throw std::runtime_error("chunk rejection changed continued logits");
        }
        printf("PP replay chunk input: rejected, chunk foreground and continuation preserved\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto stale_id = ctx.migrate_layers_async_begin(1, 2, devices[1]);
        pp_append(result, pp_decode_tokens(ctx, {tokens[6]}, 6));
        if (ctx.migrate_layers_async_commit(stale_id) || ctx.get_layer_device(1) != devices[0]) {
            throw std::runtime_error("stale async migration was committed");
        }
        const auto cancel_id = ctx.migrate_layers_async_begin(1, 2, devices[1]);
        ctx.migrate_layers_async_cancel(cancel_id);
        if (ctx.get_layer_device(1) != devices[0]) {
            throw std::runtime_error("async migration cancel changed active placement");
        }
        pp_append(result, pp_decode_tokens(ctx, {tokens.begin() + 7, tokens.end()}, 7));
        if (result != expected) {
            throw std::runtime_error("stale or cancelled async migration changed logits");
        }
        printf("PP async migration: stale frontier and cancellation verified\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        const auto request_id = ctx.migrate_layers_async_begin(1, 2, devices[1]);
        llama_memory_seq_add(ctx.get_memory(), 0, 0, -1, 1);
        if (ctx.migrate_layers_async_commit(request_id) || ctx.get_layer_device(1) != devices[0]) {
            throw std::runtime_error("async migration committed after append-only state was invalidated");
        }
        printf("PP async migration: append-only invalidation verified\n");
    }
    {
        const std::vector<ggml_backend_dev_t> uniform(4, devices[0]);
        llama_context ctx(*model, cparams, uniform);
        auto result = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 6});
        ctx.migrate_layers_sync(3, 4, devices[1]);
        auto suffix = pp_decode_tokens(ctx, {tokens.begin() + 6, tokens.end()}, 6);
        result.insert(result.end(), suffix.begin(), suffix.end());
        if (result != expected) {
            throw std::runtime_error("migration from uniform placement changed logits");
        }
        printf("PP synchronous migration created a validated stage boundary\n");
    }
    {
        const std::vector<ggml_backend_dev_t> original = {devices[0], devices[0], devices[1], devices[1]};
        llama_context ctx(*model, cparams, original);
        llama_memory_seq_add(ctx.get_memory(), 0, 0, -1, 1);
        bool rejected = false;
        try {
            ctx.migrate_layers_sync(1, 2, devices[1]);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("experimental placement allowed migration after a state mutation");
        }
    }
    pp_replay_matrix(*model, devices, seed);
    for (const auto & entry : model->tensors_by_name) {
        if (entry.first.compare(0, 4, "blk.") != 0) {
            continue;
        }
        std::vector<uint8_t> zeros(ggml_nbytes(entry.second), 0);
        ggml_backend_tensor_set(entry.second, zeros.data(), 0, zeros.size());
    }
    if (run_placement() != expected) {
        throw std::runtime_error("replica graph still reads original layer weights");
    }
    {
        llama_context ctx(*model, cparams);
        if (pp_decode_tokens(ctx, tokens) == expected) {
            throw std::runtime_error("layer mutation did not exercise weight binding");
        }
    }
    printf("PP placement: independent weights and state verified\n");
    return 0;
}

static int test_qwen35moe_layer_migration(const std::string & model_path, size_t seed, bool low_vram, bool replay_acceptance, bool replay_costs) {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count() && devices.size() < 2; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        fprintf(stderr, "SKIP: Qwen35MoE PP model test requires two GPU devices\n");
        return 77;
    }

    std::vector<ggml_backend_dev_t> model_devices;
    if (low_vram) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            throw std::runtime_error("Qwen35MoE low-VRAM test requires a CPU backend");
        }
        model_devices.push_back(cpu);
    }
    model_devices.insert(model_devices.end(), devices.begin(), devices.end());
    model_devices.push_back(nullptr);
    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    if (low_vram) {
        tensor_split[0] = 82.0f;
        tensor_split[1] = 5.0f;
        tensor_split[2] = 13.0f;
    } else {
        tensor_split[0] = 1.0f;
        tensor_split[1] = 1.0f;
    }
    auto params = llama_model_default_params();
    params.devices = model_devices.data();
    params.tensor_split = tensor_split.data();
    params.n_gpu_layers = 999;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    params.load_mtp = false;

    const int64_t load_start_us = ggml_time_us();
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), params));
    const int64_t load_us = ggml_time_us() - load_start_us;
    if (!model) {
        throw std::runtime_error("could not load Qwen35MoE PP model");
    }
    if (model->arch != LLM_ARCH_QWEN35MOE) {
        throw std::runtime_error("--pp-model requires a Qwen35MoE model");
    }

    const int32_t n_layer = model->hparams.n_layer();
    int32_t boundary = -1;
    for (int32_t il = 1; il < n_layer; ++il) {
        if (model->dev_layer(il - 1) == devices[0] && model->dev_layer(il) == devices[1]) {
            if (boundary >= 0) {
                throw std::runtime_error("Qwen35MoE PP test found multiple selected-device boundaries");
            }
            boundary = il;
        }
    }
    if (boundary < 2) {
        throw std::runtime_error("Qwen35MoE PP test could not find a movable boundary");
    }

    int32_t recurrent_layer = boundary - 1;
    while (recurrent_layer >= 0 && !model->hparams.is_recr(recurrent_layer)) {
        --recurrent_layer;
    }
    int32_t full_layer = boundary - 1;
    while (full_layer >= 0 && model->hparams.is_recr(full_layer)) {
        --full_layer;
    }
    if (recurrent_layer != boundary - 1 || full_layer != boundary - 2) {
        throw std::runtime_error("Qwen35MoE PP test requires adjacent full and recurrent layers at the PP boundary");
    }

    std::vector<int64_t> replica_us;
    for (const int32_t il : {full_layer, recurrent_layer}) {
        const int64_t start_us = ggml_time_us();
        model->prepare_layer_replica(il, devices[1]);
        replica_us.push_back(ggml_time_us() - start_us);
        const auto & replica = model->layer_on_device(il, devices[1]);
        const std::string prefix = "blk." + std::to_string(il) + ".";
        size_t n_checked = 0;
        bool checked_experts = false;
        for (const auto & entry : model->tensors_by_name) {
            if (entry.first.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            auto * original = entry.second;
            auto * copied = model->layer_tensor_on_device(il, devices[1], original);
            if (copied == original || copied->type != original->type ||
                    !ggml_are_same_shape(original, copied) || !ggml_are_same_stride(original, copied) ||
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(copied->buffer)) != devices[1]) {
                throw std::runtime_error("Qwen35MoE replica has the wrong storage, layout, or device");
            }
            std::vector<uint8_t> expected(ggml_nbytes(original));
            std::vector<uint8_t> actual(expected.size());
            ggml_backend_tensor_get(original, expected.data(), 0, expected.size());
            ggml_backend_tensor_get(copied, actual.data(), 0, actual.size());
            if (actual != expected) {
                throw std::runtime_error("Qwen35MoE replica changed weight bytes");
            }
            if (original == model->layers[il].ffn_down_exps && replica.ffn_down_exps == copied) {
                checked_experts = true;
            }
            ++n_checked;
        }
        if (n_checked == 0 || !checked_experts) {
            throw std::runtime_error("Qwen35MoE replica did not cover expert weights");
        }
        printf("Qwen35MoE replica layer %d (%s): %zu tensors, preparation %.3f ms\n", il,
                model->hparams.is_recr(il) ? "linear" : "full", n_checked, replica_us.back()/1000.0);
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    cparams.n_seq_max = 1;
    cparams.n_rs_seq = 0;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.offload_kqv = true;
    cparams.no_perf = false;

    const auto tokens = get_tokens(10, model->vocab.n_tokens(), seed);
    std::vector<float> expected;
    std::vector<int64_t> baseline_token_us;
    {
        llama_context ctx(*model, cparams);
        expected = pp_decode_tokens(ctx, tokens, 0, &baseline_token_us);
    }

    struct migration_case {
        const char * name;
        int32_t initial_boundary;
        int32_t first;
        int32_t end;
    };
    const migration_case cases[] = {
        {"linear", boundary, recurrent_layer, boundary},
        {"full", full_layer + 1, full_layer, full_layer + 1},
        {"mixed", boundary, full_layer, boundary},
    };

    if (replay_costs) {
        cparams.n_ctx = 512;
        const int checkpoint = 6;
        const int gap = 100;
        const int frontier = checkpoint + gap;
        const auto input = get_tokens(frontier + 5, model->vocab.n_tokens(), seed);
        for (const auto & tc : cases) {
            std::vector<ggml_backend_dev_t> placement;
            for (int il = 0; il < n_layer; ++il) {
                placement.push_back(model->dev_layer(il));
            }
            if (tc.initial_boundary != boundary) {
                placement[recurrent_layer] = devices[1];
            }
            std::vector<float> reference;
            std::vector<int64_t> baseline_times;
            {
                llama_context ctx(*model, cparams, placement);
                reference = pp_decode_tokens(ctx, input, 0, &baseline_times);
            }
            for (const bool shadow : {false, true}) {
                llama_context ctx(*model, cparams, placement);
                auto actual = pp_decode_tokens(ctx, {input.begin(), input.begin() + checkpoint});
                const auto id = ctx.migrate_layers_async_begin(tc.first, tc.end, devices[1]);
                if (shadow ? !ctx.enable_layer_migration_shadow(id) : !ctx.enable_layer_migration_replay(id, gap)) {
                    throw std::runtime_error("cost comparison enable failed");
                }
                std::vector<int64_t> foreground_times;
                std::vector<int64_t> shadow_times;
                int64_t previous_shadow_us = 0;
                for (int pos = checkpoint; pos < frontier; ++pos) {
                    pp_append(actual, pp_decode_tokens(ctx, {input[pos]}, pos, &foreground_times));
                    const auto shadow_us = ctx.get_last_layer_migration_stats().shadow_us;
                    shadow_times.push_back(shadow_us - previous_shadow_us);
                    previous_shadow_us = shadow_us;
                }
                std::vector<uint8_t> state(llama_state_seq_get_size(&ctx, 0));
                llama_state_seq_get_data(&ctx, state.data(), state.size(), 0);
                std::vector<int64_t> copy_times;
                size_t copy_payload = 0;
                if (!shadow) {
                    copy_payload = pp_benchmark_valid_copy(ctx, tc.first, tc.end, devices[1], checkpoint, frontier, copy_times);
                    if (!ctx.replay_layer_migration(id)) {
                        throw std::runtime_error("cost comparison replay failed");
                    }
                }
                std::vector<uint8_t> after(state.size());
                llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
                if (state != after || !ctx.migrate_layers_async_commit(id)) {
                    throw std::runtime_error("cost comparison isolation or commit failed");
                }
                llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
                if (state != after) {
                    throw std::runtime_error("cost comparison candidate state differs");
                }
                std::vector<int64_t> continuation_times;
                pp_append(actual, pp_decode_tokens(ctx, {input.begin() + frontier, input.end()}, frontier, &continuation_times));
                pp_compare_logits(reference, actual, 0, 0);
                const auto & stats = ctx.get_last_layer_migration_stats();
                const auto foreground = pp_summarize(foreground_times);
                const auto baseline = pp_summarize({baseline_times.begin() + checkpoint, baseline_times.begin() + frontier});
                if (!shadow) {
                    const auto copy = pp_summarize(copy_times);
                    const auto replay = pp_summarize(stats.replay_step_us);
                    printf("REPLAY COST %s P=%d D=%d: state/logits exact, valid-state peer-copy samples=%zu p50/p95=%.3f/%.3f ms payload=%zu bytes; replay steps=%zu p50/p95=%.3f/%.3f ms total=%.3f ms hidden=%zu bytes; checkpoint allocated/valid=%zu/%zu bytes; journal compute peak=%zu/%zu bytes; first decode=%.3f ms\n",
                            tc.name, checkpoint, gap, copy_times.size(), copy.median / 1000.0, copy.p95 / 1000.0, copy_payload,
                            stats.replay_step_us.size(), replay.median / 1000.0, replay.p95 / 1000.0, stats.replay_us / 1000.0, stats.journal_bytes,
                            stats.kv_bytes + stats.recurrent_bytes, stats.kv_valid_bytes + stats.recurrent_bytes,
                            stats.journal_peak_bytes, stats.replay_compute_bytes, continuation_times[0] / 1000.0);
                } else {
                    const auto timing = pp_summarize(shadow_times);
                    printf("SHADOW COST %s P=%d D=%d: state/logits exact, full-model shadow samples=%zu p50/p95=%.3f/%.3f ms total=%.3f ms shared-state host roundtrip=%zu bytes; first decode=%.3f ms\n",
                            tc.name, checkpoint, gap, shadow_times.size(), timing.median / 1000.0, timing.p95 / 1000.0,
                            stats.shadow_us / 1000.0, stats.shadow_shared_bytes, continuation_times[0] / 1000.0);
                }
                printf("FOREGROUND COST %s mode=%s samples=%zu baseline p50/p95=%.3f/%.3f ms maintained p50/p95=%.3f/%.3f ms\n",
                        tc.name, shadow ? "shadow" : "journal", foreground_times.size(), baseline.median / 1000.0,
                        baseline.p95 / 1000.0, foreground.median / 1000.0, foreground.p95 / 1000.0);
                const auto reverse_id = ctx.migrate_layers_async_begin(tc.first, tc.end, devices[0]);
                if (shadow ? !ctx.enable_layer_migration_shadow(reverse_id) : !ctx.enable_layer_migration_replay(reverse_id, 1)) {
                    throw std::runtime_error("token gap measurement enable failed");
                }
                pp_decode_tokens(ctx, {input[0]}, input.size());
                const int64_t gap_start = ggml_time_us();
                if ((!shadow && !ctx.replay_layer_migration(reverse_id)) || !ctx.migrate_layers_async_commit(reverse_id)) {
                    throw std::runtime_error("token gap measurement migration failed");
                }
                pp_decode_tokens(ctx, {input[1]}, input.size() + 1);
                const int64_t token_gap_us = ggml_time_us() - gap_start;
                printf("TOKEN GAP %s mode=%s reverse=1 D=1: %.3f ms, includes catchup/commit/next-decode, excludes checkpoint preparation; no state serialization inside timing\n",
                        tc.name, shadow ? "shadow" : "replay", token_gap_us / 1000.0);
                fflush(stdout);
            }
        }
        printf("REPLAY COST acceptance: 3 intervals x valid copy / interval replay / full shadow passed\n");
        return 0;
    }

    if (replay_acceptance) {
        cparams.n_ctx = 512;
        const std::vector<std::string> prompts = {
            "Explain how a CPU cache differs from main memory.",
            "Write a Python function that computes the greatest common divisor.",
            "Describe how to measure latency in a distributed system.",
            "Explain photosynthesis to a secondary school student.",
            "Give an example of a database transaction and its rollback.",
            "Explain why floating point addition is not associative.",
            "Compare breadth first search with depth first search.",
            "Write a short story about a robot exploring a library.",
            "Describe how a solar panel generates electricity.",
            "Explain what happens during a network connection timeout.",
            "List the steps for debugging an intermittent memory leak.",
            "Explain the difference between a mutex and a semaphore.",
            "Write a Python function that merges two sorted lists.",
            "Describe the water cycle in a few paragraphs.",
            "Explain how a hash table handles collisions.",
            "Compare synchronous and asynchronous file operations.",
            "Describe a practical use of a finite state machine.",
            "Explain how to test a numerical algorithm for correctness.",
            "Write a brief guide to reading a performance trace.",
            "Explain checkpointing and recovery in distributed computation.",
        };
        size_t generated_total = 0;
        std::vector<int64_t> journal_token_us;
        std::vector<int64_t> baseline_journal_token_us;
        for (size_t sample = 0; sample < prompts.size(); ++sample) {
            const auto & tc = cases[sample % 3];
            const bool reverse = (sample / 3) % 2 != 0;
            std::vector<ggml_backend_dev_t> placement;
            for (int il = 0; il < n_layer; ++il) {
                placement.push_back(model->dev_layer(il));
            }
            if (tc.initial_boundary != boundary) {
                placement[recurrent_layer] = devices[1];
            }
            if (reverse) {
                for (int il = tc.first; il < tc.end; ++il) {
                    placement[il] = devices[1];
                }
            }
            std::string text = prompts[sample];
            if (sample == prompts.size() - 1) {
                for (int repeat = 0; repeat < 8; ++repeat) {
                    text += " A worker stores a checkpoint, continues processing requests, and later transfers responsibility to a second worker.";
                }
            }
            const auto prompt = common_tokenize(&model->vocab, text, true);
            const size_t checkpoint = sample == prompts.size() - 1 ? 1 : prompt.size() / 2;
            if (prompt.size() + 128 > cparams.n_ctx || checkpoint == 0) {
                throw std::runtime_error("invalid real replay prompt length");
            }
            const bool stochastic = sample % 2 != 0;
            const auto make_sampler = [&]() {
                llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                if (stochastic) {
                    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.8f));
                    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed + sample));
                } else {
                    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
                }
                return sampler;
            };
            std::vector<llama_token> generated;
            std::vector<std::vector<float>> reference_logits;
            std::vector<int64_t> baseline_times;
            {
                llama_context baseline(*model, cparams, placement);
                std::vector<float> last_logits;
                for (size_t pos = 0; pos < prompt.size(); ++pos) {
                    last_logits = pp_decode_tokens(baseline, {prompt[pos]}, pos, &baseline_times);
                }
                auto sampler = make_sampler();
                for (size_t step = 0; step < 128; ++step) {
                    reference_logits.push_back(std::move(last_logits));
                    const auto token = llama_sampler_sample(sampler.get(), &baseline, -1);
                    generated.push_back(token);
                    if (llama_vocab_is_eog(&model->vocab, token)) {
                        break;
                    }
                    last_logits = pp_decode_tokens(baseline, {token}, prompt.size() + step);
                }
            }
            llama_context ctx(*model, cparams, placement);
            pp_decode_tokens(ctx, {prompt.begin(), prompt.begin() + checkpoint});
            const auto id = ctx.migrate_layers_async_begin(tc.first, tc.end, reverse ? devices[0] : devices[1]);
            if (!ctx.enable_layer_migration_replay(id, prompt.size() - checkpoint)) {
                throw std::runtime_error("real acceptance could not enable replay");
            }
            std::vector<float> last_logits;
            for (size_t pos = checkpoint; pos < prompt.size(); ++pos) {
                last_logits = pp_decode_tokens(ctx, {prompt[pos]}, pos, &journal_token_us);
                baseline_journal_token_us.push_back(baseline_times[pos]);
            }
            std::vector<uint8_t> state(llama_state_seq_get_size(&ctx, 0));
            llama_state_seq_get_data(&ctx, state.data(), state.size(), 0);
            const int64_t start = ggml_time_us();
            if (!ctx.replay_layer_migration(id)) {
                throw std::runtime_error("real acceptance replay failed");
            }
            std::vector<uint8_t> after(state.size());
            llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
            if (state != after || !ctx.migrate_layers_async_commit(id)) {
                throw std::runtime_error("real acceptance active isolation or commit failed");
            }
            const int64_t catchup_commit_us = ggml_time_us() - start;
            if (llama_state_seq_get_size(&ctx, 0) != state.size()) {
                throw std::runtime_error("real acceptance metadata size changed");
            }
            llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
            if (state != after) {
                throw std::runtime_error("real acceptance reconstructed state differs");
            }
            auto sampler = make_sampler();
            double max_abs = 0;
            double max_rel = 0;
            std::vector<int64_t> continuation_us;
            for (size_t step = 0; step < generated.size(); ++step) {
                const auto diff = pp_compare_logits(reference_logits[step], last_logits, 0, 0);
                max_abs = std::max(max_abs, diff.first);
                max_rel = std::max(max_rel, diff.second);
                const auto token = llama_sampler_sample(sampler.get(), &ctx, -1);
                if (token != generated[step]) {
                    throw std::runtime_error("real acceptance sampled token differs");
                }
                if (llama_vocab_is_eog(&model->vocab, token)) {
                    break;
                }
                last_logits = pp_decode_tokens(ctx, {token}, prompt.size() + step, &continuation_us);
            }
            generated_total += generated.size();
            const auto & stats = ctx.get_last_layer_migration_stats();
            printf("REAL REPLAY sample=%zu mode=%s interval=%s[%d,%d) reverse=%d P=%zu D=%zu generated=%zu eos=%d: state exact, logits abs/rel=%.9g/%.9g, tokens exact, replay=%.3f ms, catchup+commit+validation=%.3f ms, first_decode=%.3f ms, hidden=%zu bytes, checkpoint=%zu bytes, journal_payload_peak=%zu bytes, compute_peak=%zu bytes\n",
                    sample, stochastic ? "seeded" : "greedy", tc.name, tc.first, tc.end, reverse, checkpoint,
                    prompt.size() - checkpoint, generated.size(), llama_vocab_is_eog(&model->vocab, generated.back()),
                    max_abs, max_rel, stats.replay_us / 1000.0, catchup_commit_us / 1000.0,
                    continuation_us.empty() ? 0.0 : continuation_us[0] / 1000.0, stats.journal_bytes,
                    stats.kv_bytes + stats.recurrent_bytes, stats.journal_peak_bytes, stats.replay_compute_bytes);
            fflush(stdout);
        }
        const auto journal_time = pp_summarize(journal_token_us);
        const auto baseline_time = pp_summarize(baseline_journal_token_us);
        printf("REAL REPLAY acceptance: 20 workloads passed, generated=%zu, 10 greedy + 10 seeded, state and logits exact; journal maintenance samples=%zu baseline p50/p95=%.3f/%.3f ms, journal p50/p95=%.3f/%.3f ms\n",
                generated_total, journal_token_us.size(), baseline_time.median / 1000.0, baseline_time.p95 / 1000.0,
                journal_time.median / 1000.0, journal_time.p95 / 1000.0);
        return 0;
    }

    printf("Qwen35MoE PP model: %d layers, boundary %d, load %.3f ms, devices %s -> %s, CPU prefix %s\n",
            n_layer, boundary, load_us/1000.0, ggml_backend_dev_name(devices[0]), ggml_backend_dev_name(devices[1]),
            low_vram ? "enabled" : "disabled");
    for (const auto & tc : cases) {
        std::vector<ggml_backend_dev_t> placement;
        for (int32_t il = 0; il < n_layer; ++il) {
            placement.push_back(model->dev_layer(il));
        }
        if (tc.initial_boundary != boundary) {
            placement[recurrent_layer] = devices[1];
        }

        const int64_t context_start_us = ggml_time_us();
        llama_context ctx(*model, cparams, placement);
        const int64_t context_us = ggml_time_us() - context_start_us;
        auto * memory = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
        if (!memory) {
            throw std::runtime_error("Qwen35MoE PP test requires hybrid memory");
        }

        std::vector<float> actual;
        std::vector<int64_t> token_us;
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 5}, 0, &token_us));

        const auto active_view = memory->get_active_view();
        const auto candidate_view = memory->clone_active_view();
        const uint64_t active_frontier = active_view->frontier;
        const uint32_t active_attn_head = active_view->attn->v_heads[0];
        const uint32_t active_recurrent_head = active_view->recurrent->head;
        candidate_view->attn->v_heads[0] ^= 1;
        candidate_view->recurrent->head ^= 1;
        candidate_view->frontier += 1;
        if (active_view->frontier != active_frontier || active_view->attn->v_heads[0] != active_attn_head ||
                active_view->recurrent->head != active_recurrent_head) {
            throw std::runtime_error("Qwen35MoE candidate metadata changed the active view");
        }
        bool stale_rejected = false;
        try {
            memory->exchange_active_view(candidate_view, active_frontier);
        } catch (const std::runtime_error &) {
            stale_rejected = true;
        }
        if (!stale_rejected || active_view->frontier != active_frontier) {
            throw std::runtime_error("Qwen35MoE stale candidate frontier was not rejected");
        }

        std::vector<uint8_t> seq_state_before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, seq_state_before.data(), seq_state_before.size(), 0);
        const auto layer_state_before = pp_layer_state_bytes(*memory, tc.first, tc.end);
        const auto source_addresses = pp_layer_state_addresses(*memory, tc.first, tc.end);
        size_t layer_state_size = 0;
        for (const auto & tensor_bytes : layer_state_before) {
            layer_state_size += tensor_bytes.size();
        }
        const auto memory_before = memory->memory_breakdown();

        std::vector<int64_t> forward_us;
        std::vector<int64_t> reverse_us;
        std::vector<const void *> destination_addresses;
        for (int round = 0; round < 5; ++round) {
            int64_t start_us = ggml_time_us();
            ctx.migrate_layers_sync(tc.first, tc.end, devices[1]);
            forward_us.push_back(ggml_time_us() - start_us);
            const auto current_destination = pp_layer_state_addresses(*memory, tc.first, tc.end);
            if (round == 0) {
                destination_addresses = current_destination;
            } else if (current_destination != destination_addresses) {
                throw std::runtime_error("Qwen35MoE destination state bank address changed");
            }
            start_us = ggml_time_us();
            ctx.migrate_layers_sync(tc.first, tc.end, devices[0]);
            reverse_us.push_back(ggml_time_us() - start_us);
            if (pp_layer_state_addresses(*memory, tc.first, tc.end) != source_addresses) {
                throw std::runtime_error("Qwen35MoE source state bank address changed");
            }
        }
        std::vector<uint8_t> seq_state_roundtrip(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, seq_state_roundtrip.data(), seq_state_roundtrip.size(), 0);
        if (seq_state_roundtrip != seq_state_before ||
                pp_layer_state_bytes(*memory, tc.first, tc.end) != layer_state_before) {
            throw std::runtime_error("Qwen35MoE repeated migration changed state bytes");
        }

        ggml_backend_cuda_graph_stats graph_before = {};
        const bool has_graph_stats = pp_cuda_graph_stats(ctx, graph_before);
        const int64_t forward_start_us = ggml_time_us();
        ctx.migrate_layers_sync(tc.first, tc.end, devices[1]);
        const int64_t final_forward_us = ggml_time_us() - forward_start_us;
        const auto forward_stats = ctx.get_last_layer_migration_stats();
        const auto forward_reserve_stats = ctx.get_last_sched_reserve_stats();
        ggml_backend_cuda_graph_stats graph_after_migration = {};
        pp_cuda_graph_stats(ctx, graph_after_migration);
        const auto memory_after = memory->memory_breakdown();
        if (forward_stats.kv_bytes + forward_stats.recurrent_bytes != layer_state_size ||
                forward_stats.kv_valid_bytes > forward_stats.kv_bytes ||
                forward_stats.total_us <= 0 || forward_stats.copy_us > forward_stats.prepare_us ||
                forward_reserve_stats.graph_reserve_calls < 3) {
            throw std::runtime_error("Qwen35MoE migration reported invalid communication statistics");
        }
        if (pp_layer_state_bytes(*memory, tc.first, tc.end) != layer_state_before) {
            throw std::runtime_error("Qwen35MoE forward migration changed layer state bytes");
        }
        for (int32_t il = tc.first; il < tc.end; ++il) {
            if (ctx.get_layer_device(il) != devices[1]) {
                throw std::runtime_error("Qwen35MoE forward migration did not update layer ownership");
            }
            for (auto * tensor : memory->layer_state_tensors(il)) {
                if (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != devices[1]) {
                    throw std::runtime_error("Qwen35MoE forward migration put state on the wrong device");
                }
            }
        }
        if (memory_after != memory_before ||
                pp_layer_state_addresses(*memory, tc.first, tc.end) != destination_addresses) {
            throw std::runtime_error("Qwen35MoE forward migration did not reuse fixed state banks");
        }

        pp_append(actual, pp_decode_tokens(ctx, {tokens[5]}, 5, &token_us));
        ggml_backend_cuda_graph_stats graph_after_first = {};
        pp_cuda_graph_stats(ctx, graph_after_first);
        const int64_t trigger_to_next_token_us = final_forward_us + token_us[5];
        pp_append(actual, pp_decode_tokens(ctx, {tokens[6]}, 6, &token_us));
        ggml_backend_cuda_graph_stats graph_after_stable = {};
        pp_cuda_graph_stats(ctx, graph_after_stable);

        const auto reverse_state_before = pp_layer_state_bytes(*memory, tc.first, tc.end);
        const int64_t reverse_start_us = ggml_time_us();
        ctx.migrate_layers_sync(tc.first, tc.end, devices[0]);
        const int64_t final_reverse_us = ggml_time_us() - reverse_start_us;
        const auto reverse_stats = ctx.get_last_layer_migration_stats();
        const auto memory_after_reverse = memory->memory_breakdown();
        if (reverse_stats.kv_bytes != forward_stats.kv_bytes ||
                reverse_stats.recurrent_bytes != forward_stats.recurrent_bytes) {
            throw std::runtime_error("Qwen35MoE reverse migration changed communication bytes");
        }
        if (pp_layer_state_bytes(*memory, tc.first, tc.end) != reverse_state_before) {
            throw std::runtime_error("Qwen35MoE reverse migration changed layer state bytes");
        }
        for (int32_t il = tc.first; il < tc.end; ++il) {
            if (ctx.get_layer_device(il) != devices[0]) {
                throw std::runtime_error("Qwen35MoE reverse migration did not restore layer ownership");
            }
            for (auto * tensor : memory->layer_state_tensors(il)) {
                if (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != devices[0]) {
                    throw std::runtime_error("Qwen35MoE reverse migration put state on the wrong device");
                }
            }
        }
        if (memory_after_reverse != memory_after ||
                pp_layer_state_addresses(*memory, tc.first, tc.end) != source_addresses) {
            throw std::runtime_error("Qwen35MoE reverse migration did not reuse fixed state banks");
        }
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + 7, tokens.end()}, 7, &token_us));

        const auto diff = pp_compare_logits(expected, actual);
        const auto forward = pp_summarize(forward_us);
        const auto reverse = pp_summarize(reverse_us);
        printf("Qwen35MoE migration %s [%d, %d): context %.3f ms, forward min/median/p95 %.3f/%.3f/%.3f ms, reverse %.3f/%.3f/%.3f ms\n",
                tc.name, tc.first, tc.end, context_us/1000.0,
                forward.minimum/1000.0, forward.median/1000.0, forward.p95/1000.0,
                reverse.minimum/1000.0, reverse.median/1000.0, reverse.p95/1000.0);
        printf("Qwen35MoE migration %s: final forward/reverse %.3f/%.3f ms, token pre/first/stable %.3f/%.3f/%.3f ms, baseline first/stable %.3f/%.3f ms, max abs/rel %.9g/%.9g\n",
                tc.name, final_forward_us/1000.0, final_reverse_us/1000.0,
                token_us[4]/1000.0, token_us[5]/1000.0, token_us[6]/1000.0,
                baseline_token_us[5]/1000.0, baseline_token_us[6]/1000.0, diff.first, diff.second);
        printf("Qwen35MoE migration %s communication: KV capacity %zu bytes (%.3f MiB), KV valid %zu bytes (%.3f MiB), recurrent %zu bytes (%.3f MiB), total copied %zu bytes (%.3f MiB)\n",
                tc.name, forward_stats.kv_bytes, forward_stats.kv_bytes/(1024.0*1024.0),
                forward_stats.kv_valid_bytes, forward_stats.kv_valid_bytes/(1024.0*1024.0),
                forward_stats.recurrent_bytes, forward_stats.recurrent_bytes/(1024.0*1024.0),
                forward_stats.kv_bytes + forward_stats.recurrent_bytes,
                (forward_stats.kv_bytes + forward_stats.recurrent_bytes)/(1024.0*1024.0));
        printf("Qwen35MoE migration %s blocking before token position 5: sync/copy/other-prepare/switch/reserve/total %.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms, trigger-to-next-token %.3f ms\n",
                tc.name, forward_stats.synchronize_us/1000.0, forward_stats.copy_us/1000.0,
                (forward_stats.prepare_us - forward_stats.copy_us)/1000.0, forward_stats.switch_us/1000.0,
                forward_stats.reserve_us/1000.0, forward_stats.total_us/1000.0,
                trigger_to_next_token_us/1000.0);
        printf("Qwen35MoE migration %s reserve: sync/setup/fused/input/build/validation/backend/total %.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms, graph calls %u, validation builds %u\n",
                tc.name, forward_reserve_stats.synchronize_us/1000.0, forward_reserve_stats.setup_us/1000.0,
                forward_reserve_stats.fused_probe_us/1000.0, forward_reserve_stats.graph_input_us/1000.0,
                forward_reserve_stats.graph_build_us/1000.0, forward_reserve_stats.validation_graph_build_us/1000.0,
                forward_reserve_stats.backend_plan_us/1000.0, forward_reserve_stats.total_us/1000.0,
                forward_reserve_stats.graph_reserve_calls, forward_reserve_stats.validation_graph_builds);
        if (has_graph_stats) {
            const auto migration_graph_delta = pp_cuda_graph_stats_delta(graph_after_migration, graph_before);
            const auto first_graph_delta = pp_cuda_graph_stats_delta(graph_after_first, graph_after_migration);
            const auto stable_graph_delta = pp_cuda_graph_stats_delta(graph_after_stable, graph_after_first);
            printf("Qwen35MoE migration %s CUDA Graph current: supported backends %" PRIu64 ", graphs/enabled/warm/executable %" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                    tc.name, graph_after_stable.graph_support, graph_after_stable.graph_count,
                    graph_after_stable.enabled_graph_count, graph_after_stable.warmup_complete_graph_count,
                    graph_after_stable.executable_graph_count);
            printf("Qwen35MoE migration %s CUDA Graph deltas migration/first/stable direct %" PRIu64 "/%" PRIu64 "/%" PRIu64 ", warmup %" PRIu64 "/%" PRIu64 "/%" PRIu64 ", capture %" PRIu64 "/%" PRIu64 "/%" PRIu64 ", update %" PRIu64 "/%" PRIu64 "/%" PRIu64 ", launch %" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                    tc.name, migration_graph_delta.direct_evaluations, first_graph_delta.direct_evaluations, stable_graph_delta.direct_evaluations,
                    migration_graph_delta.warmup_completions, first_graph_delta.warmup_completions, stable_graph_delta.warmup_completions,
                    migration_graph_delta.captures, first_graph_delta.captures, stable_graph_delta.captures,
                    migration_graph_delta.updates, first_graph_delta.updates, stable_graph_delta.updates,
                    migration_graph_delta.launches, first_graph_delta.launches, stable_graph_delta.launches);
        }
    }

    for (const auto & tc : cases) {
        std::vector<ggml_backend_dev_t> placement;
        for (int il = 0; il < n_layer; ++il) {
            placement.push_back(model->dev_layer(il));
        }
        if (tc.initial_boundary != boundary) {
            placement[recurrent_layer] = devices[1];
        }
        llama_context ctx(*model, cparams, placement);
        auto actual = pp_decode_tokens(ctx, {tokens.begin(), tokens.begin() + 5});
        const auto id = ctx.migrate_layers_async_begin(tc.first, tc.end, devices[1]);
        if (!ctx.enable_layer_migration_replay(id, 8)) {
            throw std::runtime_error("Qwen35MoE could not enable replay");
        }
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + 5, tokens.begin() + 7}, 5));
        std::vector<uint8_t> before(llama_state_seq_get_size(&ctx, 0));
        llama_state_seq_get_data(&ctx, before.data(), before.size(), 0);
        auto * replay_mem = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
        const auto reference_tensors = pp_layer_state_bytes(*replay_mem, tc.first, tc.end);
        if (!ctx.replay_layer_migration(id)) {
            throw std::runtime_error("Qwen35MoE replay failed");
        }
        std::vector<uint8_t> after(before.size());
        llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
        if (before != after) {
            throw std::runtime_error("Qwen35MoE replay changed active state");
        }
        if (!ctx.migrate_layers_async_commit(id)) {
            throw std::runtime_error("Qwen35MoE replay commit failed");
        }
        llama_state_seq_get_data(&ctx, after.data(), after.size(), 0);
        if (before != after) {
            const auto reconstructed = pp_layer_state_bytes(*replay_mem, tc.first, tc.end);
            size_t it = 0;
            for (int il = tc.first; il < tc.end; ++il) {
                for (auto * tensor : replay_mem->layer_state_tensors(il)) {
                    size_t count = 0;
                    double max_abs = 0;
                    for (size_t j = 0; j < reconstructed[it].size(); ++j) {
                        count += reference_tensors[it][j] != reconstructed[it][j];
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        for (size_t j = 0; j < reconstructed[it].size(); j += sizeof(float)) {
                            float a, b;
                            memcpy(&a, reference_tensors[it].data() + j, sizeof(a));
                            memcpy(&b, reconstructed[it].data() + j, sizeof(b));
                            max_abs = std::max(max_abs, double(std::abs(a - b)));
                        }
                    }
                    fprintf(stderr, "Qwen35MoE replay tensor layer=%d tensor=%zu type=%s differences=%zu max_abs=%.9g\n",
                            il, it, ggml_type_name(tensor->type), count, max_abs);
                    ++it;
                }
            }
            throw std::runtime_error("Qwen35MoE replay reconstructed different state");
        }
        pp_append(actual, pp_decode_tokens(ctx, {tokens.begin() + 7, tokens.end()}, 7));
        const auto diff = pp_compare_logits(expected, actual);
        const auto & stats = ctx.get_last_layer_migration_stats();
        printf("Qwen35MoE replay %s [%d,%d): active isolated, state exact, logits max abs/rel %.9g/%.9g, tokens=%zu, hidden=%zu bytes, checkpoint=%zu bytes, replay=%.3f ms\n",
                tc.name, tc.first, tc.end, diff.first, diff.second, stats.replay_tokens, stats.journal_bytes,
                stats.kv_bytes + stats.recurrent_bytes, stats.replay_us / 1000.0);
    }

    pp_layer_compute_probe probe = {};
    probe.first = full_layer;
    probe.end = boundary;
    auto profile_params = cparams;
    profile_params.cb_eval = pp_layer_compute_callback;
    profile_params.cb_eval_user_data = &probe;
    std::vector<ggml_backend_dev_t> profile_placement;
    for (int32_t il = 0; il < n_layer; ++il) {
        profile_placement.push_back(model->dev_layer(il));
    }
    llama_context profile_ctx(*model, profile_params, profile_placement);
    probe.begin_token();
    pp_decode_tokens(profile_ctx, {tokens[0]}, 0);

    std::vector<int64_t> source_before_us;
    std::vector<int64_t> source_after_us;
    std::vector<int64_t> destination_before_us;
    std::vector<int64_t> destination_after_us;
    std::vector<int64_t> moved_source_before_us;
    std::vector<int64_t> moved_destination_after_us;
    for (int32_t pos = 1; pos < 5; ++pos) {
        probe.begin_token();
        pp_decode_tokens(profile_ctx, {tokens[pos]}, pos);
        source_before_us.push_back(probe.get_layer_us(devices[0]));
        destination_before_us.push_back(probe.get_layer_us(devices[1]));
        moved_source_before_us.push_back(probe.get_moved_layer_us(devices[0]));
        if (probe.get_moved_layer_us(devices[1]) != 0) {
            throw std::runtime_error("Qwen35MoE compute probe found moved layers on the destination before migration");
        }
    }

    profile_ctx.migrate_layers_sync(full_layer, boundary, devices[1]);
    probe.begin_token();
    pp_decode_tokens(profile_ctx, {tokens[5]}, 5);
    for (int32_t pos = 6; pos < 10; ++pos) {
        probe.begin_token();
        pp_decode_tokens(profile_ctx, {tokens[pos]}, pos);
        source_after_us.push_back(probe.get_layer_us(devices[0]));
        destination_after_us.push_back(probe.get_layer_us(devices[1]));
        moved_destination_after_us.push_back(probe.get_moved_layer_us(devices[1]));
        if (probe.get_moved_layer_us(devices[0]) != 0) {
            throw std::runtime_error("Qwen35MoE compute probe found moved layers on the source after migration");
        }
    }

    const auto source_before = pp_summarize(source_before_us);
    const auto source_after = pp_summarize(source_after_us);
    const auto destination_before = pp_summarize(destination_before_us);
    const auto destination_after = pp_summarize(destination_after_us);
    const auto moved_source_before = pp_summarize(moved_source_before_us);
    const auto moved_destination_after = pp_summarize(moved_destination_after_us);
    if (source_before.median <= 0 || destination_before.median <= 0 || moved_source_before.median <= 0 ||
            moved_destination_after.median <= 0) {
        throw std::runtime_error("Qwen35MoE compute probe did not observe both PP devices");
    }
    const double source_reduction = 100.0*(source_before.median - source_after.median)/source_before.median;
    printf("Qwen35MoE compute probe [synchronous per-layer profiling]: source layer time median %.3f -> %.3f ms (%+.1f%% reduction), destination %.3f -> %.3f ms\n",
            source_before.median/1000.0, source_after.median/1000.0, source_reduction,
            destination_before.median/1000.0, destination_after.median/1000.0);
    printf("Qwen35MoE compute probe moved layers [%d, %d): source before %.3f ms, destination after %.3f ms, samples %zu\n",
            full_layer, boundary, moved_source_before.median/1000.0, moved_destination_after.median/1000.0,
            source_before_us.size());
    return 0;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();

    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    std::string out;
    bool pp_replicas = false;
    std::string pp_model;
    bool pp_low_vram = false;
    bool pp_replay_acceptance = false;
    bool pp_replay_costs = false;
    std::vector<std::string> pp_rpc_endpoints;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pp-rpc") == 0) {
            if (i + 2 >= argc || strcmp(argv[i + 1], argv[i + 2]) == 0) {
                fprintf(stderr, "--pp-rpc requires two distinct endpoints\n");
                return 1;
            }
            pp_replicas = true;
            pp_rpc_endpoints = {argv[i + 1], argv[i + 2]};
            i += 2;
            continue;
        }
        if (strcmp(argv[i], "--pp-replicas") == 0) {
            pp_replicas = true;
            continue;
        }
        if (strcmp(argv[i], "--pp-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--pp-model requires a model path\n");
                return 1;
            }
            pp_model = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--pp-low-vram") == 0) {
            pp_low_vram = true;
            continue;
        }
        if (strcmp(argv[i], "--pp-replay-acceptance") == 0) {
            pp_replay_acceptance = true;
            continue;
        }
        if (strcmp(argv[i], "--pp-replay-costs") == 0) {
            pp_replay_costs = true;
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    if ((pp_replay_acceptance || pp_replay_costs) && pp_model.empty()) {
        fprintf(stderr, "--pp-replay-acceptance and --pp-replay-costs require --pp-model\n");
        return 1;
    }
    if (pp_replay_acceptance && pp_replay_costs) {
        fprintf(stderr, "run replay generation and cost acceptance separately\n");
        return 1;
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!pp_model.empty()) {
            return test_qwen35moe_layer_migration(pp_model, seed, pp_low_vram, pp_replay_acceptance, pp_replay_costs);
        }
        if (pp_replicas) {
            return test_layer_replicas(seed, pp_rpc_endpoints);
        }
        if (!out.empty()) {
            return save_models(arch, seed, verbosity, out);
        }
        return test_backends(arch, seed, verbosity);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
