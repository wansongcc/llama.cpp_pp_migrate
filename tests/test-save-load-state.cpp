#include "arg.h"
#include "common.h"
#include "build-info.h"
#include "json.h"
#include "log.h"
#include "llama-cpp.h"
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
extern "C" {
#include "hash/sha256/sha256.h"
}

#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"
#endif

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 6/7: seq copy (scatter)
// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static bool test_seq_cp_scatter(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, int test_num, bool on_device) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOG("\n=== Test %d: seq copy (%s, scatter) ===\n", test_num, on_device ? "device" : "host");

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    auto decode_one = [&](llama_token tok, int pos, llama_seq_id seq) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tok, pos, { seq }, false);
        return llama_decode(ctx.get(), batch.get()) == 0;
    };

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!decode_one(tokens[0], 0, 0) ||
        !decode_one(tokens[1], 1, 0) ||
        !decode_one(tokens[0], 0, 1) ||
        !decode_one(tokens[1], 1, 1) ||
        !decode_one(tokens[2], 2, 0) ||
        !decode_one(tokens[2], 2, 1)) {
        LOG_ERR("%s: failed to build interleaved state\n", __func__);
        return false;
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, uint32_t fl, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size_ext(ctx.get(), seq_id, fl);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), seq_id, fl);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return false;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(1, flags, state_save)) {
        return false;
    }
    LOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (nset != state_save.size()) {
        LOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state_save.size());
        return false;
    }
    LOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return false;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    if (state_before.size() != state_after.size() || memcmp(state_before.data(), state_after.data(), state_before.size()) != 0) {
        LOG_ERR("\n%s: error: restored KV state is not byte-identical to the saved state\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 8: state blob round-trip
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
static bool test_state_roundtrip(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOG("\n=== Test 8: state blob round-trip ===\n");

    if (llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(tokens.data()), (int32_t) tokens.size()))) {
        LOG_ERR("\n%s: failed to decode prompt\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_a(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_a = llama_state_seq_get_data(ctx.get(), blob_a.data(), blob_a.size(), 0);
    if (n_a != blob_a.size()) {
        LOG_ERR("\n%s: saved %zu bytes, expected %zu\n", __func__, n_a, blob_a.size());
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("\n%s: failed to erase seq 0\n", __func__);
        return false;
    }

    if (llama_state_seq_set_data(ctx.get(), blob_a.data(), blob_a.size(), 0) != blob_a.size()) {
        LOG_ERR("\n%s: failed to restore seq 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> blob_b(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_b = llama_state_seq_get_data(ctx.get(), blob_b.data(), blob_b.size(), 0);
    if (n_b != n_a) {
        LOG_ERR("\n%s: re-saved %zu bytes, expected %zu\n", __func__, n_b, n_a);
        return false;
    }

    size_t n_diff = 0;
    size_t i_diff = 0;
    for (size_t i = 0; i < n_a; i++) {
        if (blob_a[i] != blob_b[i]) {
            if (n_diff == 0) {
                i_diff = i;
            }
            n_diff++;
        }
    }

    if (n_diff > 0) {
        LOG_ERR("\n%s: state changed across a restore: %zu of %zu bytes differ, first at offset %zu\n",
                __func__, n_diff, n_a, i_diff);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

#if !defined(_WIN32) || !defined(LLAMA_SHARED)
struct pp_baseline_run {
    std::vector<std::vector<float>> logits;
    std::vector<llama_pos> positions;
    std::vector<int64_t> decode_start_us;
    std::vector<int64_t> decode_end_us;
    size_t state_size = 0;
    size_t seq_state_size = 0;
    size_t seq_state_size_device = 0;
    int32_t graph_reuse_count = 0;
    uint32_t n_ctx = 0;
    uint32_t n_batch = 0;
    uint32_t n_ubatch = 0;
    bool flash_attention = false;
    bool pipeline_parallel = false;
    bool component_sizes_available = false;
    size_t kv_bytes = 0;
    size_t recurrent_r_bytes = 0;
    size_t recurrent_s_bytes = 0;
    size_t recurrent_p_bytes = 0;
    json state_layers = json::array();
};

static std::string pp_baseline_tensor_device(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) {
        return "unknown";
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    return dev ? ggml_backend_dev_name(dev) : "unknown";
}

static void pp_baseline_add_state_layer(
        pp_baseline_run & result,
                   int32_t il,
        const char * kind,
        const std::vector<ggml_tensor *> & tensors) {
    json layer = {
        { "layer", il },
        { "kind", kind },
        { "device", tensors.empty() ? "unknown" : pp_baseline_tensor_device(tensors[0]) },
    };

    if (strcmp(kind, "full_attention") == 0) {
        const size_t k_bytes = tensors.size() > 0 ? ggml_nbytes(tensors[0]) : 0;
        const size_t v_bytes = tensors.size() > 1 ? ggml_nbytes(tensors[1]) : 0;
        layer["k_bytes"] = k_bytes;
        layer["v_bytes"] = v_bytes;
        layer["state_tensor_bytes"] = k_bytes + v_bytes;
        result.kv_bytes += k_bytes + v_bytes;
    } else {
        const size_t r_bytes = tensors.size() > 0 ? ggml_nbytes(tensors[0]) : 0;
        const size_t s_bytes = tensors.size() > 1 ? ggml_nbytes(tensors[1]) : 0;
        layer["r_bytes"] = r_bytes;
        layer["s_bytes"] = s_bytes;
        if (tensors.size() > 2) {
            const size_t p_bytes = ggml_nbytes(tensors[2]);
            layer["p_bytes"] = p_bytes;
            result.recurrent_p_bytes += p_bytes;
            layer["state_tensor_bytes"] = r_bytes + s_bytes + p_bytes;
        } else {
            layer["state_tensor_bytes"] = r_bytes + s_bytes;
        }
        result.recurrent_r_bytes += r_bytes;
        result.recurrent_s_bytes += s_bytes;
    }

    result.state_layers.push_back(std::move(layer));
}

static void pp_baseline_capture_component_sizes(
        llama_context * ctx,
        llama_model * model,
        pp_baseline_run & result) {
    llama_memory_t memory = llama_get_memory(ctx);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
            const bool recurrent = model->hparams.is_recr(il);
            pp_baseline_add_state_layer(result, il, recurrent ? "recurrent" : "full_attention", hybrid->layer_state_tensors(il));
        }
        result.component_sizes_available = true;
        return;
    }

    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
            pp_baseline_add_state_layer(result, il, "full_attention", kv->layer_state_tensors(il));
        }
        result.component_sizes_available = true;
        return;
    }

    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
            pp_baseline_add_state_layer(result, il, "recurrent", recurrent->layer_state_tensors(il));
        }
        result.component_sizes_available = true;
    }
}

static bool run_pp_baseline_once(
        llama_model * model,
        const common_params & params,
        const llama_tokens & tokens,
        pp_baseline_run & result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 1;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_perf_context_reset(ctx.get());

    const auto & cparams = ctx->get_cparams();
    result.n_ctx = cparams.n_ctx;
    result.n_batch = cparams.n_batch;
    result.n_ubatch = cparams.n_ubatch;
    result.flash_attention = cparams.flash_attn;
    result.pipeline_parallel = cparams.pipeline_parallel;

    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tokens[i], (llama_pos) i, { 0 }, true);
        const int64_t decode_start_us = ggml_time_us();
        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode token %zu\n", __func__, i);
            return false;
        }

        const float * token_logits = llama_get_logits_ith(ctx.get(), 0);
        if (token_logits == nullptr) {
            LOG_ERR("%s: logits are unavailable for token %zu\n", __func__, i);
            return false;
        }
        const int64_t decode_end_us = ggml_time_us();
        result.logits.emplace_back(token_logits, token_logits + n_vocab);
        result.positions.push_back((llama_pos) i);
        result.decode_start_us.push_back(decode_start_us);
        result.decode_end_us.push_back(decode_end_us);
    }

    result.state_size = llama_state_get_size(ctx.get());
    result.seq_state_size = llama_state_seq_get_size(ctx.get(), 0);
    result.seq_state_size_device = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    result.graph_reuse_count = llama_perf_context(ctx.get()).n_reused;
    pp_baseline_capture_component_sizes(ctx.get(), model, result);
    return true;
}

static json pp_baseline_requested_params_json(const common_params & params, const llama_context_params & cparams) {
    json tensor_split = json::array();
    for (size_t i = 0; i < llama_max_devices(); i++) {
        tensor_split.push_back(params.tensor_split[i]);
    }

    return json::object({
        { "n_ctx", cparams.n_ctx },
        { "n_batch", cparams.n_batch },
        { "n_ubatch", cparams.n_ubatch },
        { "n_seq_max", 1 },
        { "n_gpu_layers", params.n_gpu_layers },
        { "main_gpu", params.main_gpu },
        { "split_mode", (int) params.split_mode },
        { "tensor_split", std::move(tensor_split) },
        { "kv_unified", cparams.kv_unified },
        { "offload_kqv", cparams.offload_kqv },
        { "flash_attention", llama_flash_attn_type_name(cparams.flash_attn_type) },
        { "type_k", ggml_type_name(cparams.type_k) },
        { "type_v", ggml_type_name(cparams.type_v) },
    });
}

static json pp_baseline_resolved_params_json(const llama_model * model, const pp_baseline_run & run) {
    json devices = json::array();
    for (const auto & device : model->devices) {
        devices.push_back({
            { "name", ggml_backend_dev_name(device.dev) },
            { "description", ggml_backend_dev_description(device.dev) },
        });
    }

    json tensor_split = json::array();
    const float * split = model->tensor_split();
    for (size_t i = 0; i < model->n_devices(); i++) {
        tensor_split.push_back(split ? split[i] : 0.0f);
    }

    return json::object({
        { "n_ctx", run.n_ctx },
        { "n_batch", run.n_batch },
        { "n_ubatch", run.n_ubatch },
        { "flash_attention", run.flash_attention },
        { "pipeline_parallel", run.pipeline_parallel },
        { "devices", std::move(devices) },
        { "tensor_split", std::move(tensor_split) },
    });
}

static std::string pp_baseline_sha256(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open model file for hashing: " + path.string());
    }

    sha256_t ctx;
    sha256_init(&ctx);
    std::array<unsigned char, 1024*1024> buffer;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const auto size = input.gcount();
        if (size > 0) {
            sha256_update(&ctx, buffer.data(), size);
        }
    }
    if (input.bad()) {
        throw std::runtime_error("failed to read model file for hashing: " + path.string());
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    static const char hex[] = "0123456789abcdef";
    std::string result(2*SHA256_DIGEST_SIZE, '0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        result[2*i + 0] = hex[digest[i] >> 4];
        result[2*i + 1] = hex[digest[i] & 0x0f];
    }
    return result;
}

static std::vector<std::filesystem::path> pp_baseline_model_paths(const std::string & model_path) {
    const std::filesystem::path path = std::filesystem::absolute(model_path);
    const std::string value = path.string();
    if (value.size() < 20) {
        return {path};
    }

    int split_no = 0;
    int split_count = 0;
    if (sscanf(value.c_str() + value.size() - 20, "-%05d-of-%05d.gguf", &split_no, &split_count) != 2 ||
            split_no < 1 || split_no > split_count) {
        return {path};
    }

    std::vector<char> buffer(value.size() + 64, 0);
    const int32_t prefix_size = llama_split_prefix(buffer.data(), buffer.size(), value.c_str(), split_no - 1, split_count);
    if (prefix_size <= 0) {
        throw std::runtime_error("failed to resolve model split prefix: " + value);
    }
    const std::string prefix(buffer.data(), prefix_size);

    std::vector<std::filesystem::path> paths;
    for (int32_t i = 0; i < split_count; ++i) {
        const int32_t size = llama_split_path(buffer.data(), buffer.size(), prefix.c_str(), i, split_count);
        if (size <= 0) {
            throw std::runtime_error("failed to resolve model split path: " + value);
        }
        paths.emplace_back(std::string(buffer.data(), size));
    }
    return paths;
}

static json pp_baseline_model_manifest(const std::string & model_path) {
    std::error_code error;
    json result = json::array();
    for (const auto & path : pp_baseline_model_paths(model_path)) {
        if (!std::filesystem::is_regular_file(path, error) || error) {
            throw std::runtime_error("model file is unavailable: " + path.string());
        }
        const uintmax_t size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error("failed to get model file size: " + path.string());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("failed to get model file timestamp: " + path.string());
        }
        result.push_back({
            { "path", path.string() },
            { "size_bytes", size },
            { "last_write_time_ticks", modified.time_since_epoch().count() },
            { "sha256", pp_baseline_sha256(path) },
        });
    }
    return result;
}

static bool write_pp_baseline(
        const std::string & output_path,
        llama_model * model,
        const common_params & params,
        const llama_tokens & tokens,
        const pp_baseline_run & first,
        const pp_baseline_run & second,
        const std::vector<std::string> & command_line,
        double max_abs_diff) {
    char model_desc[1024];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    json metadata = json::object();
    const int32_t n_meta = llama_model_meta_count(model);
    for (int32_t i = 0; i < n_meta; i++) {
        char key[1024];
        const int32_t key_len = llama_model_meta_key_by_index(model, i, key, sizeof(key));
        const int32_t val_len = llama_model_meta_val_str_by_index(model, i, nullptr, 0);
        if (key_len < 0 || val_len < 0) {
            continue;
        }
        std::vector<char> val((size_t) val_len + 1);
        if (llama_model_meta_val_str_by_index(model, i, val.data(), val.size()) >= 0) {
            metadata[key] = val.data();
        }
    }

    json layers = json::array();
    for (int32_t il = 0; il < llama_model_n_layer(model); il++) {
        ggml_backend_dev_t dev = model->dev_layer(il);
        layers.push_back({
            { "layer", il },
            { "device", dev ? ggml_backend_dev_name(dev) : "unknown" },
            { "device_description", dev ? ggml_backend_dev_description(dev) : "unknown" },
        });
    }

    json token_records = json::array();
    for (size_t i = 0; i < tokens.size(); i++) {
        token_records.push_back({
            { "token", tokens[i] },
            { "position", first.positions[i] },
            { "position_interval", json::array({ first.positions[i], first.positions[i] + 1 }) },
            { "logits", first.logits[i] },
            { "timing_us", {
                { "run_1", {
                    { "start", first.decode_start_us[i] },
                    { "end", first.decode_end_us[i] },
                    { "elapsed", first.decode_end_us[i] - first.decode_start_us[i] },
                } },
                { "run_2", {
                    { "start", second.decode_start_us[i] },
                    { "end", second.decode_end_us[i] },
                    { "elapsed", second.decode_end_us[i] - second.decode_start_us[i] },
                } },
            } },
        });
    }

    const auto cparams = common_context_params_to_llama(params);
    const char * cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    json artifact = {
        { "schema", "llama.cpp.pp-baseline.v1" },
        { "invocation", {
            { "argv", command_line },
            { "working_directory", std::filesystem::current_path().string() },
            { "environment", {
                { "CUDA_VISIBLE_DEVICES", cuda_visible_devices ? json(cuda_visible_devices) : json(nullptr) },
            } },
        } },
        { "build", {
            { "version", llama_version() },
            { "build_info", llama_build_info() },
            { "system_info", llama_print_system_info() },
        } },
        { "model", {
            { "path", params.model.path },
            { "description", model_desc },
            { "size_bytes", llama_model_size(model) },
            { "parameter_count", llama_model_n_params(model) },
            { "layer_count", llama_model_n_layer(model) },
            { "vocabulary_size", llama_vocab_n_tokens(llama_model_get_vocab(model)) },
            { "file_type", llama_ftype_name(llama_model_ftype(model)) },
            { "metadata", std::move(metadata) },
            { "files", pp_baseline_model_manifest(params.model.path) },
        } },
        { "parameters", {
            { "requested", pp_baseline_requested_params_json(params, cparams) },
            { "resolved", pp_baseline_resolved_params_json(model, first) },
        } },
        { "initial_layer_ownership", std::move(layers) },
        { "state_sizes", {
            { "context_bytes", first.state_size },
            { "sequence_host_bytes", first.seq_state_size },
            { "sequence_device_bytes", first.seq_state_size_device },
            { "kv_bytes", first.component_sizes_available ? json(first.kv_bytes) : json(nullptr) },
            { "recurrent_r_bytes", first.component_sizes_available ? json(first.recurrent_r_bytes) : json(nullptr) },
            { "recurrent_s_bytes", first.component_sizes_available ? json(first.recurrent_s_bytes) : json(nullptr) },
            { "recurrent_p_bytes", first.component_sizes_available ? json(first.recurrent_p_bytes) : json(nullptr) },
            { "layers", first.state_layers },
            { "component_size_status", first.component_sizes_available ? "tensor bytes from memory state tensors" : "unhandled memory type" },
        } },
        { "graph_reuse", {
            { "run_1", first.graph_reuse_count },
            { "run_2", second.graph_reuse_count },
        } },
        { "comparison", {
            { "identical", max_abs_diff == 0.0 },
            { "max_abs_diff", max_abs_diff },
        } },
        { "tokens", std::move(token_records) },
    };

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        LOG_ERR("%s: failed to open '%s'\n", __func__, output_path.c_str());
        return false;
    }
    output << artifact.dump(2) << '\n';
    if (!output) {
        LOG_ERR("%s: failed to write '%s'\n", __func__, output_path.c_str());
        return false;
    }
    return true;
}

static bool run_pp_baseline(
        const std::string & output_path,
        const std::string & model_path,
        const common_params & base_params,
        const std::vector<std::string> & command_line) {
    common_params params = base_params;
    params.model.path = model_path;
    params.n_parallel = 1;
    params.n_sequences = 1;
    params.no_perf = false;

    auto llama_init = common_init_from_params(params, true);
    llama_model * model = llama_init ? llama_init->model() : nullptr;
    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        return false;
    }

    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int32_t i = 0; i < params.n_batch; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        if (!ctx) {
            LOG_ERR("%s: failed to create context for tokenization\n", __func__);
            return false;
        }
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }
    if (tokens.empty()) {
        LOG_ERR("%s: fixed input is empty\n", __func__);
        return false;
    }

    pp_baseline_run first;
    pp_baseline_run second;
    if (!run_pp_baseline_once(model, params, tokens, first) || !run_pp_baseline_once(model, params, tokens, second)) {
        return false;
    }

    bool valid = first.logits.size() == second.logits.size();
    double max_abs_diff = 0.0;
    for (size_t i = 0; valid && i < first.logits.size(); i++) {
        if (first.logits[i].size() != second.logits[i].size()) {
            valid = false;
            break;
        }
        for (size_t j = 0; j < first.logits[i].size(); j++) {
            const float a = first.logits[i][j];
            const float b = second.logits[i][j];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                LOG_ERR("%s: nonfinite logit at token %zu, vocabulary index %zu\n", __func__, i, j);
                valid = false;
                break;
            }
            max_abs_diff = std::max(max_abs_diff, std::abs((double) a - (double) b));
            if (a != b) {
                valid = false;
            }
        }
    }

    if (!valid) {
        LOG_ERR("%s: identical-input runs produced different logits (max abs diff %.17g)\n", __func__, max_abs_diff);
        return false;
    }

    if (!write_pp_baseline(output_path, model, params, tokens, first, second, command_line, max_abs_diff)) {
        return false;
    }
    LOG_INF("%s: wrote PP baseline to '%s'\n", __func__, output_path.c_str());
    return true;
}
#endif


// Run the full save/load test suite (tests 1-8) for a single model.
// Returns true if all tests pass, false otherwise.
static bool run_save_load_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        return false;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return false;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return false;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 6: seq copy (host, scatter)
    if (!test_seq_cp_scatter(model, params, tokens, 6, false)) {
        return false;
    }

    // Test 7: seq copy (device, scatter)
    if (!test_seq_cp_scatter(model, params, tokens, 7, true)) {
        return false;
    }

    // Test 8: state blob round-trip
    if (!test_state_roundtrip(model, params, tokens)) {
        return false;
    }

    LOG("\nAll tests passed.\n");

    return true;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

#if !defined(_WIN32) || !defined(LLAMA_SHARED)
    const std::vector<std::string> command_line(argv, argv + argc);
#endif

    // extract test-specific options before handing the rest to the common arg parser
    std::string models_dir;
    std::string pp_baseline_path;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--pp-baseline") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --pp-baseline requires an output path\n", __func__);
                return 1;
            }
            pp_baseline_path = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    if (!pp_baseline_path.empty()) {
        if (!models_dir.empty()) {
            LOG_ERR("%s: --pp-baseline cannot be combined with --models\n", __func__);
            return 1;
        }
#if !defined(_WIN32) || !defined(LLAMA_SHARED)
        return run_pp_baseline(pp_baseline_path, params.model.path, params, command_line) ? 0 : 1;
#else
        LOG_ERR("%s: --pp-baseline is unavailable in Windows shared-library builds\n", __func__);
        return 1;
#endif
    }

    if (!models_dir.empty()) {
        // run the suite over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        LOG_INF("%s: running save/load tests over %zu models in '%s'\n", __func__, models.size(), models_dir.c_str());

        size_t n_pass = 0;
        size_t n_fail = 0;
        for (const auto & model_path : models) {
            LOG("\n================================================================\n");
            LOG_INF("%s: model %s\n", __func__, model_path.c_str());

            if (run_save_load_tests_for_model(model_path, params)) {
                n_pass++;
            } else {
                n_fail++;
            }
        }

        LOG("\n================================================================\n");
        LOG_INF("%s: summary: %zu passed, %zu failed (of %zu)\n", __func__, n_pass, n_fail, models.size());

        return n_fail == 0 ? 0 : 1;
    }

    // single-model mode
    return run_save_load_tests_for_model(params.model.path, params) ? 0 : 1;
}
