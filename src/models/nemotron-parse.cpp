#include "models.h"

void llama_model_nemotron_parse::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_nemotron_parse::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // tied lm_head
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq   = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   i), {n_embd_head_k * n_head}, 0);
        layer.wk   = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   i), {n_embd_gqa}, 0);
        layer.wv   = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   i), {n_embd_gqa}, 0);
        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_nemotron_parse::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// pre-LN mBART decoder layers with no cross-attention and no position encoding;
// the image embeddings are a causal prefix of the token sequence
llama_model_nemotron_parse::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // token embeddings are scaled by sqrt(d_model), image embeddings are not
    inpL = build_inp_embd(model.tok_embd, sqrtf(n_embd));

    auto * inp_attn = build_attn_inp_kv();

    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_GELU_ERF, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
