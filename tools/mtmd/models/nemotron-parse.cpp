#include "models.h"

#include <algorithm>
#include <cmath>

// C-RADIOv4-SO400M + neck of Nemotron 3.5 Parse
// ref: hf_nemotron_parse_modeling.py (RadioWithNeck) and hf_prefix_geometry.py
ggml_cgraph * mtmd_graph_nemotron_parse::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(n_patches_x % 4 == 0);

    const int n_cls  = model.class_embedding->ne[1]; // 2 summary tokens + 8 registers
    const int n_src  = (int) std::lround(std::sqrt((double) model.position_embeddings->ne[1]));
    const int n_x    = n_patches_x;
    const int n_y    = n_patches_y;

    ggml_tensor * inp = build_inp();

    // the learned n_src x n_src position grid, resampled to the page by two small matrices (see pos_weights)
    ggml_tensor * pos_ax = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_src, n_x);
    ggml_set_name(pos_ax, "pos_ax");
    ggml_set_input(pos_ax);

    ggml_tensor * pos_ay = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_src, n_y);
    ggml_set_name(pos_ay, "pos_ay");
    ggml_set_input(pos_ay);

    ggml_tensor * pos = ggml_reshape_3d(ctx0, model.position_embeddings, n_embd, n_src, n_src); // [embd, col, row]
    pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 1, 0, 2, 3));                               // [col, embd, row]
    pos = ggml_reshape_2d(ctx0, pos, n_src, n_embd * n_src);
    pos = ggml_mul_mat(ctx0, pos, pos_ax);                                                     // [embd * row, x]
    pos = ggml_reshape_3d(ctx0, pos, n_embd, n_src, n_x);
    pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 1, 0, 2, 3));                               // [row, embd, x]
    pos = ggml_reshape_2d(ctx0, pos, n_src, n_embd * n_x);
    pos = ggml_mul_mat(ctx0, pos, pos_ay);                                                     // [embd * x, y]
    pos = ggml_reshape_2d(ctx0, pos, n_embd, n_x * n_y);
    cb(pos, "pos_embd", -1);

    inp = ggml_add(ctx0, inp, pos);
    inp = ggml_concat(ctx0, model.class_embedding, inp, 1);

    ggml_tensor * cur = build_vit(inp, n_cls + n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);

    // summary: the 2 CLS tokens side by side
    ggml_tensor * summary = ggml_view_2d(ctx0, cur, 2 * n_embd, 1, ggml_row_size(cur->type, 2 * n_embd), 0);
    summary = build_mm(model.mm_4_w, ggml_cont(ctx0, summary));
    summary = ggml_add(ctx0, summary, model.mm_4_b);
    summary = build_norm(summary, model.mm_5_w, model.mm_5_b, NORM_TYPE_NORMAL, eps, -1);
    cb(summary, "summary", -1);

    ggml_tensor * feat = ggml_view_2d(ctx0, cur, n_embd, n_patches, cur->nb[1], n_cls * cur->nb[1]);
    feat = build_mm(model.mm_0_w, feat);
    feat = ggml_add(ctx0, feat, model.mm_0_b);
    feat = build_norm(feat, model.mm_1_w, model.mm_1_b, NORM_TYPE_NORMAL, eps, -1);

    // (1, 4) conv with stride 4: each run of 4 patches in a row becomes one token
    feat = ggml_reshape_2d(ctx0, feat, 4 * feat->ne[0], n_patches / 4);
    feat = build_mm(model.mm_2_w, feat);
    feat = build_norm(feat, model.mm_3_w, model.mm_3_b, NORM_TYPE_NORMAL, eps, -1);
    cb(feat, "neck", -1);

    cur = ggml_concat(ctx0, feat, summary, 1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Along one axis, the reference resizes the grid to the content patches with interpolate(bilinear, antialias=True)
// (adaptive_avg_pool2d if the content is 1 patch wide), then grid_sample(bilinear, border) at each patch center.
// Both steps are linear, so they fold into one [n_out, n_src] matrix.
std::vector<float> mtmd_graph_nemotron_parse::pos_weights(int n_out, int content_px, int patch_size, int n_src, bool pool) {
    const int n_c = (content_px + patch_size - 1) / patch_size;
    GGML_ASSERT(n_c > 0 && n_c <= n_out);

    // resize: rs[c * n_src + s]
    std::vector<double> rs((size_t) n_c * n_src, 0.0);
    if (pool) {
        for (int c = 0; c < n_c; c++) {
            const int s0 = (int) std::floor((double) c * n_src / n_c);
            const int s1 = (int) std::ceil((double) (c + 1) * n_src / n_c);
            for (int s = s0; s < s1; s++) {
                rs[(size_t) c * n_src + s] = 1.0 / (s1 - s0);
            }
        }
    } else {
        const double scale    = (double) n_src / n_c;
        const double support  = scale >= 1.0 ? scale : 1.0;
        const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
        for (int c = 0; c < n_c; c++) {
            const double center = scale * (c + 0.5);
            const int xmin = std::max((int) (center - support + 0.5), 0);
            const int xmax = std::min((int) (center + support + 0.5), n_src);
            double total = 0.0;
            for (int s = xmin; s < xmax; s++) {
                const double w = std::max(0.0, 1.0 - std::fabs((s - center + 0.5) * invscale));
                rs[(size_t) c * n_src + s] = w;
                total += w;
            }
            if (total != 0.0) {
                for (int s = xmin; s < xmax; s++) {
                    rs[(size_t) c * n_src + s] /= total;
                }
            }
        }
    }

    // sample at patch centers, normalized by the content size (align_corners = false, border padding)
    std::vector<float> out((size_t) n_out * n_src, 0.0f);
    for (int o = 0; o < n_out; o++) {
        const double start  = (double) o * patch_size;
        const double extent = std::clamp((double) content_px - start, 0.0, (double) patch_size);
        const double norm   = std::min(start + 0.5 * extent, (double) content_px) / content_px;
        const double ix     = std::clamp(norm * n_c - 0.5, 0.0, (double) (n_c - 1));
        const int    x0     = (int) std::floor(ix);
        const int    x1     = std::min(x0 + 1, n_c - 1);
        const double w1     = ix - x0;
        for (int s = 0; s < n_src; s++) {
            out[(size_t) o * n_src + s] = (float) ((1.0 - w1) * rs[(size_t) x0 * n_src + s] + w1 * rs[(size_t) x1 * n_src + s]);
        }
    }
    return out;
}
