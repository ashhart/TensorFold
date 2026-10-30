// Ours: sample.cu's draw for a window's sampled rows in one launch (included after tf_sample::draw).
#pragma once

// Rows of bf16 logits [rows, V], row r under RULE[r] at position META[r] + 1: one launch for a window's sampled rows.
extern "C" __global__ void __launch_bounds__(1024) tf_draw_rows(const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule,
                                                                const int* meta, unsigned* TOK) {
    tf_sample::draw<false, true>(L, V, rule, meta, 0, TOK, nullptr, nullptr);
}
