//! The 27B's KV-page twins: main's attention sources, only the K/V address edited; names kept, a library swaps whole.
const se = @import("../core/source_edit.zig");
const edit = se.edit;
const between = se.between;

const headers = @embedFile("../core/kv_pages.h") ++ @embedFile("../core/kv_page_tensor.h");

/// A prompt step's keys or values as a base that its own `kt * PD` rows land in kt's page (a 32-key step is in one).
const prompt_base =
    "inline const device bfloat* q27_page_base(const device bfloat* p, Q27AttnReadPtrs c, int kt, int hk) {\n" ++
    "  return p + int64_t(tf_paged_row(c.table, uint(kt), c.page_bytes, uint(hk), 256) / 2) - int64_t(kt) * 256;\n}\n";

/// Prefix (each 64-key tile from its page), tail (each gathered row from its page), and the read pointers' struct.
pub const attention_mpp = blk: {
    var s: []const u8 = @embedFile("attention_mpp.metal");
    s = edit(s, "using namespace mpp::tensor_ops;\n", "using namespace mpp::tensor_ops;\n" ++ headers, 1);
    s = edit(s, "struct Q27AttnReadPtrs { device const bfloat* keys; device const bfloat* values; };\n", "struct Q27AttnReadPtrs { device const bfloat* keys; device const bfloat* values; device const uint* table; ulong page_bytes; };\n" ++ prompt_base, 1);
    s = edit(s, "    auto bK = tK.slice(0, kt);\n", "    auto bK = tf_page_tensor<D>(Kb, caches[st].table, caches[st].page_bytes, kt, L, hk).slice(0, 0);\n", 1);
    s = edit(s, "    auto bV = tV.slice(hi ? 128 : 0, kt);\n", "    auto bV = tf_page_tensor<D>(Vb, caches[st].table, caches[st].page_bytes, kt, L, hk).slice(hi ? 128 : 0, 0);\n", 1);
    // the tail's head base and row step are the page's now
    s = edit(s, "  const device bfloat* kbase = (const device bfloat*)Kb + (int64_t)hk * meta[MG + st * MS + 7];\n" ++
        "  const device bfloat* vbase = (const device bfloat*)Vb + (int64_t)hk * meta[MG + st * MS + 8];\n  const int64_t kstep = D, vstep = D;\n", "", 1);
    s = edit(s, "(kbase + phys * kstep + col)", "((const device bfloat*)((const device uchar*)Kb + tf_paged_row(caches[st].table, uint(phys), caches[st].page_bytes, hk, D)) + col)", 1);
    s = edit(s, "(vbase + phys * vstep + col)", "((const device bfloat*)((const device uchar*)Vb + tf_paged_row(caches[st].table, uint(phys), caches[st].page_bytes, hk, D)) + col)", 1);
    // the non-finite scan over a key range: main's, each key's row from its page (q27_nonfinite_paged, beside it)
    s = edit(s, "\n// SG tiles a threadgroup, a simdgroup pair each", "\n" ++ scan(s) ++ "\n// SG tiles a threadgroup, a simdgroup pair each", 1);
    // the masked prefix: its half's base without a head stride, its scan and each masked key's values through the table
    s = edit(s, "Vb + (int64_t)hk * meta[MG + st * MS + 8] + (hi ? 128 : 0)", "Vb + (hi ? 128 : 0)", 1);
    s = edit(s, "q27_nonfinite(vh, max(seen, kbeg), kend, lane)", "q27_nonfinite_paged(vh, caches[st], hk, max(seen, kbeg), kend, lane)", 1);
    s = edit(s, "vh + (int64_t)key * D + fn", "vh + (int64_t)(tf_paged_row(caches[st].table, uint(key), caches[st].page_bytes, hk, D) / 2) + fn", 1);
    break :blk s;
};

/// main's q27_nonfinite as q27_nonfinite_paged: `v` the half's base, a key's row in its page of `c`, head `hk`.
fn scan(comptime text: []const u8) []const u8 {
    var f = between(text, "inline bool q27_nonfinite(", "\n// SG tiles");
    f = edit(f, "q27_nonfinite(const device bfloat* v, int lo,", "q27_nonfinite_paged(const device bfloat* v, Q27AttnReadPtrs c, uint hk, int lo,", 1);
    f = edit(f, "(v + (int64_t)key * D + (lane & 15) * 8)", "(v + (int64_t)(tf_paged_row(c.table, uint(key), c.page_bytes, hk, D) / 2) + (lane & 15) * 8)", 1);
    return f;
}

/// Append, gather and scatter: each row through its stream's table.
pub const attention_io = blk: {
    var s: []const u8 = @embedFile("attention_io.metal");
    s = edit(s, "struct Q27CachePtrs { device bfloat* keys; device bfloat* values; };\n", "struct Q27CachePtrs { device bfloat* keys; device bfloat* values; device const uint* table; ulong page_bytes; };\n" ++
        "inline device bfloat* q27_paged(device bfloat* base, Q27CachePtrs c, uint hk, uint row, uint dim) {\n" ++
        "  return (device bfloat*)((device uchar*)base + tf_paged_row(c.table, row, c.page_bytes, hk, dim));\n}\n", 1);
    s = edit(s, "caches[st].keys[ulong(at.y) * uint(meta[base + 7]) + row * a.dim + at.x]", "q27_paged(caches[st].keys, caches[st], at.y, row, a.dim)[at.x]", 1);
    s = edit(s, "caches[st].values[ulong(at.y) * uint(meta[base + 8]) + row * a.dim + at.x]", "q27_paged(caches[st].values, caches[st], at.y, row, a.dim)[at.x]", 1);
    s = edit(s, "caches[r.stream].keys[ulong(at.y) * uint(meta[base + 7]) + source * 256 + at.x]", "q27_paged(caches[r.stream].keys, caches[r.stream], at.y, source, 256)[at.x]", 1);
    s = edit(s, "caches[r.stream].values[ulong(at.y) * uint(meta[base + 8]) + source * 256 + at.x]", "q27_paged(caches[r.stream].values, caches[r.stream], at.y, source, 256)[at.x]", 1);
    s = edit(s, "caches[r.stream].keys[ulong(at.y) * uint(meta[base + 7]) + destination * 256 + at.x]", "q27_paged(caches[r.stream].keys, caches[r.stream], at.y, destination, 256)[at.x]", 1);
    s = edit(s, "caches[r.stream].values[ulong(at.y) * uint(meta[base + 8]) + destination * 256 + at.x]", "q27_paged(caches[r.stream].values, caches[r.stream], at.y, destination, 256)[at.x]", 1);
    break :blk s;
};

/// A prompt kernel's keys and values: no head stride (a page's head runs are inside it), each step based at its page.
fn prompt(comptime text: []const u8) []const u8 {
    var s = text;
    s = edit(s, "caches[0].keys + (int64_t)hk * args.key_stride + h * PH", "caches[0].keys + h * PH", 1);
    s = edit(s, "caches[0].values + (int64_t)hk * args.value_stride + h * PH", "caches[0].values + h * PH", 1);
    // every step (main's, the masked steps and the edge) based at its page
    s = edit(s, "(r, kt, kend, q, k, v, xs", "(r, kt, kend, q, q27_page_base(k, caches[0], kt, hk), q27_page_base(v, caches[0], kt, hk), xs", 4);
    s = edit(s, "q27_nonfinite(v, max(seen, kbeg), kend, lane)", "q27_nonfinite_paged(v, caches[0], uint(hk), max(seen, kbeg), kend, lane)", 1);
    return s;
}

/// The prompt kernels and the scan picking between them: each scanned key's values through stream 0's table.
pub const attention_prompt = blk: {
    var s = prompt(@embedFile("attention_prompt.metal"));
    s = edit(s, "caches[0].values + (int64_t)tg.x * args.value_stride;", "caches[0].values;", 1);
    s = edit(s, "(v + (int64_t)key * PD + lane * 8)", "(v + (int64_t)(tf_paged_row(caches[0].table, uint(key), caches[0].page_bytes, tg.x, PD) / 2) + lane * 8)", 1);
    break :blk s;
};
pub const attention_prompt_sg = prompt(@embedFile("attention_prompt_sg.metal"));
