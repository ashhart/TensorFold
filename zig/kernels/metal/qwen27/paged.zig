//! The 27B's KV-page twins: main's attention sources, only the K/V address edited; names kept, a library swaps whole.
const se = @import("../core/source_edit.zig");
const edit = se.edit;

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
    break :blk s;
};

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
fn prompt(comptime text: []const u8, comptime step: []const u8) []const u8 {
    var s = text;
    s = edit(s, "caches[0].keys + (int64_t)hk * args.key_stride + h * PH", "caches[0].keys + h * PH", 1);
    s = edit(s, "caches[0].values + (int64_t)hk * args.value_stride + h * PH", "caches[0].values + h * PH", 1);
    s = edit(s, step ++ "<false>(r, kt, kend, q, k, v, ", step ++ "<false>(r, kt, kend, q, q27_page_base(k, caches[0], kt, hk), q27_page_base(v, caches[0], kt, hk), ", 1);
    s = edit(s, step ++ "<true>(r, kt, kend, q, k, v, ", step ++ "<true>(r, kt, kend, q, q27_page_base(k, caches[0], kt, hk), q27_page_base(v, caches[0], kt, hk), ", 1);
    return s;
}

pub const attention_prompt = prompt(@embedFile("attention_prompt.metal"), "q27_prompt_step");
pub const attention_prompt_sg = prompt(@embedFile("attention_prompt_sg.metal"), "q27_sg_step");
