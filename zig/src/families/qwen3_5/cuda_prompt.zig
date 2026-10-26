//! A one-part prompt chunk's attention tiles and delta-rule chain (uncaptured: the host passes the stream's pointers).
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const Seg = @import("cuda_cache.zig").Seg;
const head_dim = geometry.head_dim;

/// Two 32-key slots of 256 dims beside eight warps' staged queries (hpattn_kernel<256, 8, 2, 2>).
pub const smem: u32 = 98304;
/// Query heads a block (the kernels' HPC) and rows a block (16 a warp, four warp rows).
const hpc = 2;
const block_rows = 16 * (8 / hpc);

pub const Fns = struct { bf16: cuda.Function, fp8: cuda.Function, bf16_paged: cuda.Function, fp8_paged: cuda.Function, chain: cuda.Function };

pub const symbols = struct {
    pub const bf16 = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi";
    pub const fp8 = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
    /// The same tiles over the pool's pages (hpattn_paged_kernel, hpattn8_paged_kernel): the contiguous kernels' bits.
    pub const bf16_paged = "_ZN20tf_prefill_attention19hpattn_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKix";
    pub const fp8_paged = "_ZN20tf_prefill_attention20hpattn8_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKix";
    pub const chain = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi32EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii";
};
/// gdn_prefill.cu's value rows a block (two threads a row).
const chain_rows = 64;

pub fn resolve(attention: cuda.Module, gdn: cuda.Module) !Fns {
    const f: Fns = .{ .bf16 = try attention.function(symbols.bf16), .fp8 = try attention.function(symbols.fp8), .bf16_paged = try attention.function(symbols.bf16_paged), .fp8_paged = try attention.function(symbols.fp8_paged), .chain = try gdn.function(symbols.chain) };
    for ([_]cuda.Function{ f.bf16, f.fp8, f.bf16_paged, f.fp8_paged }) |x| try x.allowDynamicShared(smem);
    return f;
}

/// Attention layer `layer` into e.y: a chunk's queries (e.qn) over cache positions 0 .. len + row (rope wrote its own).
pub fn run(e: *E.Engine, layer: usize, rows: usize) !void {
    const g = e.g;
    const sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.Engine.segs_at..][0..@sizeOf(Seg)]);
    const paged = sg.pt != 0; // a page holds ptok rows a key head as a cache of cap ptok does (qwen35.cuh kv_row)
    const kvh: u64 = g.kv_heads;
    const cap: u64 = @intCast(if (paged) sg.ptok else sg.cap);
    const heads = kvh * cap; // rows of one layer's keys (or values)
    const group = g.q_heads / g.kv_heads;
    var a: cuda.Args = .{};
    if (e.kv8) {
        const base = sg.kv + layer * 2 * heads * (head_dim + 4);
        const keys = base;
        const values = keys + heads * head_dim;
        const kscale = values + heads * head_dim;
        for ([_]u64{ e.qn, keys, values, kscale, kscale + heads * 4, e.y }) |v| a.add(v);
    } else {
        const base = sg.kv + layer * 2 * heads * head_dim * 2;
        for ([_]u64{ e.qn, base, base + heads * head_dim * 2, e.y }) |v| a.add(v);
    }
    for ([_]usize{ @intCast(sg.len), rows, g.q_heads, g.kv_heads, group }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(@as(f32, 1.0 / 16.0));
    a.add(@as(c_int, @intCast(cap)));
    if (paged) {
        a.add(sg.pt);
        a.add(@as(c_longlong, @intCast(sg.ps)));
    }
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((rows + block_rows - 1) / block_rows), .y = @intCast(g.kv_heads * (group / hpc)) }, .block = .{ .x = 256 }, .shared = smem };
    const f = if (paged) (if (e.kv8) e.prompt_attn.fp8_paged else e.prompt_attn.bf16_paged) else (if (e.kv8) e.prompt_attn.fp8 else e.prompt_attn.bf16);
    try cuda.launch.launch(f, cfg, e.stream, &a);
}

/// Linear layer `layer`'s delta rule: e.q, e.k, e.v, e.g_, e.beta into e.y, the stream's recurrence updated in place.
pub fn delta(e: *E.Engine, layer: usize, rows: usize) !void {
    const g = e.g;
    const sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.Engine.segs_at..][0..@sizeOf(Seg)]);
    const state = sg.rec + layer * g.v_heads * geometry.dk * geometry.dk * 4;
    var a: cuda.Args = .{};
    for ([_]u64{ e.q, e.k, e.v, e.g_, e.beta, state, state, e.y }) |v| a.add(v);
    for ([_]usize{ rows, g.k_heads, g.v_heads }) |v| a.add(@as(c_int, @intCast(v)));
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(g.v_heads), .y = @intCast(geometry.dk / chain_rows) }, .block = .{ .x = 2 * chain_rows } };
    try cuda.launch.launch(e.prompt_attn.chain, cfg, e.stream, &a);
}
