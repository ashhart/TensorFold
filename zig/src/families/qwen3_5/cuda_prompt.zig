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

pub const Fns = struct { bf16: cuda.Function, fp8: cuda.Function, bf16_whole: cuda.Function, fp8_whole: cuda.Function, chain: cuda.Function };

pub const symbols = struct {
    pub const bf16 = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi";
    pub const fp8 = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi";
    pub const bf16_whole = "_ZN20tf_prefill_attention19hpattn_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi";
    pub const fp8_whole = "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
    pub const chain = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi32EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii";
};
/// gdn_prefill.cu's value rows a block (two threads a row).
const chain_rows = 64;

pub fn resolve(attention: cuda.Module, gdn: cuda.Module) !Fns {
    const f: Fns = .{ .bf16 = try attention.function(symbols.bf16), .fp8 = try attention.function(symbols.fp8), .bf16_whole = try attention.function(symbols.bf16_whole), .fp8_whole = try attention.function(symbols.fp8_whole), .chain = try gdn.function(symbols.chain) };
    try f.bf16.allowDynamicShared(smem);
    try f.fp8.allowDynamicShared(smem);
    try f.bf16_whole.allowDynamicShared(smem);
    try f.fp8_whole.allowDynamicShared(smem);
    return f;
}

fn arguments(e: *const E.Engine, sg: Seg, layer: usize, rows: usize) cuda.Args {
    const g = e.g;
    const kvh: u64 = g.kv_heads;
    const cap: u64 = @intCast(sg.cap);
    const paged = sg.pt != 0;
    const heads = kvh * (if (paged) @as(u64, @intCast(sg.ptok)) else cap); // rows of one layer's keys (or values): a slab's, or a page's
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
        a.add(@as(i64, @intCast(sg.ps)));
        a.add(@as(c_int, sg.ptok));
    }
    return a;
}

/// Attention layer `layer` into e.y: a chunk's queries (e.qn) over cache positions 0 .. len + row (rope wrote its own).
pub fn run(e: *E.Engine, layer: usize, rows: usize) !void {
    const g = e.g;
    const group = g.q_heads / g.kv_heads;
    const sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.Engine.segs_at..][0..@sizeOf(Seg)]);
    const paged = sg.pt != 0;
    var a = arguments(e, sg, layer, rows);
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((rows + block_rows - 1) / block_rows), .y = @intCast(g.kv_heads * (group / hpc)) }, .block = .{ .x = 256 }, .shared = smem };
    const f = if (paged) (if (e.kv8) e.prompt_attn.fp8 else e.prompt_attn.bf16) else (if (e.kv8) e.prompt_attn.fp8_whole else e.prompt_attn.bf16_whole);
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

test "Qwen prompt layout: whole ABI omits page metadata and keeps original layer and head strides" {
    const t = std.testing;
    var e: E.Engine = undefined;
    e.g.q_heads = 24;
    e.g.kv_heads = 4;
    e.qn = 0x10000;
    e.y = 0x20000;
    for ([_]bool{ false, true }) |fp8| for ([_]bool{ false, true }) |paged| {
        e.kv8 = fp8;
        var sg = std.mem.zeroes(Seg);
        sg.kv = 0x400000;
        sg.cap = 65536;
        sg.len = 8191;
        sg.pt = if (paged) 0xdead0000 else 0;
        sg.ps = if (paged) 0x100000 else std.math.maxInt(u64);
        sg.ptok = if (paged) 48 else 0;
        for ([_]usize{ 0, 1, 15 }) |layer| for ([_]usize{ 1, 63, 64, 65, 2048 }) |rows| {
            const a = arguments(&e, sg, layer, rows);
            const np: usize = if (fp8) 6 else 4;
            const heads: u64 = 4 * @as(u64, if (paged) 48 else 65536);
            const base = sg.kv + layer * 2 * heads * @as(u64, if (fp8) head_dim + 4 else head_dim * 2);
            try t.expectEqual(np + 7 + @as(usize, if (paged) 3 else 0), a.count);
            try t.expectEqual(e.qn, a.integer(0).?);
            try t.expectEqual(base, a.integer(1).?);
            try t.expectEqual(base + heads * head_dim * @as(u64, if (fp8) 1 else 2), a.integer(2).?);
            if (fp8) {
                try t.expectEqual(base + heads * head_dim * 2, a.integer(3).?);
                try t.expectEqual(base + heads * (head_dim * 2 + 4), a.integer(4).?);
            }
            try t.expectEqual(e.y, a.integer(np - 1).?);
            for ([_]usize{ 8191, rows, 24, 4, 6 }, 0..) |v, i| try t.expectEqual(v, a.integer(np + i).?);
            try t.expectEqual(@as(u32, @bitCast(@as(f32, 1.0 / 16.0))), a.integer(np + 5).?);
            try t.expectEqual(@as(u64, 65536), a.integer(np + 6).?);
            if (paged) {
                try t.expectEqual(sg.pt, a.integer(np + 7).?);
                try t.expectEqual(sg.ps, a.integer(np + 8).?);
                try t.expectEqual(@as(u64, 48), a.integer(np + 9).?);
            }
        };
    };
}
