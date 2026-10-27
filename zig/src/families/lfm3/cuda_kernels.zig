//! LFM3's launches on CUDA: the shared operators (linears, norms, QK rows, attention, gates), lfm3.cu's 7-tap conv.
const std = @import("std");
const cuda = @import("cuda");
const c = @import("config.zig");

const Args = cuda.Args;
const Function = cuda.Function;
const Stream = cuda.Stream;

/// prefill_attention.cu's pattn_kernel<128, 8, 4, 8> over paged caches: eight 32-key slots, 32 rows a block.
const pattn_sym = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";
const pattn_smem: u32 = 8 * 32 * 16 * 16;
const pattn_rows = 32;
/// Positions a KV page holds (nemotron_attention.cu's PAGE and prefill_attention_rows.cuh's 512).
pub const page = 512;
/// Streams one shared-window attention launch takes (nemotron_attention.cu's MAXSEG).
pub const max_segs = 8;
/// The attention's softmax scale: head_dim^-0.5.
pub const scale: f32 = 1.0 / @sqrt(@as(f32, c.head_dim));

/// nemotron_attention.cu's AttnSegs: each stream's caches, position, page table and first row.
pub const AttnSegs = extern struct { kc: [max_segs]u64, vc: [max_segs]u64, meta: [max_segs]u64, pt: [max_segs]u64, ps: i64, row0: [max_segs]c_int, n: c_int };

const images = .{ "sample", "tree_round", "prefill_attention", "nemotron_attention", "lfm2", "lfm3" };

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [images.len]cuda.Module,
    lane: cuda.qmmf.Lane,
    prompt16: cuda.nvfp4.Prompt,
    pack: cuda.modelopt.Packer,
    draws: cuda.sample.Draws,
    round: cuda.tree_round.Kernels,
    embed: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    win: cuda.window_attention.WindowAttention,
    pattn: Function,
    kv_write: Function,
    chunk: Function,
    merge: Function,
    silu_mul: Function,
    conv: Function,
    commit: Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        inline for (images, 0..) |name, i| {
            k.mods[i] = try cuda.Module.load(d, @field(cuda.kernels, name));
            loaded += 1;
        }
        const m = &k.mods;
        k.draws = try cuda.sample.Draws.resolve(m[0]);
        k.round = try cuda.tree_round.Kernels.resolve(m[1]);
        k.pattn = try m[2].function(pattn_sym);
        try k.pattn.allowDynamicShared(pattn_smem);
        k.kv_write = try m[3].function("tf_nemo_kv_write_segs");
        k.chunk = try m[3].function("tf_nemo_attn_chunk_segs");
        k.merge = try m[3].function("tf_nemo_attn_merge_segs");
        k.silu_mul = try m[4].function("tf_lfm2_silu_mul");
        k.conv = try m[5].function("tf_liquid_conv_7");
        k.commit = try m[5].function("tf_liquid_commit_7");
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.prompt16 = try cuda.nvfp4.Prompt.load(d);
        errdefer k.prompt16.unload();
        k.pack = try cuda.modelopt.Packer.load(d);
        errdefer k.pack.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        errdefer k.act.unload();
        k.win = try cuda.window_attention.WindowAttention.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.win.unload();
        k.act.unload();
        k.qk.unload();
        k.norms.unload();
        k.embed.unload();
        k.pack.unload();
        k.prompt16.unload();
        k.lane.unload();
        for (&k.mods) |*m| m.unload();
    }

    /// The lane for every row of a decode forward and the bf16 projections, prompt16 for an FP4 prompt chunk.
    pub fn linear(k: *const Kernels) cuda.qlinear.Linear {
        return .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt16 };
    }

    fn go(f: Function, s: Stream, grid: [3]usize, block: usize, smem: u32, a: *Args) !void {
        const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = @intCast(block) }, .shared = smem }, s, a);
    }

    /// Short conv over bcx [rows, 3 hidden]: each row's slot and offset; a slot's [6, hidden] state every `ss`.
    pub fn convRows(k: *const Kernels, s: Stream, bcx: u64, w: u64, state: u64, ss: usize, slot: u64, off: u64, y: u64, snap: ?u64, snap0: usize, rows: usize) !void {
        var a: Args = .{};
        for ([_]u64{ bcx, w, state, slot, off, y, snap orelse 0 }) |v| a.add(v);
        a.add(int(snap0));
        a.add(int(c.hidden));
        a.add(@as(i64, @intCast(ss)));
        try go(k.conv, s, .{ rows, c.hidden / 256, 1 }, 256, 0, &a);
    }

    /// Each stream k's conv states (every conv layer) set to its snapshot row rows[k] (int32; < 0: kept).
    pub fn commitRows(k: *const Kernels, s: Stream, state: u64, snap: u64, slots: u64, rows: u64, streams: usize, layers: usize, state_stride: usize, snap_stride: usize, ss: usize) !void {
        if (streams == 0) return;
        var a: Args = .{};
        for ([_]u64{ state, snap, slots, rows }) |v| a.add(v);
        a.add(int(c.hidden));
        a.add(@as(i64, @intCast(state_stride)));
        a.add(@as(i64, @intCast(snap_stride)));
        a.add(@as(i64, @intCast(ss)));
        try go(k.commit, s, .{ layers, streams, 1 }, 256, 0, &a);
    }

    /// out [rows, f] = bf16(bf16(silu(gate)) * up) from rows gate | up [rows, 2 f] (lfm2.cu, torch's points).
    pub fn siluMul(k: *const Kernels, s: Stream, g: u64, out: u64, rows: usize, f: usize) !void {
        var a: Args = .{};
        a.add(g);
        a.add(out);
        a.add(int(f));
        try go(k.silu_mul, s, .{ rows, (f + 255) / 256, 1 }, 256, 0, &a);
    }

    /// Rows k, v (kv [rows, 2 kv width]) into each stream's pages at its position on.
    pub fn kvWrite(k: *const Kernels, s: Stream, sg: AttnSegs, kv: u64, rows: usize) !void {
        var a: Args = .{};
        a.add(sg);
        a.add(kv);
        a.add(int(2 * c.kv_heads * c.head_dim));
        a.add(@as(c_int, 0));
        a.add(int(c.kv_heads * c.head_dim));
        try go(k.kv_write, s, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// Decode rows: q [rows, heads, 128] over each stream's pages in 512-key chunks, merged in order (xs: sums).
    pub fn attend(k: *const Kernels, s: Stream, sg: AttnSegs, q: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, nch: usize, z: usize) !void {
        const h = c.query_heads;
        const hk = c.kv_heads;
        const g = h / hk;
        var a: Args = .{};
        a.add(sg);
        a.add(q);
        for ([_]u64{ po, pm, pl }) |v| a.add(v);
        for ([_]usize{ h * c.head_dim, h, hk, g, page, nch }) |v| a.add(int(v));
        a.add(scale);
        try go(k.chunk, s, .{ rows, hk, z }, 256, 0, &a);
        var b: Args = .{};
        b.add(sg);
        for ([_]u64{ po, pm, pl, out, xs }) |v| b.add(v);
        for ([_]usize{ h, hk, g, page, nch }) |v| b.add(int(v));
        try go(k.merge, s, .{ rows, hk, 1 }, 256, 0, &b);
    }

    /// A prompt chunk's rows at positions p0.. against its stream's pages (filled through them): out [rows, 16, 128].
    pub fn prompt(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, pt: u64, ps: usize) !void {
        const g = c.query_heads / c.kv_heads;
        var a: Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, c.query_heads, c.kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        a.add(pt);
        a.add(@as(i64, @intCast(ps)));
        try go(k.pattn, s, .{ (rows + pattn_rows - 1) / pattn_rows, c.kv_heads * (g / 4), 1 }, 256, pattn_smem, &a);
    }
};

test "the attention launches' geometry: 4 query heads a KV head, a pattn block's 4 heads, head 128" {
    try std.testing.expectEqual(@as(usize, 4), c.query_heads / c.kv_heads);
    try std.testing.expectEqual(@as(usize, 304), @sizeOf(AttnSegs)); // nemotron_attention.cu's struct, padded to 8
    try std.testing.expectEqual(@as(u32, 65536), pattn_smem); // lfm3_cuda_records.zig's dynamic bytes
    try std.testing.expectEqual(@as(usize, 0), c.hidden % 256); // the conv's channel blocks
}
