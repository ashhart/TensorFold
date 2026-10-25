//! The Qwen3.5 vision tower on Intel Xe2 (kernels/xpu/qwen_vision.cl), as vision/qwen_cuda.zig runs it on CUDA: an
//! image's patches to the rows its placeholders take in the language model, bf16 weights as the checkpoint stores them.
const std = @import("std");
const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const dev = @import("dev.zig");
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const off = dev.off;
const alloc = std.heap.c_allocator;

/// What every Qwen3.5 tower shares (vision_config); the width, heads, depth and MLP are the checkpoint's.
pub const patch = 16;
pub const temporal = 2;
pub const merge = 2;
pub const patch_width = 3 * temporal * patch * patch; // 1536
pub const grid_side = 48; // sqrt(num_position_embeddings 2304)
pub const head_dim = 72; // qv_attention's HD
const eps: f32 = 1e-6;

pub const Config = struct {
    hidden: u32,
    heads: u32,
    depth: u32,
    intermediate: u32,
    out_hidden: u32, // the language model's width

    /// The merger's width: four merge-ordered rows read as one.
    pub fn merged(c: Config) u32 {
        return c.hidden * merge * merge;
    }
};

const Block = struct { norm1_w: *anyopaque, norm1_b: *anyopaque, qkv_w: *anyopaque, qkv_b: *anyopaque, proj_w: *anyopaque, proj_b: *anyopaque, norm2_w: *anyopaque, norm2_b: *anyopaque, fc1_w: *anyopaque, fc1_b: *anyopaque, fc2_w: *anyopaque, fc2_b: *anyopaque };

pub const Tower = struct {
    c: Config,
    max_patches: u32,
    k: struct { gemm: ze.H, embed: ze.H, layernorm: ze.H, qkv_rope: ze.H, attention: ze.H, bias_gelu: ze.H, bias_residual: ze.H, bias_round: ze.H },
    patch_w: *anyopaque, // [hidden, channel x frame x y x x]: the patches' own order
    patch_b: *anyopaque,
    pos: *anyopaque, // [48 x 48, hidden]
    blocks: []Block,
    norm_w: *anyopaque, // the merger's
    norm_b: *anyopaque,
    fc1_w: *anyopaque,
    fc1_b: *anyopaque,
    fc2_w: *anyopaque,
    fc2_b: *anyopaque,
    inv: *anyopaque, // the axial rotary's head_dim / 4 frequencies
    // scratch for max_patches patches
    px: *anyopaque, // [P, patch_width] bf16
    x: *anyopaque, // [P, hidden] bf16: the residual stream
    h: *anyopaque, // [P, hidden] bf16: a norm's output, the merger's input [T, merged]
    acc: *anyopaque, // [P, max(3 hidden, intermediate)] fp32: a GEMM's sums
    q: *anyopaque, // [heads, P, head_dim] bf16
    kk: *anyopaque,
    v: *anyopaque,
    att: *anyopaque, // [P, hidden] bf16: the heads side by side
    mid: *anyopaque, // [P, intermediate] bf16: the MLP's hidden rows, the merger's [T, merged]
    hw: *anyopaque, // [P, 2] i32: a patch's row and column
    idx: *anyopaque, // [P, 4] i32: its four learned-position taps
    wts: *anyopaque, // [P, 4] f32
    bytes: usize, // device bytes: weights and scratch

    /// The tower `prefix` (vision_tower. for MLX, model.visual. for HF) names in `ck`, bf16, with scratch for images of
    /// up to `max_patches` patches.
    pub fn load(d: *Dev, ck: *const Checkpoint, prefix: []const u8, c: Config, max_patches: u32) !*Tower {
        if (c.hidden / c.heads != head_dim or c.hidden % c.heads != 0) return error.UnsupportedVisionTower;
        const t = try alloc.create(Tower);
        errdefer alloc.destroy(t);
        const before = d.allocated;
        const m = try d.g.module(&kb.qwen_vision);
        t.c = c;
        t.max_patches = max_patches;
        t.k = .{
            .gemm = try d.kernel(m, "qv_gemm", 128),
            .embed = try d.kernel(m, "qv_embed", 256),
            .layernorm = try d.kernel(m, "qv_layernorm", 256),
            .qkv_rope = try d.kernel(m, "qv_qkv_rope", 128),
            .attention = try d.kernel(m, "qv_attention", 64),
            .bias_gelu = try d.kernel(m, "qv_bias_gelu", 256),
            .bias_residual = try d.kernel(m, "qv_bias_residual", 256),
            .bias_round = try d.kernel(m, "qv_bias_round", 256),
        };
        const h: u64 = c.hidden;
        const ff: u64 = c.intermediate;
        const mg: u64 = c.merged();
        const r = Reader{ .d = d, .ck = ck, .prefix = prefix };
        t.patch_w = try r.patchWeight(c);
        t.patch_b = try r.get("patch_embed.proj.bias", &.{h});
        t.pos = try r.get("pos_embed.weight", &.{ grid_side * grid_side, h });
        t.blocks = try alloc.alloc(Block, c.depth);
        errdefer alloc.free(t.blocks);
        for (t.blocks, 0..) |*b, i| {
            const fields = .{
                .{ "norm1_w", "norm1.weight", .{h} },                .{ "norm1_b", "norm1.bias", .{h} },
                .{ "qkv_w", "attn.qkv.weight", .{ 3 * h, h } },      .{ "qkv_b", "attn.qkv.bias", .{3 * h} },
                .{ "proj_w", "attn.proj.weight", .{ h, h } },        .{ "proj_b", "attn.proj.bias", .{h} },
                .{ "norm2_w", "norm2.weight", .{h} },                .{ "norm2_b", "norm2.bias", .{h} },
                .{ "fc1_w", "mlp.linear_fc1.weight", .{ ff, h } },   .{ "fc1_b", "mlp.linear_fc1.bias", .{ff} },
                .{ "fc2_w", "mlp.linear_fc2.weight", .{ h, ff } },   .{ "fc2_b", "mlp.linear_fc2.bias", .{h} },
            };
            inline for (fields) |f| {
                var nb: [96]u8 = undefined;
                const shape: [f[2].len]u64 = f[2];
                @field(b, f[0]) = try r.get(try std.fmt.bufPrint(&nb, "blocks.{d}.{s}", .{ i, f[1] }), &shape);
            }
        }
        t.norm_w = try r.get("merger.norm.weight", &.{h});
        t.norm_b = try r.get("merger.norm.bias", &.{h});
        t.fc1_w = try r.get("merger.linear_fc1.weight", &.{ mg, mg });
        t.fc1_b = try r.get("merger.linear_fc1.bias", &.{mg});
        t.fc2_w = try r.get("merger.linear_fc2.weight", &.{ c.out_hidden, mg });
        t.fc2_b = try r.get("merger.linear_fc2.bias", &.{c.out_hidden});
        var inv: [head_dim / 4]f32 = undefined;
        for (&inv, 0..) |*f, j| f.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * j)) / (head_dim / 2));
        t.inv = try d.upload(std.mem.sliceAsBytes(&inv));
        const P: usize = max_patches;
        t.px = try d.alloc_(P * patch_width * 2);
        t.x = try d.alloc_(P * h * 2);
        t.h = try d.alloc_(P * h * 2);
        t.acc = try d.alloc_(P * @max(3 * h, ff) * 4);
        t.q = try d.alloc_(P * h * 2);
        t.kk = try d.alloc_(P * h * 2);
        t.v = try d.alloc_(P * h * 2);
        t.att = try d.alloc_(P * h * 2);
        t.mid = try d.alloc_(P * @max(ff, h) * 2);
        t.hw = try d.alloc_(P * 2 * 4);
        t.idx = try d.alloc_(P * 4 * 4);
        t.wts = try d.alloc_(P * 4 * 4);
        t.bytes = d.allocated - before;
        return t;
    }

    /// The host's copy only: the device memory goes with the runner's.
    pub fn deinit(t: *Tower) void {
        alloc.free(t.blocks);
        alloc.destroy(t);
    }

    /// One image's patches ([gh * gw, patch_width] fp32, merge order) to its gh * gw / 4 rows at `out`: fp32
    /// [rows, out_hidden], each value a bf16 one, as the language model's residual stream takes them.
    pub fn encode(t: *Tower, d: *Dev, pixels: []const f32, gh: u32, gw: u32, out: *anyopaque) !void {
        const c = t.c;
        const n: u32 = gh * gw;
        if (n == 0 or gh % merge != 0 or gw % merge != 0 or pixels.len != @as(usize, n) * patch_width) return error.ImageGrid;
        if (n > t.max_patches) return error.ImageTooLarge;
        // the host's part: bf16 patches, each patch's (row, column) and its four position-grid taps
        const px16 = try alloc.alloc(u16, pixels.len);
        defer alloc.free(px16);
        for (px16, pixels) |*o, p| o.* = bf16(p);
        const hw = try alloc.alloc(i32, n * 2);
        defer alloc.free(hw);
        const idx = try alloc.alloc(i32, n * 4);
        defer alloc.free(idx);
        const wts = try alloc.alloc(f32, n * 4);
        defer alloc.free(wts);
        positions(gh, gw, hw, idx, wts);
        try d.copyIn(t.px, std.mem.sliceAsBytes(px16));
        try d.copyIn(t.hw, std.mem.sliceAsBytes(hw));
        try d.copyIn(t.idx, std.mem.sliceAsBytes(idx));
        try d.copyIn(t.wts, std.mem.sliceAsBytes(wts));
        errdefer d.settle();

        const h: i32 = @intCast(c.hidden);
        const ff: i32 = @intCast(c.intermediate);
        const mg: i32 = @intCast(c.merged());
        const T = n / (merge * merge);
        const rows: i32 = @intCast(n);
        const heads: i32 = @intCast(c.heads);
        try t.gemm(d, t.px, t.patch_w, n, c.hidden, patch_width);
        try d.launch(t.k.embed, .{ .x = n }, .{ t.acc, t.patch_b, t.pos, t.idx, t.wts, t.x, h });
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (t.blocks) |b| {
            try d.launch(t.k.layernorm, .{ .x = n }, .{ t.x, b.norm1_w, b.norm1_b, t.h, h, eps });
            try t.gemm(d, t.h, b.qkv_w, n, 3 * c.hidden, c.hidden);
            try d.launch(t.k.qkv_rope, .{ .x = n, .y = c.heads }, .{ t.acc, b.qkv_b, t.hw, t.inv, t.q, t.kk, t.v, rows, heads, @as(i32, head_dim) });
            try d.launch(t.k.attention, .{ .x = (n + 63) / 64, .y = c.heads }, .{ t.q, t.kk, t.v, t.att, rows, heads, @as(i32, 0), rows, scale });
            try t.gemm(d, t.att, b.proj_w, n, c.hidden, c.hidden);
            try d.launch(t.k.bias_residual, .{ .x = n }, .{ t.acc, b.proj_b, t.x, h });
            try d.launch(t.k.layernorm, .{ .x = n }, .{ t.x, b.norm2_w, b.norm2_b, t.h, h, eps });
            try t.gemm(d, t.h, b.fc1_w, n, c.intermediate, c.hidden);
            try d.launch(t.k.bias_gelu, .{ .x = n }, .{ t.acc, b.fc1_b, t.mid, ff, @as(i32, 1) });
            try t.gemm(d, t.mid, b.fc2_w, n, c.hidden, c.intermediate);
            try d.launch(t.k.bias_residual, .{ .x = n }, .{ t.acc, b.fc2_b, t.x, h });
        }
        // the merger: LayerNorm over hidden, then four merge-ordered rows read as one row of `merged`
        try d.launch(t.k.layernorm, .{ .x = n }, .{ t.x, t.norm_w, t.norm_b, t.h, h, eps });
        try t.gemm(d, t.h, t.fc1_w, T, c.merged(), c.merged());
        try d.launch(t.k.bias_gelu, .{ .x = T }, .{ t.acc, t.fc1_b, t.mid, mg, @as(i32, 0) });
        try t.gemm(d, t.mid, t.fc2_w, T, c.out_hidden, c.merged());
        try d.launch(t.k.bias_round, .{ .x = T }, .{ t.acc, t.fc2_b, out, @as(i32, @intCast(c.out_hidden)) });
        try d.sync();
    }

    /// acc [M, N] fp32 = a [M, K] bf16 x w [N, K]^T bf16.
    fn gemm(t: *Tower, d: *Dev, a: *anyopaque, w: *anyopaque, M: u32, N: u32, K: u32) !void {
        try d.launch(t.k.gemm, .{ .x = (M + 127) / 128, .y = (N + 63) / 64 }, .{ a, w, t.acc, @as(i32, @intCast(M)), @as(i32, @intCast(N)), @as(i32, @intCast(K)) });
    }
};

const Reader = struct {
    d: *Dev,
    ck: *const Checkpoint,
    prefix: []const u8,

    fn tensor(r: Reader, name: []const u8) !dev.Tensor {
        var nb: [128]u8 = undefined;
        return r.ck.get(try std.fmt.bufPrint(&nb, "{s}{s}", .{ r.prefix, name }));
    }

    /// A bf16 tensor of `shape`, uploaded as stored.
    fn get(r: Reader, name: []const u8, shape: []const u64) !*anyopaque {
        const tn = try r.tensor(name);
        if (!std.mem.eql(u8, tn.dtype, "BF16") or !std.mem.eql(u64, tn.shape[0..tn.rank], shape)) {
            std.log.err("XPU vision: {s}{s} is {s} {any}; the tower reads BF16 {any}", .{ r.prefix, name, tn.dtype, tn.shape[0..tn.rank], shape });
            return error.UnexpectedVisionTensor;
        }
        return r.d.upload(tn.bytes);
    }

    /// The patch convolution's weight as [hidden, channel, frame, y, x] rows: HF's as stored, MLX's [hidden, frame, y,
    /// x, channel] turned.
    fn patchWeight(r: Reader, c: Config) !*anyopaque {
        const h: u64 = c.hidden;
        const tn = try r.tensor("patch_embed.proj.weight");
        if (std.mem.eql(u8, tn.dtype, "BF16") and std.mem.eql(u64, tn.shape[0..tn.rank], &.{ h, 3, temporal, patch, patch })) return r.d.upload(tn.bytes);
        if (!std.mem.eql(u8, tn.dtype, "BF16") or !std.mem.eql(u64, tn.shape[0..tn.rank], &.{ h, temporal, patch, patch, 3 })) return error.UnexpectedVisionTensor;
        const host = try alloc.alloc(u16, c.hidden * patch_width);
        defer alloc.free(host);
        const stored: []align(1) const u16 = std.mem.bytesAsSlice(u16, tn.bytes);
        for (0..c.hidden) |o| for (0..3) |ch| for (0..temporal) |tt| for (0..patch) |y| for (0..patch) |x| {
            host[o * patch_width + ((ch * temporal + tt) * patch + y) * patch + x] = stored[(((o * temporal + tt) * patch + y) * patch + x) * 3 + ch];
        };
        return r.d.upload(std.mem.sliceAsBytes(host));
    }
};

/// The patches of a gh x gw grid in merge order: each one's (row, column) and its 4 bilinear taps into the 48 x 48
/// learned grid (align_corners), as transformers' fast_pos_embed_interpolate takes them.
pub fn positions(gh: u32, gw: u32, hw: []i32, idx: []i32, wts: []f32) void {
    const bw = gw / merge;
    for (0..@as(usize, gh) * gw) |i| {
        const in_col: u32 = @intCast(i % merge);
        const in_row: u32 = @intCast((i / merge) % merge);
        const block_col: u32 = @intCast((i / (merge * merge)) % bw);
        const block_row: u32 = @intCast(i / (merge * merge * bw));
        const r = block_row * merge + in_row;
        const c = block_col * merge + in_col;
        hw[i * 2] = @intCast(r);
        hw[i * 2 + 1] = @intCast(c);
        const ht = taps(r, gh);
        const wt = taps(c, gw);
        for (0..2) |a| for (0..2) |b| {
            idx[i * 4 + a * 2 + b] = ht.i[a] * grid_side + wt.i[b];
            wts[i * 4 + a * 2 + b] = ht.w[a] * wt.w[b];
        };
    }
}

const Taps = struct { i: [2]i32, w: [2]f32 };

fn taps(index: u32, size: u32) Taps {
    const src = @as(f32, @floatFromInt(index)) * (grid_side - 1) / @as(f32, @floatFromInt(@max(size, 2) - 1));
    const fl = @floor(src);
    var out: Taps = undefined;
    for (0..2) |o| {
        const raw = @as(i64, @intFromFloat(fl)) + @as(i64, @intCast(o));
        out.i[o] = @intCast(std.math.clamp(raw, 0, grid_side - 1));
        out.w[o] = @max(1.0 - @abs(src - fl - @as(f32, @floatFromInt(o))), 0.0);
    }
    return out;
}

/// fp32 to bf16, round to nearest even (torch's .to(bfloat16)).
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

test "patches in merge order, and the taps at the grid's corners" {
    var hw: [24 * 2]i32 = undefined;
    var idx: [24 * 4]i32 = undefined;
    var wts: [24 * 4]f32 = undefined;
    positions(4, 6, &hw, &idx, &wts);
    // block (0, 0) holds (0, 0) (0, 1) (1, 0) (1, 1), then block (0, 1) from column 2
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 0, 1, 1, 0, 2 }, hw[0..10]);
    try std.testing.expectEqual(@as(i32, 0), idx[0]);
    try std.testing.expectEqual(@as(f32, 1), wts[0]);
    try std.testing.expectEqual(@as(i32, 47 * grid_side + 47), idx[23 * 4]); // (3, 5): the last row and column
    try std.testing.expectEqual(@as(f32, 1), wts[23 * 4]);
}
