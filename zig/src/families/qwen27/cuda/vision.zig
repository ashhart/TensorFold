//! The Qwen3.5 vision tower on CUDA as 0.6.6's --vision runs it in bf16: patch embed, 27 rotary blocks, 2x2 merger.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const source = nemotron.source;

const lt = cuda.cublaslt;
const prefix = "vision_tower.";

/// vision_config as the 27B's checkpoint gives it; another tower is refused.
pub const hidden = 1152;
pub const heads = 16;
pub const head_dim = hidden / heads; // 72
pub const depth = 27;
pub const intermediate = 4304;
pub const out_hidden = 5120;
pub const patch = 16;
pub const temporal = 2;
pub const merge = 2;
pub const channels = 3;
pub const patch_width = channels * temporal * patch * patch; // 1536
pub const grid_side = 48; // sqrt(num_position_embeddings 2304)
pub const merged = hidden * merge * merge; // 4608
pub const eps: f32 = 1e-6;
/// The language model's rope_parameters.mrope_section (t, h, w frequencies, interleaved).
pub const mrope_section = [3]u32{ 11, 11, 10 };
/// Tokens the template writes around an image, and the image placeholder it expands.
pub const vision_start: u32 = 248053;
pub const vision_end: u32 = 248054;
pub const image_token: u32 = 248056;
/// One tower call's patches: a 4,096-token image's (0.6.6's MAX_PATCHES).
pub const max_patches = 16384;
const workspace_bytes = 32 << 20;

/// The device, stream and memory kind the tower runs on (the engine's, or a checker's own).
pub const Gpu = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    discrete: bool,

    fn upload(g: Gpu, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try g.d.check(g.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, g.s.handle), "cuMemcpyHtoDAsync");
    }
};

/// One image's patch grid (rows and columns of 16x16 patches, both even).
pub const Grid = struct { h: u32, w: u32 };

const Block = struct { norm1_w: u64, norm1_b: u64, qkv_w: u64, qkv_b: u64, proj_w: u64, proj_b: u64, norm2_w: u64, norm2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };

const Fns = struct {
    embed: cuda.Function,
    layernorm: cuda.Function,
    qkv_rope: cuda.Function,
    attention: cuda.Function,
    bias_gelu: cuda.Function,
    bias_residual: cuda.Function,
    bias_cast: cuda.Function,
    splice: cuda.Function,
    attn_prep: cuda.Function,
};

/// The image tower's weights on the GPU and its kernels; `encode` turns patches into language-width rows.
pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    f: Fns,
    lib: lt.Library,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos: u64 = 0,
    blocks: [depth]Block = undefined,
    merger_norm_w: u64 = 0,
    merger_norm_b: u64 = 0,
    fc1_w: u64 = 0,
    fc1_b: u64 = 0,
    fc2_w: u64 = 0,
    fc2_b: u64 = 0,
    inv: u64 = 0,
    weight_bytes: usize = 0,

    /// The checkpoint's vision_tower tensors (bf16); the patch convolution's weight as [channel, frame, y, x] rows.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: Gpu, dir: []const u8) !*Tower {
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = ops.d, .mod = undefined, .f = undefined, .lib = undefined };
        t.lib = try lt.Library.open();
        errdefer t.lib.close();
        t.mod = try cuda.Module.load(ops.d, cuda.kernels.qwen_vision);
        errdefer t.mod.unload();
        t.f = .{
            .embed = try t.mod.function("tf_qv_embed"),
            .layernorm = try t.mod.function("tf_qv_layernorm"),
            .qkv_rope = try t.mod.function("tf_qv_qkv_rope"),
            .attention = try t.mod.function("tf_qv_attention72"),
            .bias_gelu = try t.mod.function("tf_qv_bias_gelu"),
            .bias_residual = try t.mod.function("tf_qv_bias_residual"),
            .bias_cast = try t.mod.function("tf_qv_bias_cast"),
            .splice = try t.mod.function("tf_qv_splice"),
            .attn_prep = try t.mod.function("tf_qv_attn_prep_mrope"),
        };
        errdefer t.freeBuffers();
        var src = try source.Source.init(gpa, .{ .d = ops.d, .s = ops.s, .discrete = ops.discrete });
        defer src.deinit();
        var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
        defer ck.close();
        try src.add(&ck);
        const L = struct {
            t: *Tower,
            ck: *core.Checkpoint,
            src: *source.Source,
            fn get(l: @This(), name: []const u8, shape: []const usize) !u64 {
                var buf: [96]u8 = undefined;
                const tn = try l.ck.expect(try std.fmt.bufPrint(&buf, prefix ++ "{s}", .{name}), .bf16, shape);
                const ptr = try l.t.alloc(tn.bytes.len);
                try l.src.upload(ptr, tn.bytes);
                return ptr;
            }
        };
        const l: L = .{ .t = t, .ck = &ck, .src = &src };
        { // [hidden, T, P, P, C] as stored (MLX) to [hidden, C, T, P, P]
            const tn = try ck.expect(prefix ++ "patch_embed.proj.weight", .bf16, &.{ hidden, temporal, patch, patch, channels });
            const host = try gpa.alloc(u16, hidden * patch_width);
            defer gpa.free(host);
            const raw = try gpa.alloc(u8, tn.bytes.len);
            defer gpa.free(raw);
            try src.read(raw, tn.bytes);
            const stored: []align(1) const u16 = std.mem.bytesAsSlice(u16, raw);
            for (0..hidden) |o| for (0..channels) |ch| for (0..temporal) |tt| for (0..patch) |y| for (0..patch) |x| {
                host[o * patch_width + ((ch * temporal + tt) * patch + y) * patch + x] = stored[(((o * temporal + tt) * patch + y) * patch + x) * channels + ch];
            };
            t.patch_w = try t.alloc(host.len * 2);
            try src.upload(t.patch_w, std.mem.sliceAsBytes(host));
            try ops.s.synchronize();
        }
        t.patch_b = try l.get("patch_embed.proj.bias", &.{hidden});
        t.pos = try l.get("pos_embed.weight", &.{ grid_side * grid_side, hidden });
        for (&t.blocks, 0..) |*b, i| {
            var pb: [32]u8 = undefined;
            const p = try std.fmt.bufPrint(&pb, "blocks.{d}.", .{i});
            var nb: [96]u8 = undefined;
            const n = struct {
                fn f(buf: []u8, a: []const u8, s: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, s });
                }
            }.f;
            b.norm1_w = try l.get(try n(&nb, p, "norm1.weight"), &.{hidden});
            b.norm1_b = try l.get(try n(&nb, p, "norm1.bias"), &.{hidden});
            b.qkv_w = try l.get(try n(&nb, p, "attn.qkv.weight"), &.{ 3 * hidden, hidden });
            b.qkv_b = try l.get(try n(&nb, p, "attn.qkv.bias"), &.{3 * hidden});
            b.proj_w = try l.get(try n(&nb, p, "attn.proj.weight"), &.{ hidden, hidden });
            b.proj_b = try l.get(try n(&nb, p, "attn.proj.bias"), &.{hidden});
            b.norm2_w = try l.get(try n(&nb, p, "norm2.weight"), &.{hidden});
            b.norm2_b = try l.get(try n(&nb, p, "norm2.bias"), &.{hidden});
            b.fc1_w = try l.get(try n(&nb, p, "mlp.linear_fc1.weight"), &.{ intermediate, hidden });
            b.fc1_b = try l.get(try n(&nb, p, "mlp.linear_fc1.bias"), &.{intermediate});
            b.fc2_w = try l.get(try n(&nb, p, "mlp.linear_fc2.weight"), &.{ hidden, intermediate });
            b.fc2_b = try l.get(try n(&nb, p, "mlp.linear_fc2.bias"), &.{hidden});
        }
        t.merger_norm_w = try l.get("merger.norm.weight", &.{hidden});
        t.merger_norm_b = try l.get("merger.norm.bias", &.{hidden});
        t.fc1_w = try l.get("merger.linear_fc1.weight", &.{ merged, merged });
        t.fc1_b = try l.get("merger.linear_fc1.bias", &.{merged});
        t.fc2_w = try l.get("merger.linear_fc2.weight", &.{ out_hidden, merged });
        t.fc2_b = try l.get("merger.linear_fc2.bias", &.{out_hidden});
        if (ck.unused() != 0) return error.UnusedVisionTensors;
        var inv: [head_dim / 4]f32 = undefined;
        invFreq(&inv);
        t.inv = try t.alloc(@sizeOf(@TypeOf(inv)));
        try src.flush();
        try ops.upload(t.inv, std.mem.asBytes(&inv));
        try ops.s.synchronize();
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.freeBuffers();
        t.mod.unload();
        t.lib.close();
        t.gpa.destroy(t);
    }

    fn freeBuffers(t: *Tower) void {
        for (t.buffers.items) |*b| b.free();
        t.buffers.deinit(t.gpa);
    }

    fn alloc(t: *Tower, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(t.d, len);
        try t.buffers.append(t.gpa, b);
        t.weight_bytes += len;
        return b.ptr;
    }

    /// The scratch one tower call of `n` patches needs, in bytes (the patches' bf16 copy included).
    pub fn scratchBytes(n: usize) usize {
        return n * patch_width * 2 + // patches
            2 * n * hidden * 2 + // x, h
            n * @max(3 * hidden, intermediate) * 4 + // fp32 Linear out
            3 * n * hidden * 2 + // q, k, v
            n * hidden * 2 + // attention out
            n * @max(intermediate, hidden) * 2 + // gelu rows
            n * (2 + 4) * 4 + n * 4 * 4 + // hw, idx, weights
            workspace_bytes;
    }

    /// Images' patches (fp32, merge order) to merged bf16 [sum(h * w) / 4, out_hidden] at `out`, each image on its own.
    pub fn encode(t: *Tower, ops: Gpu, pixels: []const f32, grids: []const Grid, out: u64) !void {
        var n: usize = 0;
        for (grids) |g| {
            if (g.h == 0 or g.w == 0 or g.h % merge != 0 or g.w % merge != 0) return error.BadImageGrid;
            n += @as(usize, g.h) * g.w;
        }
        if (n == 0 or n > max_patches or pixels.len != n * patch_width) return error.BadImagePatches;
        const gpa = t.gpa;
        // host side: bf16 patches, each patch's (row, column) and its 4 position-grid taps
        const px16 = try gpa.alloc(u16, n * patch_width);
        defer gpa.free(px16);
        for (px16, pixels) |*o, v| o.* = bf16(v);
        const hw = try gpa.alloc(i32, n * 2);
        defer gpa.free(hw);
        const idx = try gpa.alloc(i32, n * 4);
        defer gpa.free(idx);
        const wts = try gpa.alloc(f32, n * 4);
        defer gpa.free(wts);
        var at: usize = 0;
        for (grids) |g| {
            positions(g, hw[at * 2 ..][0 .. @as(usize, g.h) * g.w * 2], idx[at * 4 ..][0 .. @as(usize, g.h) * g.w * 4], wts[at * 4 ..][0 .. @as(usize, g.h) * g.w * 4]);
            at += @as(usize, g.h) * g.w;
        }
        var scratch = try cuda.DeviceBuffer.alloc(t.d, scratchBytes(n));
        defer scratch.free();
        var cur: u64 = scratch.ptr;
        const take = struct {
            fn f(p: *u64, len: usize) u64 {
                const r = p.*;
                p.* += std.mem.alignForward(usize, len, 256);
                return r;
            }
        }.f;
        const d_px = take(&cur, n * patch_width * 2);
        const x = take(&cur, n * hidden * 2);
        const h = take(&cur, n * hidden * 2);
        const acc = take(&cur, n * @max(3 * hidden, intermediate) * 4);
        const q = take(&cur, n * hidden * 2);
        const k = take(&cur, n * hidden * 2);
        const v = take(&cur, n * hidden * 2);
        const att = take(&cur, n * hidden * 2);
        const mid = take(&cur, n * @max(intermediate, hidden) * 2);
        const d_hw = take(&cur, n * 2 * 4);
        const d_idx = take(&cur, n * 4 * 4);
        const d_wts = take(&cur, n * 4 * 4);
        const ws = cur;
        const ws_len = scratch.ptr + scratch.len - cur;
        try ops.upload(d_px, std.mem.sliceAsBytes(px16));
        try ops.upload(d_hw, std.mem.sliceAsBytes(hw));
        try ops.upload(d_idx, std.mem.sliceAsBytes(idx));
        try ops.upload(d_wts, std.mem.sliceAsBytes(wts));

        var mm: Gemms = .{ .lib = &t.lib };
        defer mm.deinit();
        const m4 = n / (merge * merge);
        try mm.run(n, hidden, patch_width, d_px, t.patch_w, acc, ws, ws_len, ops.s);
        try t.go(ops, t.f.embed, n, 1, 256, .{ acc, t.patch_b, t.pos, d_idx, d_wts, x, int(hidden) });
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (t.blocks) |b| {
            try t.go(ops, t.f.layernorm, n, 1, 256, .{ x, b.norm1_w, b.norm1_b, h, int(hidden), eps });
            try mm.run(n, 3 * hidden, hidden, h, b.qkv_w, acc, ws, ws_len, ops.s);
            try t.go(ops, t.f.qkv_rope, n, heads, 128, .{ acc, b.qkv_b, d_hw, t.inv, q, k, v, int(n), int(heads), int(head_dim) });
            var s0: usize = 0;
            for (grids) |g| {
                const len = @as(usize, g.h) * g.w;
                try t.go(ops, t.f.attention, (len + 63) / 64, heads, 64, .{ q, k, v, att, int(n), int(heads), int(s0), int(len), scale });
                s0 += len;
            }
            try mm.run(n, hidden, hidden, att, b.proj_w, acc, ws, ws_len, ops.s);
            try t.go(ops, t.f.bias_residual, n, 1, 256, .{ acc, b.proj_b, x, int(hidden) });
            try t.go(ops, t.f.layernorm, n, 1, 256, .{ x, b.norm2_w, b.norm2_b, h, int(hidden), eps });
            try mm.run(n, intermediate, hidden, h, b.fc1_w, acc, ws, ws_len, ops.s);
            try t.go(ops, t.f.bias_gelu, n, 1, 256, .{ acc, b.fc1_b, mid, int(intermediate), int(1) });
            try mm.run(n, hidden, intermediate, mid, b.fc2_w, acc, ws, ws_len, ops.s);
            try t.go(ops, t.f.bias_residual, n, 1, 256, .{ acc, b.fc2_b, x, int(hidden) });
        }
        // merger: LayerNorm over 1152, four merge-ordered rows read as one of 4608
        try t.go(ops, t.f.layernorm, n, 1, 256, .{ x, t.merger_norm_w, t.merger_norm_b, h, int(hidden), eps });
        try mm.run(m4, merged, merged, h, t.fc1_w, acc, ws, ws_len, ops.s);
        try t.go(ops, t.f.bias_gelu, m4, 1, 256, .{ acc, t.fc1_b, mid, int(merged), int(0) });
        try mm.run(m4, out_hidden, merged, mid, t.fc2_w, acc, ws, ws_len, ops.s);
        try t.go(ops, t.f.bias_cast, m4, 1, 256, .{ acc, t.fc2_b, out, int(out_hidden) });
        try ops.s.synchronize();
    }

    /// Splices image rows into a chunk's embeddings `x`: row r takes `features` row src[r] where src[r] >= 0.
    pub fn splice(t: *Tower, ops: Gpu, x: u64, features: u64, src: u64, rows: usize) !void {
        try t.go(ops, t.f.splice, rows, 1, 256, .{ x, features, src, int(out_hidden) });
    }

    /// glue.attn_prep with three position rows (pos [3][rows], device int32): normed, M-RoPE-rotated q and k.
    pub fn attnPrep(t: *Tower, ops: Gpu, qg: u64, key: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, a: struct { heads: usize, kv_heads: usize, dim: usize, half: usize }, eps_: f32) !void {
        try t.go(ops, t.f.attn_prep, rows, a.heads + a.kv_heads, @intCast(a.dim), .{ qg, key, qn, kn, pos, inv, qout, kout, eps_, int(a.heads), int(a.kv_heads), int(a.dim), int(a.half), int(rows), int(mrope_section[1]), int(mrope_section[2]) });
    }

    fn go(t: *Tower, ops: Gpu, f: cuda.Function, gx: usize, gy: usize, block: u32, args: anytype) !void {
        _ = t;
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy) }, .block = .{ .x = block } }, ops.s, &a);
    }
};

/// cuBLASLt plans for the shapes one tower call runs, made on first use.
const Gemms = struct {
    lib: *const lt.Library,
    plans: [8]?struct { m: usize, n: usize, k: usize, lin: lt.Linear } = @splat(null),

    fn run(g: *Gemms, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64, ws: u64, ws_len: usize, s: cuda.Stream) !void {
        for (&g.plans) |*p| {
            if (p.*) |*have| {
                if (have.m == m and have.n == n and have.k == k) return have.lin.run(x, w, out, ws, ws_len, s.handle);
                continue;
            }
            p.* = .{ .m = m, .n = n, .k = k, .lin = try lt.Linear.init(g.lib, m, n, k, .f32, ws_len) };
            return p.*.?.lin.run(x, w, out, ws, ws_len, s.handle);
        }
        return error.TooManyShapes;
    }

    fn deinit(g: *Gemms) void {
        for (&g.plans) |*p| if (p.*) |*have| have.lin.deinit();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

/// fp32 to bf16, round to nearest even (torch's .to(bfloat16)).
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// The axial rotary's frequencies: 1 / 10000 ** (i / (head_dim / 2)) for even i, in fp32 as torch computes them.
pub fn invFreq(out: []f32) void {
    const dim: f32 = @floatFromInt(head_dim / 2);
    for (out, 0..) |*f, j| {
        const e: f32 = @as(f32, @floatFromInt(2 * j)) / dim;
        f.* = 1.0 / std.math.pow(f32, 10000.0, e);
    }
}

/// One image's patches in merge order: each one's (row, column) for the rotary and 4 bilinear taps into the 48x48 grid.
pub fn positions(g: Grid, hw: []i32, idx: []i32, wts: []f32) void {
    const bw = g.w / merge;
    const n = @as(usize, g.h) * g.w;
    for (0..n) |i| {
        const in_col: u32 = @intCast(i % merge);
        const in_row: u32 = @intCast((i / merge) % merge);
        const block_col: u32 = @intCast((i / (merge * merge)) % bw);
        const block_row: u32 = @intCast(i / (merge * merge * bw));
        const r = block_row * merge + in_row;
        const c = block_col * merge + in_col;
        hw[i * 2] = @intCast(r);
        hw[i * 2 + 1] = @intCast(c);
        const ht = taps(r, g.h);
        const wt = taps(c, g.w);
        for (0..2) |a| for (0..2) |b| {
            idx[i * 4 + a * 2 + b] = ht.i[a] * grid_side + wt.i[b];
            wts[i * 4 + a * 2 + b] = ht.w[a] * wt.w[b];
        };
    }
}

const Taps = struct { i: [2]i32, w: [2]f32 };

fn taps(index: u32, size: u32) Taps {
    const side_m1: f32 = grid_side - 1;
    const denom: f32 = @floatFromInt(@max(size, 2) - 1);
    const src = @as(f32, @floatFromInt(index)) * side_m1 / denom;
    const fl = @floor(src);
    var out: Taps = undefined;
    for (0..2) |o| {
        const raw: i32 = @as(i32, @intFromFloat(fl)) + @as(i32, @intCast(o));
        out.i[o] = std.math.clamp(raw, 0, grid_side - 1);
        const dist = @abs(src - fl - @as(f32, @floatFromInt(o)));
        out.w[o] = @max(1.0 - dist, 0.0);
    }
    return out;
}

test "bilinear taps match align_corners resampling at the ends and middle" {
    const a = taps(0, 10);
    try std.testing.expectEqual(@as(i32, 0), a.i[0]);
    try std.testing.expectEqual(@as(f32, 1), a.w[0]);
    const b = taps(9, 10);
    try std.testing.expectEqual(@as(i32, 47), b.i[0]);
    try std.testing.expectEqual(@as(f32, 1), b.w[0]);
}

/// An image prompt's rows for its prompt pass (0.6.6's EncodedVision): tower rows, feature rows, positions.
pub const Media = struct {
    gpa: std.mem.Allocator,
    tower: *Tower,
    features: cuda.DeviceBuffer, // bf16 [image rows, out_hidden]
    src: []i32, // [prompt]: feature row or -1
    pos: []i32, // [3][prompt]: t, h, w
    delta: i32,
    dev: cuda.DeviceBuffer, // a chunk's src and [3][rows] positions

    /// Encodes `images` and lays out the prompt's positions as transformers' get_rope_index does (see `layout`).
    pub fn init(gpa: std.mem.Allocator, tower: *Tower, ops: Gpu, prompt: []const u32, images: anytype, chunk_rows: usize) !*Media {
        const n = prompt.len;
        const m = try gpa.create(Media);
        errdefer gpa.destroy(m);
        var rows: usize = 0;
        for (images) |img| rows += img.tokens;
        m.* = .{ .gpa = gpa, .tower = tower, .features = try cuda.DeviceBuffer.alloc(ops.d, rows * out_hidden * 2), .src = &.{}, .pos = &.{}, .delta = 0, .dev = undefined };
        errdefer m.features.free();
        m.dev = try cuda.DeviceBuffer.alloc(ops.d, chunk_rows * 4 * 4);
        errdefer m.dev.free();
        m.src = try gpa.alloc(i32, n);
        errdefer gpa.free(m.src);
        m.pos = try gpa.alloc(i32, 3 * n);
        errdefer gpa.free(m.pos);
        @memset(m.src, -1);
        // the tower: runs of whole images within one call's patches, in prompt order
        var done: usize = 0; // feature rows written
        var i: usize = 0;
        while (i < images.len) {
            var j = i;
            var patches: usize = 0;
            while (j < images.len and (j == i or patches + images[j].gh * images[j].gw <= max_patches)) : (j += 1) patches += images[j].gh * images[j].gw;
            const px = try gpa.alloc(f32, patches * patch_width);
            defer gpa.free(px);
            const grids = try gpa.alloc(Grid, j - i);
            defer gpa.free(grids);
            var at: usize = 0;
            for (images[i..j], grids) |img, *g| {
                if (img.pixels.len != @as(usize, img.gh) * img.gw * patch_width or img.tokens * merge * merge != img.gh * img.gw) return error.BadImagePatches;
                @memcpy(px[at..][0..img.pixels.len], img.pixels);
                at += img.pixels.len;
                g.* = .{ .h = img.gh, .w = img.gw };
            }
            try tower.encode(ops, px, grids, m.features.ptr + done * out_hidden * 2);
            for (images[i..j]) |img| done += img.tokens;
            i = j;
        }
        m.delta = try layout(prompt, images, m.src, m.pos);
        return m;
    }

    pub fn deinit(m: *Media) void {
        m.features.free();
        m.dev.free();
        m.gpa.free(m.src);
        m.gpa.free(m.pos);
        m.gpa.destroy(m);
    }

    /// Whether rows [p0, p0 + w) hold an image row.
    pub fn hasImage(m: *const Media, p0: usize, w: usize) bool {
        for (m.src[p0..][0..w]) |v| if (v >= 0) return true;
        return false;
    }
};

/// The prompt's rotary positions ([3][n]) and feature rows (src, -1 for text) per get_rope_index; returns rope_delta.
pub fn layout(prompt: []const u32, images: anytype, src: []i32, pos: []i32) !i32 {
    const n = prompt.len;
    @memset(src, -1);
    var next: i32 = 0;
    var cursor: usize = 0;
    var row: i32 = 0;
    for (images) |img| {
        const begin: usize = img.at;
        const h: usize = img.gh / merge;
        const w: usize = img.gw / merge;
        const end = begin + h * w;
        if (begin < cursor or begin == 0 or end >= n or prompt[begin - 1] != vision_start or prompt[end] != vision_end) return error.BadImagePlaceholders;
        for (prompt[begin..end]) |t| if (t != image_token) return error.BadImagePlaceholders;
        for (cursor..begin) |r| {
            const v = next + @as(i32, @intCast(r - cursor));
            for (0..3) |a| pos[a * n + r] = v;
        }
        const base = next + @as(i32, @intCast(begin - cursor));
        for (0..h * w) |k| {
            const r = begin + k;
            pos[r] = base;
            pos[n + r] = base + @as(i32, @intCast(k / w));
            pos[2 * n + r] = base + @as(i32, @intCast(k % w));
            src[r] = row;
            row += 1;
        }
        next = base + @as(i32, @intCast(@max(h, w)));
        cursor = end;
    }
    for (cursor..n) |r| {
        const v = next + @as(i32, @intCast(r - cursor));
        for (0..3) |a| pos[a * n + r] = v;
    }
    return next - @as(i32, @intCast(cursor));
}

test "media positions: text counts up, an image sits at one position with its grid, text resumes past its side" {
    const I = struct { gh: u32, gw: u32, at: u32, tokens: u32 };
    const prompt = [_]u32{ 7, vision_start, image_token, image_token, image_token, image_token, image_token, image_token, vision_end, 9 };
    var src: [10]i32 = undefined;
    var pos: [30]i32 = undefined;
    const delta = try layout(&prompt, &[_]I{.{ .gh = 4, .gw = 6, .at = 2, .tokens = 6 }}, &src, &pos);
    try std.testing.expectEqual(@as(i32, -3), delta);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, pos[0..10]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, pos[10..20]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, pos[20..30]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 1, 2, 3, 4, 5, -1, -1 }, &src);
}
