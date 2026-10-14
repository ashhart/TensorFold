//! LFM2-VL's pictures on CUDA: Siglip2VisionModel's NaFlex tower, then Lfm2VlMultiModalProjector a tile at a time.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Attention = @import("attention.zig").Attention;
const VisionWeights = @import("weights.zig").VisionWeights;
const image = @import("image.zig");
const Allocator = std.mem.Allocator;
const Linear = cuda.cublaslt.Linear;

const threads = 256; // lfm2_vision.cu's NT
const norm_most = 8 * threads; // NT * MAXE
const patches_most = 1024; // a 512-pixel tile, or a thumbnail of at most 256 tokens
const tiles_most = image.max_grid + 1; // max_tiles and the thumbnail
const workspace_bytes = 32 << 20;
const px_bytes = 2 * tiles_most * patches_most * image.patch_width; // a picture's bf16 patches, 256-byte aligned

/// What tf_lfm2v_pointwise applies to a Linear's output: nothing, torch's tanh GELU, or its exact (erf) GELU.
pub const Act = enum(i32) { none = 0, gelu_tanh = 1, gelu = 2 };

pub const Kernels = struct {
    mod: cuda.Module,
    layer_norm_fn: cuda.Function,
    pointwise_fn: cuda.Function,
    bias_fn: cuda.Function,
    unshuffle_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.lfm2_vision);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .layer_norm_fn = try mod.function("tf_lfm2v_layer_norm"),
            .pointwise_fn = try mod.function("tf_lfm2v_pointwise"),
            .bias_fn = try mod.function("tf_lfm2v_bias"),
            .unshuffle_fn = try mod.function("tf_lfm2v_unshuffle"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// y = LayerNorm(x) * w + b over rows up to 2048 wide.
    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        if (width > norm_most) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, b, y }) |p| a.add(p);
        a.add(int(width));
        a.add(eps);
        try go(k.layer_norm_fn, s, rows, 1, &a);
    }

    /// o [count] = x (a Linear's bf16 output) through `act`, then plus r when given; x or r may be o.
    pub fn pointwise(k: *const Kernels, s: cuda.Stream, x: u64, r: ?u64, o: u64, count: usize, act: Act) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, r orelse 0, o }) |p| a.add(p);
        a.add(@as(i64, @intCast(count)));
        a.add(@backingInt(act));
        try go(k.pointwise_fn, s, blocks(count, threads), 1, &a);
    }

    /// o [rows, n] = bf16(acc + b), acc f32.
    pub fn bias(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, o: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ acc, b, o }) |p| a.add(p);
        a.add(int(n));
        try go(k.bias_fn, s, rows, blocks(n, threads), &a);
    }

    /// y [h/2 * w/2, 4c] = the projector's pixel unshuffle of x [h * w, c], an even grid.
    pub fn unshuffle(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, h: usize, w: usize, c: usize) !void {
        if (h % 2 != 0 or w % 2 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(w));
        a.add(int(c));
        try go(k.unshuffle_fn, s, h * w / 4, 1, &a);
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, by: usize) usize {
    return (n + by - 1) / by;
}

fn go(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, a: *cuda.Args) !void {
    if (x == 0) return;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = 1 }, .block = .{ .x = threads } }, s, a);
}

/// torch Linears through cuBLASLt: a plan kept a shape, one workspace.
pub const Gemms = struct {
    gpa: Allocator,
    lt: cuda.cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([3]u64, Linear) = .empty,

    pub fn init(gpa: Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, f32 accumulation, `out` rows.
    pub fn run(g: *Gemms, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) !void {
        try (try g.plan(rows, n, k, out, .none)).run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
    }

    /// Whether the shape's own algorithm carries the bias in cuBLASLt's epilogue; else `linear` sums through f32.
    pub fn fused(g: *Gemms, rows: usize, n: usize, k: usize) !bool {
        return (try g.plan(rows, n, k, .bf16, .bias)).epilogue == .bias;
    }

    fn plan(g: *Gemms, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType, epilogue: cuda.cublaslt.Epilogue) !*const Linear {
        const key = [3]u64{ rows, n, k | @as(u64, @intCast(@backingInt(out))) << 40 | @as(u64, @backingInt(epilogue)) << 48 };
        const got = try g.plans.getOrPut(g.gpa, key);
        if (!got.found_existing) got.value_ptr.* = Linear.initEpilogue(&g.lt, rows, n, k, out, workspace_bytes, epilogue) catch |e| {
            g.plans.removeByPtr(got.key_ptr);
            return e;
        };
        return got.value_ptr;
    }
};

/// y [rows, n] = bf16(x . w^T + b), the bias in cuBLASLt's epilogue, or else through f32 `acc` and tf_lfm2v_bias.
pub fn linear(g: *Gemms, k: *const Kernels, s: cuda.Stream, acc: u64, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, kk: usize) !void {
    const p = try g.plan(rows, n, kk, .bf16, .bias);
    if (p.epilogue == .bias) return p.runBias(x, w, b, y, g.workspace.ptr, workspace_bytes, s.handle);
    try g.run(s, x, w, acc, rows, n, kk, .f32);
    try k.bias(s, acc, b, y, rows, n);
}

/// a [rows, H D] = SDPA over one tile's rows (no mask), from qkv [rows, 3 H D] with its biases added.
pub fn attention(at: *const Attention, s: cuda.Stream, qkv: u64, a: u64, rows: usize, heads: usize, dim: usize) !void {
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
    try at.full(s, qkv, a, rows, heads, dim, scale);
}

pub const Vision = struct {
    c: config.Vision,
    text_hidden: usize,
    w: *const VisionWeights,
    k: Kernels,
    at: Attention,
    g: Gemms,
    position: []f32, // the position table [side, side, hidden] on the host, resized for each tile's grid
    scratch: cuda.DeviceBuffer,
    staging: cuda.HostBuffer, // pinned: a picture's bf16 patches, then its tiles' position rows
    copied: cuda.Event, // recorded after the staging's copies: refilling it waits for them, not for the tower
    b: Buffers,

    const Buffers = struct { px: u64, pos: u64, h: u64, x: u64, wide: u64, a: u64, acc: u64 };

    /// Kernels, attention, cuBLASLt and scratch for one picture of up to 11 tiles of 1024 patches.
    pub fn init(gpa: Allocator, d: *const cuda.Driver, cfg: *const config.Config, w: *const VisionWeights) !Vision {
        const c = cfg.vision orelse return error.NoVisionTower;
        const dim = c.hidden / c.heads;
        const side = std.math.sqrt(c.num_patches);
        if (dim * c.heads != c.hidden or c.hidden > norm_most or c.downsample != 2 or side * side != c.num_patches or c.channels * c.patch * c.patch != image.patch_width) return error.VisionShape;
        var k = try Kernels.load(d);
        errdefer k.unload();
        var attn = try Attention.load(d);
        errdefer attn.unload();
        var g = try Gemms.init(gpa, d);
        errdefer g.deinit();
        const position = try gpa.alloc(f32, c.num_patches * c.hidden);
        errdefer gpa.free(position);
        const table = try gpa.alloc(u16, position.len);
        defer gpa.free(table);
        try d.check(d.api.cuMemcpyDtoH_v2(table.ptr, w.position, 2 * table.len), "cuMemcpyDtoH");
        for (position, table) |*p, t| p.* = @bitCast(@as(u32, t) << 16);
        if (!finite(position)) {
            std.log.err("the vision tower's position embedding holds a non-finite value, which its resize cannot carry", .{});
            return error.UnreadableLfm2Checkpoint;
        }
        const n = patches_most;
        const merged = 4 * c.hidden;
        const staged = [_]usize{ px_bytes, 2 * tiles_most * n * c.hidden };
        // bf16 rows: residual, normed and Linear outputs, wide Linears' outputs, the attention's; f32 unfused products
        const sizes = staged ++ [_]usize{
            2 * n * c.hidden,
            2 * n * @max(c.hidden, c.projector_hidden / 4),
            2 * n * @max(3 * c.hidden, c.ffn, merged / 4),
            2 * n * c.hidden,
            4 * n * @max(3 * c.hidden, c.ffn, @max(c.projector_hidden, cfg.hidden) / 4),
        };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var at: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &at) |sz, *p| {
            p.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        var staging = try cuda.HostBuffer.alloc(d, px_bytes + staged[1]);
        errdefer staging.free();
        return .{
            .c = c,
            .text_hidden = cfg.hidden,
            .w = w,
            .k = k,
            .at = attn,
            .g = g,
            .position = position,
            .scratch = scratch,
            .staging = staging,
            .copied = try cuda.Event.init(d, false),
            .b = .{ .px = at[0], .pos = at[1], .h = at[2], .x = at[3], .wide = at[4], .a = at[5], .acc = at[6] },
        };
    }

    pub fn deinit(v: *Vision) void {
        v.copied.deinit();
        v.staging.free();
        v.scratch.free();
        v.g.gpa.free(v.position);
        v.g.deinit();
        v.at.unload();
        v.k.unload();
    }

    /// One picture's features, bf16 [tokens, text hidden] at `out`, tiles row by row then the thumbnail; the rows.
    pub fn features(v: *Vision, s: cuda.Stream, picture: *const image.Picture, out: u64) !usize {
        const c = v.c;
        if (picture.tiles.len > tiles_most) return error.TileShape;
        var n: usize = 0;
        for (picture.tiles) |t| {
            if (t.h * t.w > patches_most or t.h % 2 != 0 or t.w % 2 != 0) return error.TileShape;
            n += t.h * t.w;
        }
        try v.copied.synchronize(); // the staging may still feed the last picture's copies, never its tiles' work
        const host = v.staging.slice(u16);
        const px = host[0 .. n * image.patch_width];
        for (px, picture.patches[0..px.len]) |*o, p| o.* = bf16(p);
        const pos = host[px_bytes / 2 ..][0 .. n * c.hidden];
        var at: usize = 0;
        for (picture.tiles, 0..) |t, i| {
            const rows = pos[at * c.hidden ..][0 .. t.h * t.w * c.hidden];
            if (i > 0 and std.meta.eql(t, picture.tiles[i - 1])) @memcpy(rows, pos[(at - t.h * t.w) * c.hidden ..][0..rows.len]) else try v.resizePositions(t, rows);
            at += t.h * t.w;
        }
        try v.scratch.uploadAsync(v.b.px - v.scratch.ptr, std.mem.sliceAsBytes(px), s.handle);
        errdefer s.synchronize() catch {}; // unrecorded copies may still read the staging the next call refills
        try v.scratch.uploadAsync(v.b.pos - v.scratch.ptr, std.mem.sliceAsBytes(pos), s.handle);
        try v.copied.record(s);
        var row: usize = 0;
        at = 0;
        for (picture.tiles) |t| {
            try v.tile(s, t, v.b.px + 2 * at * image.patch_width, v.b.pos + 2 * at * c.hidden, out + 2 * row * v.text_hidden);
            at += t.h * t.w;
            row += t.h * t.w / 4;
        }
        return row;
    }

    /// The tower and projector over one tile's patches and position rows: its tokens' features at `out`.
    fn tile(v: *Vision, s: cuda.Stream, t: image.Grid, px: u64, pos: u64, out: u64) !void {
        const c = v.c;
        const b = v.b;
        const k = &v.k;
        const g = &v.g;
        const w = v.w;
        const n = t.h * t.w;
        const hd = c.hidden;
        try linear(g, k, s, b.acc, px, w.patch, w.patch_bias, b.x, n, hd, image.patch_width);
        try k.pointwise(s, b.x, pos, b.h, n * hd, .none);
        for (w.layers) |l| {
            try k.layerNorm(s, b.h, l.norm1, l.norm1_bias, b.x, n, hd, c.eps);
            try linear(g, k, s, b.acc, b.x, l.qkv, l.qkv_bias, b.wide, n, 3 * hd, hd);
            try attention(&v.at, s, b.wide, b.a, n, c.heads, hd / c.heads);
            try linear(g, k, s, b.acc, b.a, l.out, l.out_bias, b.x, n, hd, hd);
            try k.pointwise(s, b.x, b.h, b.h, n * hd, .none);
            try k.layerNorm(s, b.h, l.norm2, l.norm2_bias, b.x, n, hd, c.eps);
            try linear(g, k, s, b.acc, b.x, l.fc1, l.fc1_bias, b.wide, n, c.ffn, hd);
            try k.pointwise(s, b.wide, null, b.wide, n * c.ffn, .gelu_tanh);
            try linear(g, k, s, b.acc, b.wide, l.fc2, l.fc2_bias, b.x, n, hd, c.ffn);
            try k.pointwise(s, b.x, b.h, b.h, n * hd, .none);
        }
        try k.layerNorm(s, b.h, w.post_norm, w.post_norm_bias, b.x, n, hd, c.eps);
        const tokens = n / 4;
        try k.unshuffle(s, b.x, b.wide, t.h, t.w, hd);
        try linear(g, k, s, b.acc, b.wide, w.linear_1, w.linear_1_bias, b.x, tokens, c.projector_hidden, 4 * hd);
        try k.pointwise(s, b.x, null, b.x, tokens * c.projector_hidden, .gelu);
        try linear(g, k, s, b.acc, b.x, w.linear_2, w.linear_2_bias, out, tokens, v.text_hidden, c.projector_hidden);
    }

    /// The position table resized to grid `t` (Siglip2VisionEmbeddings.resize_positional_embeddings), bf16 rows.
    fn resizePositions(v: *Vision, t: image.Grid, out: []u16) !void {
        const side = std.math.sqrt(v.c.num_patches);
        const gpa = v.g.gpa;
        const resized = try gpa.alloc(f32, out.len);
        defer gpa.free(resized);
        try resize(gpa, v.position, side, v.c.hidden, t.h, t.w, resized);
        for (out, resized) |*o, x| o.* = bf16(x);
    }
};

/// cvt.rn.bf16.f32 of a finite value.
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Every value finite: the resize skips zero weights, so a NaN or infinity there must not be dropped silently.
fn finite(values: []const f32) bool {
    for (values) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

test "a position table with a NaN or an infinity is refused" {
    var table = [_]f32{ 0.5, -1.0, 0.0, 3.0 };
    try std.testing.expect(finite(&table));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        table[2] = bad;
        try std.testing.expect(!finite(&table));
    }
}

/// torch's antialiased bilinear weights from `from` samples to `to` (align_corners false), [to][from].
fn weights(a: Allocator, from: usize, to: usize) ![]f64 {
    const out = try a.alloc(f64, to * from);
    @memset(out, 0);
    const scale = @as(f64, @floatFromInt(from)) / @as(f64, @floatFromInt(to));
    const support = @max(scale, 1.0);
    const inv = 1.0 / @max(scale, 1.0);
    for (0..to) |i| {
        const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
        const lo: usize = @intFromFloat(@max(center - support + 0.5, 0));
        const hi: usize = @min(@as(usize, @intFromFloat(center + support + 0.5)), from);
        const row = out[i * from ..][0..from];
        var total: f64 = 0;
        for (lo..hi) |j| {
            row[j] = @max(0, 1 - @abs((@as(f64, @floatFromInt(j)) - center + 0.5) * inv));
            total += row[j];
        }
        if (total != 0) for (row[lo..hi]) |*x| {
            x.* /= total;
        };
    }
    return out;
}

/// F.interpolate(bilinear, antialias) [side, side, c] -> [h, w, c]: a column at a time, zero weights skipped.
fn resize(a: Allocator, table: []const f32, side: usize, c: usize, h: usize, w: usize, out: []f32) !void {
    const wy = try weights(a, side, h);
    defer a.free(wy);
    const wx = try weights(a, side, w);
    defer a.free(wx);
    const across = try a.alloc(f64, side * c);
    defer a.free(across);
    const sum = try a.alloc(f64, c);
    defer a.free(sum);
    for (0..w) |x| {
        @memset(across, 0);
        for (0..side) |i| for (wx[x * side ..][0..side], 0..) |f, j| {
            if (f == 0) continue;
            for (across[i * c ..][0..c], table[(i * side + j) * c ..][0..c]) |*o, v| o.* += f * v;
        };
        for (0..h) |y| {
            @memset(sum, 0);
            for (wy[y * side ..][0..side], 0..) |f, i| {
                if (f == 0) continue;
                for (sum, across[i * c ..][0..c]) |*o, v| o.* += f * v;
            }
            for (out[(y * w + x) * c ..][0..c], sum) |*o, v| o.* = @floatCast(v);
        }
    }
}

test "position rows resize as torch's antialiased bilinear interpolate" {
    const a = std.testing.allocator;
    const side = 16;
    const c = 3;
    var table: [side * side * c]f32 = undefined;
    for (0..side) |i| for (0..side) |j| for (0..c) |k| {
        table[(i * side + j) * c + k] = @floatCast(@sin(0.37 * @as(f64, @floatFromInt(i)) + 0.11 * @as(f64, @floatFromInt(j)) + 0.05 * @as(f64, @floatFromInt(k))));
    };
    // torch 2.13 F.interpolate on the CPU: (h, w) then (y, x, channel, value) picks
    const Pick = struct { y: usize, x: usize, k: usize, v: f32 };
    const cases = [_]struct { h: usize, w: usize, picks: [4]Pick }{
        .{ .h = 32, .w = 32, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.0 }, .{ .y = 31, .x = 31, .k = 2, .v = 0.8504366278648376 }, .{ .y = 16, .x = 10, .k = 1, .v = -0.290686696767807 }, .{ .y = 1, .x = 30, .k = 0, .v = 0.9759851694107056 } } },
        .{ .h = 8, .w = 34, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.25130489468574524 }, .{ .y = 7, .x = 33, .k = 2, .v = 0.6617922186851501 }, .{ .y = 4, .x = 11, .k = 1, .v = -0.5310338735580444 }, .{ .y = 1, .x = 32, .k = 0, .v = 0.5272247195243835 } } },
        .{ .h = 22, .w = 14, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.018296388909220695 }, .{ .y = 21, .x = 13, .k = 2, .v = 0.8399547338485718 }, .{ .y = 11, .x = 4, .k = 1, .v = -0.31715455651283264 }, .{ .y = 1, .x = 12, .k = 0, .v = 0.9701255559921265 } } },
        .{ .h = 4, .w = 100, .picks = .{ .{ .y = 0, .x = 0, .k = 0, .v = 0.559954822063446 }, .{ .y = 3, .x = 99, .k = 2, .v = 0.27992990612983704 }, .{ .y = 2, .x = 33, .k = 1, .v = -0.6748089790344238 }, .{ .y = 1, .x = 98, .k = 0, .v = -0.4268251955509186 } } },
    };
    for (cases) |case| {
        const out = try a.alloc(f32, case.h * case.w * c);
        defer a.free(out);
        try resize(a, &table, side, c, case.h, case.w, out);
        for (case.picks) |p| try std.testing.expectApproxEqAbs(p.v, out[(p.y * case.w + p.x) * c + p.k], 1e-6);
    }
}

test "position rows: a column at a time, zero weights skipped, bit for bit as every weight in row-major passes" {
    const a = std.testing.allocator;
    const side = 16;
    const c = 40;
    var prng = std.Random.DefaultPrng.init(11);
    const table = try a.alloc(f32, side * side * c);
    defer a.free(table);
    for (table) |*t| t.* = prng.random().floatNorm(f32) * 0.03;
    for ([_][2]usize{ .{ 32, 32 }, .{ 26, 38 }, .{ 22, 14 }, .{ 8, 34 }, .{ 4, 100 }, .{ 16, 16 }, .{ 2, 50 }, .{ 50, 2 } }) |g| {
        const h, const w = g;
        const got = try a.alloc(f32, h * w * c);
        defer a.free(got);
        try resize(a, table, side, c, h, w, got);
        const wy = try weights(a, side, h);
        defer a.free(wy);
        const wx = try weights(a, side, w);
        defer a.free(wx);
        for (0..h) |y| for (0..w) |x| for (0..c) |k| {
            var sum: f64 = 0;
            for (0..side) |i| {
                var across: f64 = 0;
                for (0..side) |j| across += wx[x * side + j] * table[(i * side + j) * c + k];
                sum += wy[y * side + i] * across;
            }
            try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatCast(sum)))), @as(u32, @bitCast(got[(y * w + x) * c + k])));
        };
    }
}
