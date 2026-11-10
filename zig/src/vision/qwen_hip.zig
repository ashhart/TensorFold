//! Qwen3.5's native HIP image tower and merger: BF16 rows, FP32 accumulators and losslessly widened weights.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const op = @import("qwen_hip_ops.zig");
pub const Kernels = op.Kernels;
const weight = @import("qwen_hip_weights.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;
const Allocator = std.mem.Allocator;
pub const Config = core.qwen_vl.Config;
pub const Grid = struct { h: u32, w: u32 };
pub const max_depth = 32;
pub const max_patches = op.max_rows;
pub const patch_width = 1536;
pub const merge = 2;
pub const grid_side = 48;
pub const eps: f32 = 1e-6;

const Block = struct {
    norm1_w: u64,
    norm1_b: u64,
    qkv_w: u64,
    qkv_b: u64,
    proj_w: u64,
    proj_b: u64,
    norm2_w: u64,
    norm2_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
};

pub fn check(c: Config) !void {
    if (c.hidden == 0 or c.hidden > 2048 or c.heads == 0 or c.hidden / c.heads != 64 or
        c.hidden % c.heads != 0 or c.depth == 0 or c.depth > max_depth or c.intermediate == 0 or
        c.intermediate > 8192 or c.out_hidden == 0 or c.out_hidden > 8192 or c.patch != 16 or
        c.temporal != 2 or c.merge != merge or c.positions != grid_side * grid_side)
        return error.UnsupportedQwenVision;
}

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// Every stored BF16 tensor is widened to FP32, with a separately aligned inverse-frequency vector.
pub fn weightBytes(c: Config) !usize {
    try check(c);
    const h: usize = c.hidden;
    const mid: usize = c.intermediate;
    const merged: usize = c.merged();
    var n = aligned(h * patch_width * 4) + aligned(h * 4) + aligned(grid_side * grid_side * h * 4);
    n += c.depth * (4 * aligned(h * 4) + aligned(3 * h * h * 4) + aligned(3 * h * 4) +
        aligned(h * h * 4) + aligned(h * 4) + aligned(mid * h * 4) + aligned(mid * 4) +
        aligned(h * mid * 4) + aligned(h * 4));
    n += 2 * aligned(h * 4) + aligned(merged * merged * 4) + aligned(merged * 4) +
        aligned(@as(usize, c.out_hidden) * merged * 4) + aligned(@as(usize, c.out_hidden) * 4);
    return n + aligned(16 * 4);
}

pub const Scratch = struct {
    pixels: usize,
    x: usize,
    norm: usize,
    acc: usize,
    q: usize,
    key: usize,
    value: usize,
    attention: usize,
    mid: usize,
    hw: usize,
    index: usize,
    weights: usize,
    bytes: usize,

    pub fn init(c: Config, n: usize) !Scratch {
        try check(c);
        if (n == 0 or n > max_patches) return error.BadImagePatches;
        const h: usize = c.hidden;
        const widest = @max(@max(@max(3 * h, c.intermediate), c.merged()), (@as(usize, c.out_hidden) + 3) / 4);
        var p: usize = 0;
        return .{
            .pixels = take(&p, n * patch_width * 2),
            .x = take(&p, n * h * 2),
            .norm = take(&p, n * h * 2),
            .acc = take(&p, n * widest * 4),
            .q = take(&p, n * h * 2),
            .key = take(&p, n * h * 2),
            .value = take(&p, n * h * 2),
            .attention = take(&p, n * h * 2),
            .mid = take(&p, n * @max(@as(usize, c.intermediate), c.merged()) * 2),
            .hw = take(&p, n * 2 * 4),
            .index = take(&p, n * 4 * 4),
            .weights = take(&p, n * 4 * 4),
            .bytes = p,
        };
    }

    fn take(p: *usize, n: usize) usize {
        const at = p.*;
        p.* += aligned(n);
        return at;
    }
};

pub fn scratchBytes(c: Config, n: usize) !usize {
    return (try Scratch.init(c, n)).bytes;
}

pub const Tower = struct {
    gpa: Allocator,
    r: *const hip.Runtime,
    s: hip.Stream,
    c: Config,
    capacity: usize,
    ops: op.Kernels,
    arena: hip.DeviceBuffer,
    at: usize = 0,
    scratch: u64 = 0,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos: u64 = 0,
    blocks: [max_depth]Block = undefined,
    merger_norm_w: u64 = 0,
    merger_norm_b: u64 = 0,
    fc1_w: u64 = 0,
    fc1_b: u64 = 0,
    fc2_w: u64 = 0,
    fc2_b: u64 = 0,
    inv: u64 = 0,

    pub fn init(gpa: Allocator, io: std.Io, r: *const hip.Runtime, s: hip.Stream, launcher: *const hip.Launcher, dir: []const u8, c: Config, capacity: usize) !*Tower {
        _ = launcher;
        try check(c);
        const sb = try scratchBytes(c, capacity);
        if (s.r != r) return error.Invalid;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .r = r, .s = s, .c = c, .capacity = capacity, .ops = undefined, .arena = undefined };
        t.ops = try op.Kernels.load(r, hip.kernels.images[@backingInt(hip.kernels.Group.qwen_vision)]);
        errdefer t.ops.deinit();
        const wb = try weightBytes(c);
        t.arena = try hip.DeviceBuffer.alloc(r, wb + sb);
        errdefer t.arena.free();
        try weight.load(gpa, io, dir, c, t);
        var inv: [16]f32 = undefined;
        core.qwen_vl.grid.invFreq(&inv, 64, 10000);
        t.inv = try t.putFloats(&inv);
        if (t.at != wb) return error.VisionWeightSize;
        t.scratch = try t.take(sb);
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.s.synchronize() catch @panic("Qwen HIP vision resources are still in use after a drain failure");
        t.arena.free();
        t.ops.deinit();
        t.gpa.destroy(t);
    }

    fn take(t: *Tower, n: usize) !u64 {
        const at = aligned(t.at);
        if (at > t.arena.len or n > t.arena.len - at) return error.ArenaExhausted;
        t.at = at + aligned(n);
        return t.arena.base() + at;
    }

    /// The shared tensor map supplies BF16 bytes; the device's existing dense weight convention is FP32.
    pub fn put(t: *Tower, bytes: []const u8) !u64 {
        if (bytes.len % 2 != 0) return error.UnexpectedTensor;
        const values = try t.gpa.alloc(f32, bytes.len / 2);
        defer t.gpa.free(values);
        for (values, std.mem.bytesAsSlice(u16, bytes)) |*out, bits| out.* = @bitCast(@as(u32, bits) << 16);
        return t.putFloats(values);
    }

    fn putFloats(t: *Tower, values: []const f32) !u64 {
        const dst = try t.take(values.len * 4);
        try t.upload(dst - t.arena.base(), std.mem.sliceAsBytes(values));
        return dst;
    }

    fn upload(t: *Tower, offset: usize, bytes: []const u8) !void {
        const default: hip.Stream = .{ .r = t.r, .handle = null };
        errdefer default.synchronize() catch @panic("Qwen HIP's upload did not drain before releasing its source");
        try t.arena.upload(offset, bytes);
        try default.synchronize();
    }

    /// Images stay separate in full attention; the caller owns a BF16 output row for each merged token.
    pub fn encode(t: *Tower, pixels: []const f32, grids: []const Grid, out: u64) !void {
        return t.encodeCancelable(pixels, grids, out, null);
    }

    pub fn encodeCancelable(t: *Tower, pixels: []const f32, grids: []const Grid, out: u64, cancel: ?CancelCheck) !void {
        try cancels.poll(cancel);
        const n = try rowsOf(grids, pixels.len);
        if (n > t.capacity) return error.BadImagePatches;
        const c = t.c;
        const sc = try Scratch.init(c, n);
        const p = t.scratch;
        const bytes = n / 4 * c.out_hidden * 2;
        if (out == 0 or out % 2 != 0 or out > std.math.maxInt(u64) - bytes) return error.BadImagePatches;
        if (out < t.arena.base() + t.arena.len and t.arena.base() < out + bytes) return error.Aliasing;
        const px = try t.gpa.alloc(u16, pixels.len);
        defer t.gpa.free(px);
        const hw = try t.gpa.alloc(i32, n * 2);
        defer t.gpa.free(hw);
        const indices = try t.gpa.alloc(i32, n * 4);
        defer t.gpa.free(indices);
        const weights = try t.gpa.alloc(f32, n * 4);
        defer t.gpa.free(weights);
        for (px, pixels) |*v, x| v.* = core.contract.spec.toBf16(x);
        tables(grids, hw, indices, weights);
        try t.upload(p + sc.pixels - t.arena.base(), std.mem.sliceAsBytes(px));
        try t.upload(p + sc.hw - t.arena.base(), std.mem.sliceAsBytes(hw));
        try t.upload(p + sc.index - t.arena.base(), std.mem.sliceAsBytes(indices));
        try t.upload(p + sc.weights - t.arena.base(), std.mem.sliceAsBytes(weights));
        errdefer t.s.synchronize() catch @panic("Qwen HIP vision work did not drain after failure");
        const x = p + sc.x;
        const h = p + sc.norm;
        const acc = p + sc.acc;
        const q = p + sc.q;
        const key = p + sc.key;
        const v = p + sc.value;
        const att = p + sc.attention;
        const mid = p + sc.mid;
        try t.ops.linear(t.s, p + sc.pixels, t.patch_w, acc, n, c.hidden, patch_width);
        try t.ops.embed(t.s, acc, t.patch_b, t.pos, p + sc.index, p + sc.weights, x, n, c.hidden);
        try cancels.drain(t.s, cancel);
        for (t.blocks[0..c.depth]) |b| {
            try t.ops.layernorm(t.s, x, b.norm1_w, b.norm1_b, h, n, c.hidden, eps);
            try t.ops.linear(t.s, h, b.qkv_w, acc, n, 3 * c.hidden, c.hidden);
            try t.ops.qkv(t.s, acc, b.qkv_b, p + sc.hw, t.inv, q, key, v, n, c.heads);
            var start: usize = 0;
            for (grids) |grid| {
                const count = @as(usize, grid.h) * grid.w;
                try t.ops.attention(t.s, q, key, v, att, n, c.heads, start, count);
                start += count;
            }
            try t.ops.linear(t.s, att, b.proj_w, acc, n, c.hidden, c.hidden);
            try t.ops.residual(t.s, acc, b.proj_b, x, n, c.hidden);
            try t.ops.layernorm(t.s, x, b.norm2_w, b.norm2_b, h, n, c.hidden, eps);
            try t.ops.linear(t.s, h, b.fc1_w, acc, n, c.intermediate, c.hidden);
            try t.ops.gelu(t.s, acc, b.fc1_b, mid, n, c.intermediate, true);
            try t.ops.linear(t.s, mid, b.fc2_w, acc, n, c.hidden, c.intermediate);
            try t.ops.residual(t.s, acc, b.fc2_b, x, n, c.hidden);
            try cancels.drain(t.s, cancel);
        }
        try t.ops.layernorm(t.s, x, t.merger_norm_w, t.merger_norm_b, h, n, c.hidden, eps);
        try t.ops.linear(t.s, h, t.fc1_w, acc, n / 4, c.merged(), c.merged());
        try t.ops.gelu(t.s, acc, t.fc1_b, mid, n / 4, c.merged(), false);
        try t.ops.linear(t.s, mid, t.fc2_w, acc, n / 4, c.out_hidden, c.merged());
        try t.ops.cast(t.s, acc, t.fc2_b, out, n / 4, c.out_hidden);
        try t.s.synchronize();
        try cancels.poll(cancel);
    }
};

pub fn rowsOf(grids: []const Grid, pixel_values: usize) !usize {
    var n: usize = 0;
    for (grids) |g| {
        if (g.h == 0 or g.w == 0 or g.h % merge != 0 or g.w % merge != 0) return error.BadImageGrid;
        n = try std.math.add(usize, n, try std.math.mul(usize, g.h, g.w));
    }
    if (n == 0 or n > max_patches or pixel_values != n * patch_width) return error.BadImagePatches;
    return n;
}

pub fn tables(grids: []const Grid, hw: []i32, index: []i32, weights: []f32) void {
    var at: usize = 0;
    for (grids) |grid| {
        const n = @as(usize, grid.h) * grid.w;
        for (0..n) |i| {
            const xy = core.qwen_vl.grid.place(i, grid.w, merge);
            hw[(at + i) * 2] = @intCast(xy[0]);
            hw[(at + i) * 2 + 1] = @intCast(xy[1]);
            const taps = core.qwen_vl.grid.torchTaps(xy[0], xy[1], grid.h, grid.w, grid_side);
            for (0..4) |j| {
                index[(at + i) * 4 + j] = @intCast(taps.index[j]);
                weights[(at + i) * 4 + j] = taps.weight[j];
            }
        }
        at += n;
    }
}

test {
    _ = op;
    _ = @import("qwen_hip_test.zig");
}
