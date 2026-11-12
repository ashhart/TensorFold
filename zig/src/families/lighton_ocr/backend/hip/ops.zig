//! Native HIP LightOnOCR primitives, with Torch BF16 points and borrowed shared elementwise launches.
const std = @import("std");
const hip = @import("hip");

pub const threads = 256;
pub const max_grid = 110;
pub const max_rows = 16384;
pub const head_dim = 64;

/// The checkpoint's inverse frequencies, height first then width: alternate entries of theta^(-2j / dim).
pub fn inverseFrequencies(out: []f32, theta: f32) !void {
    if (out.len != head_dim / 2 or !std.math.isFinite(theta) or theta <= 0) return error.RotaryShape;
    const quarter = head_dim / 4;
    for (0..quarter) |j| {
        const eh: f32 = @as(f32, @floatFromInt(4 * j)) / head_dim;
        const ew: f32 = @as(f32, @floatFromInt(4 * j + 2)) / head_dim;
        out[j] = 1.0 / std.math.pow(f32, theta, eh);
        out[quarter + j] = 1.0 / std.math.pow(f32, theta, ew);
    }
}

/// One image's raster patch grid; only complete 2x2 groups become decoder rows.
pub const Grid = struct {
    h: usize,
    w: usize,

    pub fn init(h: usize, w: usize) !Grid {
        if (h == 0 or w == 0 or h > max_grid or w > max_grid) return error.ImageGrid;
        return .{ .h = h, .w = w };
    }

    pub fn rows(g: Grid) usize {
        return g.h * g.w;
    }

    pub fn tokens(g: Grid) usize {
        return (g.h / 2) * (g.w / 2);
    }

    pub fn tableBytes(g: Grid) usize {
        return g.rows() * (head_dim / 2) * 2;
    }

    pub fn mergedBytes(g: Grid, width: usize) !usize {
        if (width == 0 or width > std.math.maxInt(i32)) return error.RowShape;
        return std.math.mul(usize, g.tokens() * 8, width) catch error.RowShape;
    }
};

const Range = struct { ptr: u64, len: usize };

fn checked(r: Range) !void {
    if (r.ptr == 0 or r.ptr % 2 != 0) return error.RowShape;
    _ = std.math.add(u64, r.ptr, r.len) catch return error.RowShape;
}

fn disjoint(outputs: []const Range, inputs: []const Range) !void {
    for (outputs, 0..) |out, i| {
        try checked(out);
        for (inputs) |in| {
            try checked(in);
            if (out.ptr < in.ptr + in.len and in.ptr < out.ptr + out.len) return error.Aliasing;
        }
        for (outputs[0..i]) |other| {
            if (out.ptr < other.ptr + other.len and other.ptr < out.ptr + out.len) return error.Aliasing;
        }
    }
}

pub const TextShape = struct { h: usize = 16, hk: usize = 8, dim: usize = 128, eps: f32 = 1e-6 };

pub const Kernels = struct {
    mod: hip.Module,
    shared: *const hip.Launcher,
    tables_fn: hip.Function,
    rotary_fn: hip.Function,
    merge_fn: hip.Function,
    gelu_fn: hip.Function,
    rms_fn: hip.Function,
    text_qk_fn: hip.Function,
    rows_to_heads_fn: hip.Function,
    heads_to_rows_fn: hip.Function,
    attention_fn: hip.Function,

    /// The caller keeps the shared launcher and its runtime alive until this module and its work are released.
    pub fn load(r: *const hip.Runtime, image: []const u8, shared: *const hip.Launcher) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        if (shared.r != r) return error.Invalid;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .shared = shared,
            .tables_fn = try mod.function("tf_lo_tables"),
            .rotary_fn = try mod.function("tf_lo_rotary"),
            .merge_fn = try mod.function("tf_lo_merge"),
            .gelu_fn = try mod.function("tf_lo_gelu"),
            .rms_fn = try mod.function("tf_lo_rms"),
            .text_qk_fn = try mod.function("tf_lo_text_qk"),
            .rows_to_heads_fn = try mod.function("tf_lo_rows_to_heads"),
            .heads_to_rows_fn = try mod.function("tf_lo_heads_to_rows"),
            .attention_fn = try mod.function("tf_lo_attention64"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    pub fn tables(k: *const Kernels, s: hip.Stream, inv: u64, cs: u64, sn: u64, gh: usize, gw: usize) !void {
        const grid = try Grid.init(gh, gw);
        const n = grid.tableBytes();
        if (inv % 4 != 0) return error.RowShape;
        try disjoint(&.{ .{ .ptr = cs, .len = n }, .{ .ptr = sn, .len = n } }, &.{.{ .ptr = inv, .len = head_dim / 2 * 4 }});
        var a: hip.launch.Args = .{};
        for ([_]u64{ inv, cs, sn }) |v| try a.add(v);
        for ([_]usize{ gh, gw, head_dim }) |v| try a.add(@as(i32, @intCast(v)));
        try flat(k.tables_fn, s, n / 2, &a);
    }

    /// Packed BF16 raster q/k/v become head-major BF16 rows for full attention, with Torch rotary rounding.
    pub fn rotary(k: *const Kernels, s: hip.Stream, q: u64, key: u64, v: u64, cs: u64, sn: u64, qo: u64, ko: u64, vo: u64, rows: usize, heads: usize) !void {
        if (rows == 0 or rows > max_grid * max_grid or heads == 0 or heads > 16) return error.RowShape;
        const n = rows * heads * head_dim * 2;
        const t = rows * (head_dim / 2) * 2;
        try disjoint(&.{ .{ .ptr = qo, .len = n }, .{ .ptr = ko, .len = n }, .{ .ptr = vo, .len = n } }, &.{ .{ .ptr = q, .len = n }, .{ .ptr = key, .len = n }, .{ .ptr = v, .len = n }, .{ .ptr = cs, .len = t }, .{ .ptr = sn, .len = t } });
        var a: hip.launch.Args = .{};
        for ([_]u64{ q, key, v, cs, sn, qo, ko, vo }) |p| try a.add(p);
        for ([_]usize{ rows, heads, head_dim }) |count| try a.add(@as(i32, @intCast(count)));
        try flat(k.rotary_fn, s, n / 2, &a);
    }

    /// Channel-first unfold: unmatched final encoder rows or columns are not sent to the learned merger.
    pub fn merge(k: *const Kernels, s: hip.Stream, x: u64, out: u64, gh: usize, gw: usize, width: usize) !void {
        const grid = try Grid.init(gh, gw);
        const n = try grid.mergedBytes(width);
        if (n == 0) return;
        const in_len = std.math.mul(usize, grid.rows() * 2, width) catch return error.RowShape;
        try disjoint(&.{.{ .ptr = out, .len = n }}, &.{.{ .ptr = x, .len = in_len }});
        var a: hip.launch.Args = .{};
        try a.add(x);
        try a.add(out);
        for ([_]usize{ gh, gw, width }) |count| try a.add(@as(i32, @intCast(count)));
        try flat(k.merge_fn, s, n / 2, &a);
    }

    /// The shared HIP SiLU multiply already contains Torch's BF16 rounding between SiLU and the product.
    pub fn torchSwiGlu(k: *const Kernels, s: hip.Stream, gate: u64, up: u64, out: u64, rows: usize, width: usize) !void {
        const n = try pointShape(rows, width);
        if (n == 0) return;
        try sameRuntime(k, s);
        try pointAliases(out, &.{ gate, up }, n * 2);
        try k.shared.tf_silu_mul(@ptrFromInt(gate), @ptrFromInt(up), @ptrFromInt(out), 2, @intCast(n), s.handle);
    }

    pub fn gelu(k: *const Kernels, s: hip.Stream, x: u64, out: u64, rows: usize, width: usize) !void {
        const n = try pointShape(rows, width);
        if (n == 0) return;
        try pointAliases(out, &.{x}, n * 2);
        var a: hip.launch.Args = .{};
        try a.add(x);
        try a.add(out);
        try a.add(@as(i64, @intCast(n)));
        try flat(k.gelu_fn, s, n, &a);
    }

    /// The shared HIP add supplies the raw BF16 residual step; x may be both the source and destination.
    pub fn residual(k: *const Kernels, s: hip.Stream, branch: u64, x: u64, rows: usize, width: usize) !void {
        const n = try pointShape(rows, width);
        if (n == 0) return;
        try sameRuntime(k, s);
        try pointAliases(x, &.{branch}, n * 2);
        try k.shared.tf_add(@ptrFromInt(x), @ptrFromInt(branch), @ptrFromInt(x), 2, @intCast(n), s.handle);
    }

    /// Weights are original BF16 widened exactly to FP32; the normalized activation still rounds before the weight.
    pub fn rms(k: *const Kernels, s: hip.Stream, x: u64, weight: u64, out: u64, rows: usize, width: usize, eps_: f32) !void {
        const n = try pointShape(rows, width);
        if (!std.math.isFinite(eps_) or eps_ <= 0 or weight % 4 != 0) return error.RowShape;
        if (n == 0) return;
        try pointAliases(out, &.{x}, n * 2);
        try disjoint(&.{.{ .ptr = out, .len = n * 2 }}, &.{.{ .ptr = weight, .len = width * 4 }});
        var a: hip.launch.Args = .{};
        for ([_]u64{ x, weight, out }) |p| try a.add(p);
        try a.add(@as(i32, @intCast(width)));
        try a.add(eps_);
        try hip.launch.launch(k.rms_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, s, &a);
    }

    /// Qwen3 text: separate packed BF16 projections to packed BF16 Q/K/V, full head128 RoPE, no V norm or KV write.
    pub fn textQk(k: *const Kernels, s: hip.Stream, q: u64, key: u64, v: u64, wq: u64, wk: u64, inv: u64, pos: u64, qo: u64, ko: u64, vo: u64, rows: usize, sh: TextShape) !void {
        if (rows == 0 or rows > max_rows or sh.h != 16 or sh.hk != 8 or sh.dim != 128 or !std.math.isFinite(sh.eps) or sh.eps <= 0) return error.RowShape;
        if ((wq | wk | inv | pos) % 4 != 0) return error.RowShape;
        const qn = rows * sh.h * sh.dim * 2;
        const kvn = rows * sh.hk * sh.dim * 2;
        try disjoint(&.{ .{ .ptr = qo, .len = qn }, .{ .ptr = ko, .len = kvn }, .{ .ptr = vo, .len = kvn } }, &.{ .{ .ptr = q, .len = qn }, .{ .ptr = key, .len = kvn }, .{ .ptr = v, .len = kvn }, .{ .ptr = wq, .len = sh.dim * 4 }, .{ .ptr = wk, .len = sh.dim * 4 }, .{ .ptr = inv, .len = sh.dim / 2 * 4 }, .{ .ptr = pos, .len = rows * 4 } });
        var a: hip.launch.Args = .{};
        for ([_]u64{ q, key, v, wq, wk, inv, pos, qo, ko, vo }) |p| try a.add(p);
        for ([_]usize{ sh.h, sh.hk, sh.dim }) |count| try a.add(@as(i32, @intCast(count)));
        try a.add(sh.eps);
        try hip.launch.launch(k.text_qk_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(sh.h + 2 * sh.hk) }, .block = .{ .x = @intCast(sh.dim) } }, s, &a);
    }

    pub fn rowsToHeads(k: *const Kernels, s: hip.Stream, src: u64, dst: u64, rows: usize, heads: usize, dim: usize) !void {
        return transpose(k.rows_to_heads_fn, s, src, dst, rows, heads, dim, false);
    }

    pub fn headsToRows(k: *const Kernels, s: hip.Stream, src: u64, dst: u64, rows: usize, heads: usize, dim: usize) !void {
        return transpose(k.heads_to_rows_fn, s, src, dst, rows, heads, dim, true);
    }

    /// Each call isolates one image span; static LDS holds only two 64x64 FP32 tiles, independent of image size.
    pub fn attention(k: *const Kernels, s: hip.Stream, q: u64, key: u64, v: u64, out: u64, total: usize, heads: usize, start: usize, count: usize, scale: f32) !void {
        if (total == 0 or total > max_rows or heads == 0 or heads > 16 or count == 0 or start > total or count > total - start or !std.math.isFinite(scale) or scale <= 0) return error.RowShape;
        const bytes = total * heads * head_dim * 2;
        try disjoint(&.{.{ .ptr = out, .len = bytes }}, &.{ .{ .ptr = q, .len = bytes }, .{ .ptr = key, .len = bytes }, .{ .ptr = v, .len = bytes } });
        var a: hip.launch.Args = .{};
        for ([_]u64{ q, key, v, out }) |p| try a.add(p);
        for ([_]usize{ total, heads, start, count }) |n| try a.add(@as(i32, @intCast(n)));
        try a.add(scale);
        try hip.launch.launch(k.attention_fn, .{ .grid = .{ .x = @intCast((count + 63) / 64), .y = @intCast(heads) }, .block = .{ .x = 64 } }, s, &a);
    }
};

fn sameRuntime(k: *const Kernels, s: hip.Stream) !void {
    if (k.shared.r != s.r) return error.Invalid;
}

fn pointShape(rows: usize, width: usize) !usize {
    if (width == 0 or width > 4096 or rows > max_rows) return error.RowShape;
    return rows * width;
}

fn pointAliases(out: u64, inputs: []const u64, bytes: usize) !void {
    try checked(.{ .ptr = out, .len = bytes });
    for (inputs) |in| {
        try checked(.{ .ptr = in, .len = bytes });
        if (in != out and out < in + bytes and in < out + bytes) return error.Aliasing;
    }
}

fn transpose(f: hip.Function, s: hip.Stream, src: u64, dst: u64, rows: usize, heads: usize, dim: usize, wide_in: bool) !void {
    if (rows == 0 or rows > max_rows or heads == 0 or heads > 32 or (dim != 64 and dim != 128)) return error.RowShape;
    const n = rows * heads * dim;
    if ((if (wide_in) src else dst) % 4 != 0) return error.RowShape;
    try disjoint(&.{.{ .ptr = dst, .len = n * (if (wide_in) @as(usize, 2) else 4) }}, &.{.{ .ptr = src, .len = n * (if (wide_in) @as(usize, 4) else 2) }});
    var a: hip.launch.Args = .{};
    try a.add(src);
    try a.add(dst);
    for ([_]usize{ rows, heads, dim }) |count| try a.add(@as(i32, @intCast(count)));
    try flat(f, s, n, &a);
}

fn flat(f: hip.Function, s: hip.Stream, n: usize, args: *hip.launch.Args) !void {
    try hip.launch.launch(f, .{ .grid = .{ .x = @intCast((n + threads - 1) / threads) }, .block = .{ .x = threads } }, s, args);
}

test "LightOn HIP ops: actual image geometry and disjoint ranges are checked without a device" {
    const t = std.testing;
    const grid = try Grid.init(3, 5);
    try t.expectEqual(@as(usize, 15), grid.rows());
    try t.expectEqual(@as(usize, 2), grid.tokens());
    try t.expectEqual(@as(usize, 16384), try grid.mergedBytes(1024));
    try t.expectError(error.ImageGrid, Grid.init(111, 110));
    try t.expectError(error.RowShape, pointShape(1, 4097));
    try t.expectError(error.Aliasing, disjoint(&.{.{ .ptr = 0x1000, .len = 8 }}, &.{.{ .ptr = 0x1002, .len = 8 }}));
    try t.expectError(error.RowShape, checked(.{ .ptr = std.math.maxInt(u64) - 1, .len = 8 }));
    try pointAliases(0x1000, &.{0x1000}, 16);
}
