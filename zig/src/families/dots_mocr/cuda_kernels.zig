//! dots.mocr's own CUDA launches beside the dense engine's: text q|k|v RoPE, vision 2-D rotary and patch bias.
const std = @import("std");
const cuda = @import("cuda");

/// Threads a (row, head) block takes: dims j and j + 64 of a 128-dim head.
const threads = 64;
/// Threads a patch bias block takes, one column each.
const bias_threads = 256;
/// The vision rotary's heads a row: dots' 12 query heads, then its 12 key heads (values stay).
const vision_heads = 24;

pub const Kernels = struct {
    mod: cuda.Module,
    rope: cuda.Function,
    vision_rope: cuda.Function,
    patch_bias: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(ctx.d, cuda.kernels.dots_mocr);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .rope = try mod.function("tf_dots_qkv_rope"),
            .vision_rope = try mod.function("tf_dots_vision_rope"),
            .patch_bias = try mod.function("tf_dots_patch_bias"),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    /// q [rows, h, 128] and kv [rows, k hk | v hk] from fp32 acc [rows, (h + 2 hk) 128] plus bias, rotated at pos.
    pub fn qkvRope(k: *const Kernels, s: cuda.Stream, acc: u64, bias: u64, inv: u64, pos: u64, q: u64, kv: u64, rows: usize, h: usize, hk: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ acc, bias, inv, pos, q, kv }) |v| a.add(v);
        a.add(@as(c_int, @intCast(h)));
        a.add(@as(c_int, @intCast(hk)));
        try cuda.launch.launch(k.rope, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(h + 2 * hk), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// qkv [rows, 3, 12, 128] bf16: q and k rotated in place at hw [rows, 2] i32 (h, w), with inv [32] f32.
    pub fn visionRope(k: *const Kernels, s: cuda.Stream, qkv: u64, hw: u64, inv: u64, rows: usize) !void {
        const grid = try visionGrid(rows) orelse return;
        var a: cuda.Args = .{};
        for ([_]u64{ qkv, hw, inv }) |v| a.add(v);
        a.add(@as(c_int, @intCast(rows)));
        try cuda.launch.launch(k.vision_rope, .{ .grid = grid, .block = .{ .x = threads } }, s, &a);
    }

    /// o [rows, n] = bf16(y + b) from bf16 y [rows, n] and b [n]; o may be y.
    pub fn patchBias(k: *const Kernels, s: cuda.Stream, y: u64, b: u64, o: u64, rows: usize, n: usize) !void {
        const grid = try biasGrid(rows, n) orelse return;
        var a: cuda.Args = .{};
        for ([_]u64{ y, b, o }) |v| a.add(v);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(n)));
        try cuda.launch.launch(k.patch_bias, .{ .grid = grid, .block = .{ .x = bias_threads } }, s, &a);
    }
};

/// The vision rotary's grid (rows, 24), null for no rows; rows past an int's range refused.
fn visionGrid(rows: usize) !?cuda.launch.Dim3 {
    if (rows == 0) return null;
    if (rows > std.math.maxInt(c_int)) return error.RowShape;
    return .{ .x = @intCast(rows), .y = vision_heads, .z = 1 };
}

/// The patch bias's grid (rows, ceil(n / 256)), null when empty; rows or outputs a row past an int's range refused.
fn biasGrid(rows: usize, n: usize) !?cuda.launch.Dim3 {
    if (rows == 0 or n == 0) return null;
    if (rows > std.math.maxInt(c_int) or n > std.math.maxInt(c_int)) return error.RowShape;
    return .{ .x = @intCast(rows), .y = @intCast((n + bias_threads - 1) / bias_threads), .z = 1 };
}

test "dots vision launches: grids cover every row and column, empty calls launch nothing, oversize refused" {
    try std.testing.expectEqual(@as(?cuda.launch.Dim3, null), try visionGrid(0));
    const v = (try visionGrid(19520)).?;
    try std.testing.expect(v.x == 19520 and v.y == 24 and v.z == 1);
    try std.testing.expectError(error.RowShape, visionGrid(@as(usize, std.math.maxInt(c_int)) + 1));
    try std.testing.expectEqual(@as(?cuda.launch.Dim3, null), try biasGrid(0, 1536));
    try std.testing.expectEqual(@as(?cuda.launch.Dim3, null), try biasGrid(4, 0));
    const b = (try biasGrid(57600, 1536)).?;
    try std.testing.expect(b.x == 57600 and b.y == 6 and b.z == 1);
    try std.testing.expectEqual(@as(c_uint, 7), (try biasGrid(1, 1537)).?.y);
    try std.testing.expectError(error.RowShape, biasGrid(1, @as(usize, std.math.maxInt(c_int)) + 1));
}
