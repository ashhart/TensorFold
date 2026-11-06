//! MOSS-Image's own CUDA kernels (moss_image.cu): the modulated norm, q/k with the rotary, situ and the gated add.
const std = @import("std");
const cuda = @import("cuda");

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    norm: cuda.Function,
    qk: cuda.Function,
    situ: cuda.Function,
    gate_add: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.moss_image);
        errdefer mod.unload();
        return .{ .mod = mod, .norm = try mod.function("tf_moss_norm"), .qk = try mod.function("tf_moss_qk"), .situ = try mod.function("tf_moss_situ"), .gate_add = try mod.function("tf_moss_gate_add") };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// y = the RMSNorm of `rows` rows of x with weight w, then bf16(bf16(y s) + b) when `modulated`.
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, scale: u64, shift: u64, y: u64, rows: usize, width: usize, eps: f32, modulated: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, scale, shift, y }) |v| a.add(v);
    a.add(int(width));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(modulated)));
    try go(k.norm, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// q, k heads normed and rotated, v copied: rows `stride` apart into packed q | k | v rows of 3 heads 128.
pub fn qk(k: *const Kernels, s: cuda.Stream, qr: u64, kr: u64, vr: u64, nq: u64, nk: u64, cos: u64, sin: u64, qkv: u64, rows: usize, heads: usize, stride: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ qr, kr, vr, nq, nk, cos, sin, qkv }) |v| a.add(v);
    for ([_]usize{ rows, heads, stride }) |v| a.add(int(v));
    a.add(eps);
    try go(k.qk, s, .{ .x = @intCast(rows), .y = ceil(heads, 8) }, 256, &a);
}

/// o [rows][width] = situ(h, g), h and g rows `stride` apart.
pub fn situ(k: *const Kernels, s: cuda.Stream, h: u64, g: u64, o: u64, rows: usize, width: usize, stride: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ h, g, o }) |v| a.add(v);
    for ([_]usize{ width, stride, rows }) |v| a.add(int(v));
    try go(k.situ, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

/// x += y gated by the vector g by column.
pub fn gateAdd(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, g: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, y, g }) |v| a.add(v);
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.gate_add, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

test "launch geometry: rows on grid x (past y's 65,535), 256-column strips on y" {
    try std.testing.expectEqual(@as(u32, 12), ceil(3072, 256));
    try std.testing.expectEqual(@as(u32, 3), ceil(24, 8));
}
