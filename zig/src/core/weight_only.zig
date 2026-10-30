//! Raw stored weight layouts with BF16 activations, shared by the Metal and CUDA projection adapters.
const std = @import("std");
pub const Format = enum(c_int) { bf16, mlx4, mlx8, fp8, nvfp4 };
pub const Scale = enum(c_int) { bf16, f16, f32 };
pub const Params = extern struct {
    rows: c_int,
    n: c_int,
    k: c_int,
    format: Format,
    group: c_int,
    scale_kind: Scale,
    reserved0: c_int = 0,
    reserved1: c_int = 0,
    global_scale: f32 = 1,
};
pub const Sizes = struct { x: usize, weight: usize, scales: usize, biases: usize, out: usize };
pub fn validate(p: Params) !Sizes {
    if (p.rows < 1 or p.rows > 128 or p.n < 1 or p.k < 1 or p.reserved0 != 0 or p.reserved1 != 0) return error.WeightOnlyShape;
    if (!std.math.isFinite(p.global_scale)) return error.WeightOnlyScale;
    switch (p.format) {
        .mlx4, .mlx8 => if ((p.group != 32 and p.group != 64 and p.group != 128) or @mod(p.k, p.group) != 0) return error.WeightOnlyGroup,
        .nvfp4 => if (@mod(p.k, 16) != 0 or p.group != 64) return error.WeightOnlyGroup,
        .bf16, .fp8 => if (p.group != 64) return error.WeightOnlyGroup,
    }
    const n: usize = @intCast(p.n);
    const k: usize = @intCast(p.k);
    const rows: usize = @intCast(p.rows);
    const nk = try std.math.mul(usize, n, k);
    const scale_bytes: usize = if (p.scale_kind == .f32) 4 else 2;
    const groups = switch (p.format) {
        .mlx4, .mlx8 => nk / @as(usize, @intCast(p.group)),
        else => 0,
    };
    const tables = try std.math.mul(usize, groups, scale_bytes);
    return .{
        .x = try std.math.mul(usize, try std.math.mul(usize, rows, k), 2),
        .weight = switch (p.format) {
            .bf16 => try std.math.mul(usize, nk, 2),
            .mlx4, .nvfp4 => nk / 2,
            .mlx8, .fp8 => nk,
        },
        .scales = if (p.format == .nvfp4) nk / 16 else tables,
        .biases = tables,
        .out = try std.math.mul(usize, try std.math.mul(usize, rows, n), 2),
    };
}
pub const Range = struct { address: usize, len: usize };
fn overlap(a: Range, b: Range) bool {
    if (a.len == 0 or b.len == 0) return false;
    return if (a.address <= b.address) b.address - a.address < a.len else a.address - b.address < b.len;
}
pub fn buffers(p: Params, x: Range, w: Range, s: Range, b: Range, out: Range) !void {
    const sizes = try validate(p);
    const ranges = [_]Range{ x, w, s, b, out };
    const needed = [_]usize{ sizes.x, sizes.weight, sizes.scales, sizes.biases, sizes.out };
    for (ranges, needed) |range, bytes| {
        if (range.len < bytes or (bytes > 0 and range.address == 0)) return error.WeightOnlyBuffer;
        _ = std.math.add(usize, range.address, range.len) catch return error.WeightOnlyBuffer;
    }
    if (x.address % 2 != 0 or out.address % 2 != 0) return error.WeightOnlyAlignment;
    for (ranges[0..4]) |input| if (overlap(out, input)) return error.WeightOnlyAlias;
}
