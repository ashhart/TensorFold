//! Validate an EXL3 projection before the dry run can allocate or upload anything.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const mul1_magic: u32 = 0x83DCD12D;
fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// One EXL3 group's tensors, checked: trellis [k/16][n/16][16 * bits] i16, the mul1 marker, fp16 suh [k] and svh [n].
pub const Group = struct { trellis: core.checkpoint.Tensor, suh: core.checkpoint.Tensor, svh: core.checkpoint.Tensor, halves: u32 };

pub fn group(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    if (n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0 or n > std.math.maxInt(u32) or k > std.math.maxInt(u32)) return error.UnexpectedTensor;
    var buf: [240]u8 = undefined;
    const t = try ck.get(try name(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) {
        return error.UnexpectedTensor;
    }
    const halves = t.dim(2) / 8;
    if (halves < 2 or halves > 16) return error.UnsupportedQwenQuantization;
    const marker = try ck.get(try name(&buf, "{s}.mul1", .{base}));
    if ((marker.dtype != .i32 and marker.dtype != .u32) or marker.numel() != 1 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedQwenQuantization;
    const suh = try ck.get(try name(&buf, "{s}.suh", .{base}));
    const svh = try ck.get(try name(&buf, "{s}.svh", .{base}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedTensor;
    return .{ .trellis = t, .suh = suh, .svh = svh, .halves = @intCast(halves) };
}

/// Dense projections must also have a compiled linear variant.
pub fn denseGroup(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    const g = try group(ck, base, n, k);
    if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return error.UnsupportedQwenQuantization;
    return g;
}
