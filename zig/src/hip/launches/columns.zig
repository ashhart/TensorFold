//! BF16 column broadcasts used by checkpoint RMS norms and biased projections.
const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

pub const Functions = struct {
    mul: Function,
    fill: Function,

    pub fn load(module: Module) util.Error!Functions {
        return .{ .mul = try module.function("tf_columns_mul_bf16"), .fill = try module.function("tf_columns_fill_bf16") };
    }
};

pub fn count(rows: usize, width: usize) error{Invalid}!usize {
    if (rows == 0 or width == 0 or width > 65536) return error.Invalid;
    const n = std.math.mul(usize, rows, width) catch return error.Invalid;
    if (n > std.math.maxInt(u32) - 255) return error.Invalid;
    return n;
}

pub fn multiply(l: *const Launcher, x: u64, weight: u64, out: u64, rows: usize, width: usize, s: util.S) util.Error!void {
    const n = try count(rows, width);
    var args: util.Args = .{};
    try args.add(x);
    try args.add(weight);
    try args.add(out);
    try args.add(@as(i64, @intCast(n)));
    try args.add(@as(c_int, @intCast(width)));
    try l.flat(l.columns.mul, @intCast(n), s, &args);
}

pub fn fill(l: *const Launcher, weight: u64, out: u64, rows: usize, width: usize, s: util.S) util.Error!void {
    const n = try count(rows, width);
    var args: util.Args = .{};
    try args.add(weight);
    try args.add(out);
    try args.add(@as(i64, @intCast(n)));
    try args.add(@as(c_int, @intCast(width)));
    try l.flat(l.columns.fill, @intCast(n), s, &args);
}
