//! DeepSeek-V4.1's indexer top-512 over fp32 score rows (deepseek_v41/topk512.hip): a 1,024-thread block a row.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const invalid = util.invalid;

/// The indices a row keeps, and the threads of the block that ranks it.
pub const keep = 512;
pub const block = 1024;

pub const Fns = struct {
    topk512: Function,

    pub fn load(m: Module) Error!Fns {
        return .{ .topk512 = try m.function("tf_dsv41_topk512") };
    }
};

/// At least one row of at least one score, each within an int: rows a grid's x, a row's indices int32 outputs.
pub fn fits(rows: usize, width: usize) bool {
    return rows > 0 and width > 0 and rows <= std.math.maxInt(c_int) and width <= std.math.maxInt(c_int);
}

/// selected [rows, 512] int32: row r's top min(512, visible[r]) of scores [rows, width] fp32 (fault 0 when served).
pub fn topk512(l: *const Launcher, scores: u64, visible: u64, selected: u64, rows: usize, width: usize, fault: c_int, s: S) Error!void {
    if (!fits(rows, width) or fault < 0 or fault > 4) return invalid("dsv41 topk512");
    var a: Args = .{};
    try a.add(scores);
    try a.add(visible);
    try a.add(selected);
    try a.add(@as(u32, @intCast(width)));
    try a.add(fault);
    try l.go(l.dsv41.topk512, dim(rows, 1, 1), dim(block, 1, 1), 0, s, &a);
}

test "dsv41 topk512 launches: a row a block, any width whose indices an int32 holds" {
    try std.testing.expect(fits(1, 1) and fits(8192, 1 << 20) and fits(1, std.math.maxInt(c_int)));
    try std.testing.expect(!fits(0, 512) and !fits(1, 0) and !fits(@as(usize, std.math.maxInt(c_int)) + 1, 512));
    try std.testing.expect(!fits(1, @as(usize, std.math.maxInt(c_int)) + 1));
}
