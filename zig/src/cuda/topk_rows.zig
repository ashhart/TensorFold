//! topk_rows.cu for every CUDA family: each row's k best scores, ties to the lower index, their indices ascending.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// The 1,024-thread form: at most one wave of rows (48 SMs) over 4,096 scores or more, GLM's select rule.
pub const wide_rows = 48;
pub const wide_scores = 4096;

/// Whether run launches the 1,024-thread form.
pub fn wide(sh: Shape) bool {
    return sh.rows <= wide_rows and sh.ld >= wide_scores;
}

/// rows of scores `ld` apart; row r takes its first counts[r] (all `ld` when counts is 0); k picks a row at most.
pub const Shape = struct { rows: usize, ld: usize, k: usize };

pub fn check(sh: Shape) error{TopkShape}!void {
    if (sh.k == 0 or sh.k > std.math.maxInt(i32) or sh.ld > std.math.maxInt(i32) or sh.rows > std.math.maxInt(i32)) return error.TopkShape;
}

pub const TopkRows = struct {
    mod: Module,
    narrow: Function,
    wide: Function,

    pub fn load(d: *const Driver) !TopkRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.topk_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .narrow = try mod.function("tf_topk_rows"), .wide = try mod.function("tf_topk_rows_wide") };
    }

    pub fn unload(t: *TopkRows) void {
        t.mod.unload();
    }

    /// scores f32 [rows, ld]; counts i32 [rows] or 0; out i32 [rows, k] (-1 past a row's picks); picked i32 [rows].
    pub fn run(t: *const TopkRows, s: Stream, scores: u64, counts: u64, out: u64, picked: u64, sh: Shape) !void {
        try check(sh);
        if (sh.rows == 0) return;
        try t.launchWith(if (wide(sh)) t.wide else t.narrow, if (wide(sh)) 1024 else 128, s, scores, counts, out, picked, sh);
    }

    /// One form for every row count (the contract runs both on the same rows).
    pub fn launchWith(_: *const TopkRows, f: Function, threads: u32, s: Stream, scores: u64, counts: u64, out: u64, picked: u64, sh: Shape) !void {
        var a: launch_.Args = .{};
        a.add(scores);
        a.add(@as(i64, @intCast(sh.ld)));
        a.add(counts);
        a.add(@as(i32, @intCast(sh.k)));
        a.add(out);
        a.add(picked);
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(sh.rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};

test "check refuses no picks and sizes past i32" {
    try check(.{ .rows = 2048, .ld = 16384, .k = 2048 });
    try std.testing.expectError(error.TopkShape, check(.{ .rows = 1, .ld = 16, .k = 0 }));
    try std.testing.expectError(error.TopkShape, check(.{ .rows = 1, .ld = @as(usize, std.math.maxInt(i32)) + 1, .k = 4 }));
}
