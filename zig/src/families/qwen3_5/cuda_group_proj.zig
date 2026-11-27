//! TF_Q35_GROUP_PROJ=1: same-input ModelOpt projections in one lane launch, each with its own launch's bits (opt-in).
const cuda = @import("cuda");
const platform = @import("sys");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const Proj = engine.Proj;
const group = cuda.qmmf_group;

/// The switch, read once when the ModelOpt kernels load; default off.
pub fn on() !bool {
    return platform.environment.read(bool, "TF_Q35_GROUP_PROJ", false);
}

/// The grouped kernels when the switch is on and slices meet in clusters (sm_90 on), else null.
pub fn load(lane: *const cuda.qmmf.Lane) !?group.Group {
    if (!lane.clusters or !try on()) return null;
    return try group.Group.load(lane);
}

/// The lane weight a ModelOpt projection decodes through (NVFP4, FP8, BF16), else null.
fn laneWeight(p: Proj) ?cuda.qmmf.Weight {
    if (p != .ql) return null;
    return switch (p.ql) {
        .nvfp4, .fp8, .bf16 => |q| q,
        else => null,
    };
}

/// `ls` over x into `outs` as compatible grouped launches, or false: the caller runs separate launches.
pub fn run(e: *Engine, ls: []const Proj, x: u64, outs: []const u64, rows: usize, prompt: bool) !bool {
    if (@import("cuda_launch_trace.zig").active != null) return false; // the trace keeps counting separate launches
    const k = if (e.mo) |*m| m else return false;
    const g = k.group orelse return false;
    if (prompt and rows > 16) return false; // modelopt.project's prompt GEMM
    if (ls.len > group.most) return false;
    var ws: [group.most]cuda.qmmf.Weight = undefined;
    for (ls, ws[0..ls.len]) |l, *w| w.* = laneWeight(l) orelse return false;
    for (ws[0..ls.len]) |w| if (w.k != ws[0].k) return false;
    for (outs, 0..) |out, i| for (outs[i + 1 ..]) |other| if (out == other) return false;
    if (!prompt and rows > group.max_rows) {
        var parts: [group.most]cuda.qmmf.Many = undefined;
        for (ws[0..ls.len], outs, 0..) |w, out, i| {
            if (w.mode != .fp4) return false;
            parts[i] = .{ .w = w, .out = out };
        }
        if (!cuda.qmmf.manyEligible(k.lane, rows, parts[0..ls.len])) return false;
        try k.lane.matmulMany(e.stream, x, ws[0].k, rows, parts[0..ls.len], if (k.part) |p| p.ptr else 0);
        return true;
    }
    var shapes: [2]group.Shape = undefined;
    var count: usize = 0;
    var at: usize = 0;
    while (at < ls.len) {
        var end = at + 1;
        while (end < ls.len and ws[end].mode == ws[at].mode) : (end += 1) {}
        shapes[count] = group.plan(k.lane, ws[at..end], outs[at..end], rows) orelse return false;
        count += 1;
        at = end;
    }
    for (shapes[0..count]) |sh| try g.launch(e.stream, x, rows, ws[0].k, sh, k.lane.pdl);
    return true;
}

test "Qwen wide group: NVFP4 verification launches once per pair and controls stay separate" {
    const std = @import("std");
    const Part = extern struct { codes: u64, scales: u64, out: u64, scale: f32, n: i32, npad: i32, tiles: i32 };
    const Parts = extern struct { p: [4]Part, count: i32 };
    const Located = struct { part: usize, tile: usize };
    const Oracle = struct {
        fn expected(rows: usize, weights: []const cuda.qmmf.Weight, outs: []const u64) Parts {
            var result = std.mem.zeroes(Parts);
            result.count = @intCast(weights.len);
            for (weights, outs, 0..) |w, out, i| {
                const bm = cuda.qmmf.bucket(rows);
                const count = (rows + bm - 1) / bm * ((w.n + 63) / 64);
                result.p[i] = .{ .codes = w.codes, .scales = w.scales, .out = out, .scale = w.scale, .n = @intCast(w.n), .npad = @intCast(w.npad), .tiles = @intCast(count) };
            }
            return result;
        }
        fn locate(parts: Parts, block: usize) ?Located {
            var local = block;
            for (parts.p[0..@intCast(parts.count)], 0..) |part, i| {
                const count: usize = @intCast(part.tiles);
                if (local < count) return .{ .part = i, .tile = local };
                local -= count;
            }
            return null;
        }
        fn check(actual: Parts, want: Parts) !void {
            try std.testing.expectEqual(want.count, actual.count);
            var offset: usize = 0;
            for (want.p[0..@intCast(want.count)], 0..) |member, i| {
                const part = actual.p[i];
                try std.testing.expectEqual(member.codes, part.codes);
                try std.testing.expectEqual(member.scales, part.scales);
                try std.testing.expectEqual(member.out, part.out);
                try std.testing.expectEqual(member.scale, part.scale);
                try std.testing.expectEqual(member.n, part.n);
                try std.testing.expectEqual(member.npad, part.npad);
                try std.testing.expectEqual(member.tiles, part.tiles);
                const count: usize = @intCast(member.tiles);
                const first = locate(actual, offset);
                try std.testing.expectEqual(true, first != null);
                try std.testing.expectEqual(i, first.?.part);
                try std.testing.expectEqual(@as(usize, 0), first.?.tile);
                const last = locate(actual, offset + count - 1);
                try std.testing.expectEqual(true, last != null);
                try std.testing.expectEqual(i, last.?.part);
                try std.testing.expectEqual(count - 1, last.?.tile);
                offset += count;
            }
            try std.testing.expectEqual(true, locate(actual, offset) == null);
        }
    };
    const Mock = struct {
        var calls: usize = 0;
        var grid: u32 = 0;
        var parts: Parts = std.mem.zeroes(Parts);
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            calls += 1;
            grid = gx;
            parts = @as(*const Parts, @ptrCast(@alignCast(args.?[1].?))).*;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Mock.launch;
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = true;
    lane.fuse_split_n = 0;
    lane.half_below = 0;
    for (&lane.many, 0..) |*function, i| function.* = .{ .d = &driver, .handle = @ptrFromInt(100 + i) };
    var e: Engine = undefined;
    e.stream = .{ .d = &driver, .handle = null };
    e.mo = .{ .lane = lane, .prompt = undefined, .pack = undefined, .embed = undefined, .group = .{ .fns = undefined } };
    const weights = [_]cuda.qmmf.Weight{
        .{ .mode = .fp4, .codes = 0x101000, .scales = 0x102000, .scale = 0.75, .n = 17408, .k = 5120, .npad = 17408 },
        .{ .mode = .fp4, .codes = 0x201000, .scales = 0x202000, .scale = 1.25, .n = 17536, .k = 5120, .npad = 17536 },
    };
    const ls = [_]Proj{ .{ .ql = .{ .nvfp4 = weights[0] } }, .{ .ql = .{ .nvfp4 = weights[1] } } };
    const outs = [_]u64{ 0x1000000, 0x2000000 };
    var good48: Parts = undefined;
    for (17..129) |rows| {
        Mock.calls = 0;
        try std.testing.expect(try run(&e, &ls, 0x1000, &outs, rows, false));
        try std.testing.expectEqual(@as(usize, 1), Mock.calls);
        const tiles = (rows + cuda.qmmf.bucket(rows) - 1) / cuda.qmmf.bucket(rows) * 546;
        try std.testing.expectEqual(@as(u32, @intCast(tiles)), Mock.grid);
        try Oracle.check(Mock.parts, Oracle.expected(rows, &weights, &outs));
        if (rows == 48) good48 = Mock.parts;
    }
    for ([_]struct { rows: usize, prompt: bool, enabled: bool }{
        .{ .rows = 129, .prompt = false, .enabled = true },
        .{ .rows = 17, .prompt = true, .enabled = true },
        .{ .rows = 17, .prompt = false, .enabled = false },
    }) |control| {
        e.mo.?.group = if (control.enabled) .{ .fns = undefined } else null;
        Mock.calls = 0;
        try std.testing.expect(!try run(&e, &ls, 0x1000, &outs, control.rows, control.prompt));
        try std.testing.expectEqual(@as(usize, 0), Mock.calls);
    }
    e.mo.?.group = .{ .fns = undefined };
    const fp8_weight: cuda.qmmf.Weight = .{ .mode = .fp8, .codes = 1, .scales = 2, .n = 17408, .k = 5120, .npad = 17408 };
    const fp8 = [_]Proj{ .{ .ql = .{ .fp8 = fp8_weight } }, .{ .ql = .{ .fp8 = fp8_weight } } };
    Mock.calls = 0;
    try std.testing.expect(!try run(&e, &fp8, 0x1000, &outs, 32, false));
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);
    const overlap = [_]u64{ outs[0], outs[0] + 2 };
    try std.testing.expect(!try run(&e, &ls, 0x1000, &overlap, 32, false));
    try std.testing.expectEqual(@as(usize, 0), Mock.calls);

    const expected48 = Oracle.expected(48, &weights, &outs);
    try Oracle.check(good48, expected48);
    var swapped = good48;
    std.mem.swap(Part, &swapped.p[0], &swapped.p[1]);
    // Planted swapped member must be red.
    try std.testing.expectError(error.TestExpectedEqual, Oracle.check(swapped, expected48));
    var shifted = good48;
    shifted.p[0].tiles += 1;
    shifted.p[1].tiles -= 1;
    // Planted off-by-one member tile offset must be red.
    try std.testing.expectError(error.TestExpectedEqual, Oracle.check(shifted, expected48));
    var zero = good48;
    zero.p[1].tiles = 0;
    try std.testing.expectError(error.TestExpectedEqual, Oracle.check(zero, expected48)); // planted tiles=0 must be red
}
