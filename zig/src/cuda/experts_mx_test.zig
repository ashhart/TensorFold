//! Stored-format experts on a mock launcher: the entry each call takes, its arguments, refusals that launch none.
const std = @import("std");
const abi = @import("abi.zig");
const experts = @import("experts.zig");
const mx = @import("experts_mx.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Plan = @import("grouped.zig").Plan;

const Mock = struct {
    var launches: usize = 0;
    var selected: usize = 0;
    var grid: u32 = 0;
    var shared: u32 = 0;
    var words: [16]u64 = undefined;
    fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        launches += 1;
        selected = @intFromPtr(f.?);
        grid = gx;
        shared = sm;
        // the soft-capped NVFP4 entry has 15 arguments, the MXFP8 ones 14 (no scale pointer)
        const wide = [_]usize{ 0, 3, 4, 7, 8, 9, 10 };
        const n: usize = if (selected < 40) 15 else 14;
        for (0..n) |i| {
            const big = if (selected < 40) std.mem.indexOfScalar(usize, &wide, i) != null else std.mem.indexOfScalar(usize, &[_]usize{ 0, 3, 6, 7, 8, 9 }, i) != null;
            words[i] = if (big) @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).* else @as(*const u32, @ptrCast(@alignCast(params.?[i].?))).*;
        }
        return 0;
    }
};

const plan: Plan = .{ .items = 60, .counts = 70, .members = 80, .rank = 90, .hist = 100 };
const in_up: experts.Rows = .{ .x = 10, .stride = 2304, .slots = 8 };
const mini: experts.Layer = .{ .format = .nvfp4, .up = 20, .down = 30, .up_scale = 40, .down_scale = 50, .width = 1152, .dims = 2304, .experts = 128, .caps = .{ 4, 25 } };

fn kernelsOn(d: *Driver) struct { k4: experts.Nvfp4Experts, k8: mx.Mxfp8Experts } {
    d.api.cuLaunchKernel = Mock.launch;
    var k4: experts.Nvfp4Experts = undefined; // only the soft-capped entries are reachable from the calls below
    k4.softcap = .{ .fns = .{ .{ .d = d, .handle = @ptrFromInt(31) }, .{ .d = d, .handle = @ptrFromInt(32) } }, .resident = 96 };
    var k8: mx.Mxfp8Experts = .{ .mod = undefined, .fns = undefined, .resident = @splat(96) };
    for (&k8.fns, 0..) |*f, i| f.* = .{ .d = d, .handle = @ptrFromInt(41 + i) };
    return .{ .k4 = k4, .k8 = k8 };
}

test "soft-capped NVFP4 gate-up takes tf_nvfp4_softcap with both caps; prompt rows the prompt entry and its shared" {
    var d: Driver = undefined;
    const k = kernelsOn(&d);
    const g: experts.Grouped = .{ .nvfp4 = &k.k4 };
    const s: Stream = .{ .d = &d, .handle = null };
    try g.decode(s, .up, in_up, mini, plan, 110, 5, 32, 16, -1);
    try std.testing.expectEqual(@as(usize, 31), Mock.selected);
    try std.testing.expectEqual(@as(u32, (5 * 36 + 3) / 4), Mock.grid);
    try std.testing.expectEqual(@as(u64, 8), Mock.words[2]); // slots
    try std.testing.expectEqual(@as(u64, 40), Mock.words[4]); // the per-expert factors
    try std.testing.expectEqual(@as(u64, 2304 / 32), Mock.words[5]);
    try std.testing.expectEqual(@as(u64, 1152 / 32), Mock.words[6]);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 4))), @as(u32, @truncate(Mock.words[12])));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 25))), @as(u32, @truncate(Mock.words[13])));
    try std.testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(i32, -1)))), Mock.words[14]);
    try g.prompt(s, .up, in_up, mini, plan, 110, 3, 192, 64, 7);
    try std.testing.expectEqual(@as(usize, 32), Mock.selected);
    try std.testing.expectEqual(@as(u32, 3 * 9), Mock.grid);
    try std.testing.expectEqual(experts.Nvfp4Experts.promptSmem(2, 4), Mock.shared);
    try std.testing.expectEqual(@as(u64, 7), Mock.words[14]);
}

test "MXFP8 calls take their entry by form and epilogue, with no scale pointer and the layer's caps" {
    var d: Driver = undefined;
    const k = kernelsOn(&d);
    const g: experts.Grouped = .{ .mxfp8 = &k.k8 };
    const s: Stream = .{ .d = &d, .handle = null };
    var l = mini;
    l.format = .mxfp8;
    const in_down: experts.Rows = .{ .x = 11, .stride = 1152 };
    const cases = [_]struct { prompt: bool, epi: experts.Epi, want: usize, grid: u32, kg: u64, nb: u64 }{
        .{ .prompt = false, .epi = .up, .want = 41, .grid = (5 * 36 + 3) / 4, .kg = 72, .nb = 36 },
        .{ .prompt = false, .epi = .down_f32, .want = 42, .grid = (5 * 72 + 3) / 4, .kg = 36, .nb = 72 },
        .{ .prompt = false, .epi = .down_bf16, .want = 43, .grid = (5 * 72 + 3) / 4, .kg = 36, .nb = 72 },
        .{ .prompt = true, .epi = .up, .want = 44, .grid = 5 * 18, .kg = 72, .nb = 36 },
        .{ .prompt = true, .epi = .down_bf16, .want = 45, .grid = 5 * 18, .kg = 36, .nb = 72 },
    };
    for (cases) |c| {
        const in = if (c.epi == .up) in_up else in_down;
        if (c.prompt) try g.prompt(s, c.epi, in, l, plan, 110, 5, 40, 64, -1) else try g.decode(s, c.epi, in, l, plan, 110, 5, 40, 16, -1);
        try std.testing.expectEqual(c.want, Mock.selected);
        try std.testing.expectEqual(c.grid, Mock.grid);
        try std.testing.expectEqual(@as(u64, if (c.epi == .up) 8 else 0), Mock.words[2]);
        try std.testing.expectEqual(@as(u64, if (c.epi == .up) 20 else 30), Mock.words[3]);
        try std.testing.expectEqual(c.kg, Mock.words[4]);
        try std.testing.expectEqual(c.nb, Mock.words[5]);
        try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 25))), @as(u32, @truncate(Mock.words[12])));
    }
}

test "refusals by name launch nothing: MXFP8 not loaded, caps unbuilt or malformed, shapes, prompt fp32 down" {
    var d: Driver = undefined;
    const k = kernelsOn(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    const none: experts.Grouped = .{};
    const both: experts.Grouped = .{ .nvfp4 = &k.k4, .mxfp8 = &k.k8 };
    Mock.launches = 0;
    var l = mini;
    l.format = .mxfp8;
    try std.testing.expectError(error.FormatNotLoaded, none.decode(s, .up, in_up, l, plan, 0, 1, 8, 16, -1));
    try std.testing.expectError(error.FormatNotLoaded, none.prompt(s, .down_bf16, in_up, l, plan, 0, 1, 8, 64, -1));
    try std.testing.expectError(error.EpilogueNotBuilt, both.prompt(s, .down_f32, in_up, l, plan, 0, 1, 8, 64, -1));
    l.caps = .{ 0, 0 }; // MXFP8's gate-up is built soft-capped only
    try std.testing.expectError(error.OptionNotBuilt, both.decode(s, .up, in_up, l, plan, 0, 1, 8, 16, -1));
    for ([_][2]f32{ .{ 4, 0 }, .{ -4, 25 }, .{ 4, std.math.inf(f32) }, .{ std.math.nan(f32), 25 } }) |cp| {
        l.caps = cp;
        try std.testing.expectError(error.BadSoftcap, both.decode(s, .up, in_up, l, plan, 0, 1, 8, 16, -1));
    }
    for (std.enums.values(experts.Format)) |f| if (f != .nvfp4 and f != .mxfp8) {
        var o = mini;
        o.format = f;
        try std.testing.expectError(error.OptionNotBuilt, both.decode(s, .up, in_up, o, plan, 0, 1, 8, 16, -1));
        try std.testing.expectError(error.OptionNotBuilt, both.prompt(s, .up, in_up, o, plan, 0, 1, 8, 64, -1));
    };
    var r = mini;
    r.relu2 = true;
    try std.testing.expectError(error.OptionNotBuilt, both.decode(s, .up, in_up, r, plan, 0, 1, 8, 16, -1));
    r = mini;
    r.limit = 7;
    try std.testing.expectError(error.OptionNotBuilt, both.prompt(s, .up, in_up, r, plan, 0, 1, 8, 64, -1));
    r = mini;
    r.width = 1144;
    try std.testing.expectError(error.UnexpectedShape, both.decode(s, .up, in_up, r, plan, 0, 1, 8, 16, -1));
    r.format = .mxfp8;
    r.width = 1152 + 32; // 37 column blocks: the prompt gate-up takes 2 a CTA
    try std.testing.expectError(error.UnexpectedShape, both.prompt(s, .up, in_up, r, plan, 0, 1, 8, 64, -1));
    try std.testing.expectEqual(@as(usize, 0), Mock.launches);
}
