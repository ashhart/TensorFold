//! Calls the production encoder and its frozen predecessor through the same recording driver.
const std = @import("std");
const cuda = @import("root.zig");
const kda = @import("kda.zig");
const old = @import("kda_pre_shared_test.zig");
const Record = struct {
    id: usize = 0,
    grid: [3]u32 = @splat(0),
    block: [3]u32 = @splat(0),
    shared: u32 = 0,
    stream: usize = 0,
    args: [24]u64 = @splat(0),
};
const Capture = struct {
    var calls: [3]Record = undefined;
    var count: usize = 0;
    var symbols: usize = 0;
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const names = [_][]const u8{ "chain", "replay", "replay_layers", "prep", "step", "out" };
        inline for (names, 1..) |suffix, id| if (std.mem.eql(u8, std.mem.span(name), "tf_glm_kda_" ++ suffix)) {
            out.* = @ptrFromInt(id);
            symbols += 1;
            return 0;
        };
        return 500;
    }
    fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shared: u32, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        const widths: []const u8 = switch (id) {
            1 => &.{ 4, 8, 4, 4, 8, 4, 8, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8 },
            2 => &.{ 4, 8, 8, 8, 8, 8, 4, 8 },
            3 => &.{ 4, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8 },
            4 => &.{ 4, 8, 4, 4, 8, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8 },
            5 => &.{ 4, 8, 8, 8, 8, 8, 8, 4, 8, 8 },
            6 => &.{ 4, 8, 8, 4, 8, 4, 8 },
            else => unreachable,
        };
        var r: Record = .{ .id = id, .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = shared, .stream = @intFromPtr(stream) };
        for (widths, 0..) |width, i| {
            const p: [*]const u8 = @ptrCast(args.?[i].?);
            r.args[i] = if (width == 8) std.mem.readInt(u64, p[0..8], .little) else std.mem.readInt(u32, p[0..4], .little);
        }
        calls[count] = r;
        count += 1;
        return 0;
    }
};
fn cast(comptime T: type, from: anytype) T {
    var to: T = undefined;
    inline for (@typeInfo(T).@"struct".field_names) |name| @field(to, name) = @field(from, name);
    return to;
}
fn equal(expected: []const Record) !void {
    try std.testing.expectEqual(expected.len, Capture.count);
    for (expected, Capture.calls[0..Capture.count]) |a, b| try std.testing.expectEqualDeep(a, b);
}

test "shared KDA: fused, wide, cuts and layered replay preserve every predecessor launch byte" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    d.api.cuModuleGetFunction = Capture.get;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x1234) };
    Capture.symbols = 0;
    const ks = try kda.Kernels.load(.{ .d = &d, .handle = null });
    try std.testing.expectEqual(@as(usize, 6), Capture.symbols);
    const prev = cast(old.Kernels, ks);
    const s: kda.Saves = .{ .out = 0x100000001, .state_out = 0x200000002, .k = 0x300000003, .v = 0x400000004, .g = 0x500000005, .b = 0x600000006 };
    const w: kda.Wide = .{ .q = 0x700000007, .y = 0x800000008 };
    for ([_]usize{ 1, 21, 22, 32, 64 }) |heads| for ([_]usize{ 1, 3, 8, 16, 63, 64, 65, 70, 2048, 4096 }) |rows| {
        const n = heads * kda.dk;
        const c: kda.Chain = .{ .heads = heads, .p = 0x900000009, .p_stride = 3 * n + 256 + heads + 7, .b_off = 3 * n + 256, .a = 0xa0000000a, .a_stride = n + 9, .g = 0xb0000000b, .g_stride = n + 11, .conv_state = 0xc0000000c, .conv_w = 0xd0000000d, .state_in = 0xe0000000e, .a_log = 0xf0000000f, .dt_bias = 0x1000000010, .norm_w = 0x1100000011, .eps = 1e-6, .lower = -5 };
        for ([_]bool{ false, true }) |wide| {
            Capture.count = 0;
            try old.chain(&prev, stream, cast(old.Chain, c), rows, cast(old.Saves, s), if (wide) cast(old.Wide, w) else null);
            const expected = Capture.calls;
            const count = Capture.count;
            Capture.count = 0;
            try kda.chain(&ks, stream, c, rows, s, if (wide) w else null);
            try equal(expected[0..count]);
        }
        for ([_]usize{ 0, 1, rows - 1, rows }) |keep| {
            Capture.count = 0;
            try old.replay(&prev, stream, c.state_in, s.k, s.v, s.g, s.b, keep, heads, s.state_out);
            const expected = Capture.calls;
            Capture.count = 0;
            try kda.replay(&ks, stream, c.state_in, s.k, s.v, s.g, s.b, keep, heads, s.state_out);
            try equal(expected[0..1]);
            const rp: kda.Replay = .{ .state_in = c.state_in, .state_stride = heads * kda.dk * kda.dv, .k = s.k, .v = s.v, .g = s.g, .b = s.b, .kv_stride = rows * n, .b_stride = rows * heads, .state_out = s.state_out };
            for ([_]usize{ 1, 3, 72 }) |layers| {
                Capture.count = 0;
                try old.replayLayers(&prev, stream, rp, keep, heads, layers);
                const batch = Capture.calls;
                Capture.count = 0;
                try kda.replayLayers(&ks, stream, rp, keep, heads, layers);
                try equal(batch[0..1]);
            }
        }
    };
}

test "shared KDA: malformed geometry refuses before the driver is touched" {
    const ks: kda.Kernels = undefined;
    const stream: cuda.Stream = undefined;
    var c: kda.Chain = .{ .heads = 1, .p = 0, .p_stride = 385, .b_off = 384, .a = 0, .a_stride = 128, .g = 0, .g_stride = 128, .conv_state = 0, .conv_w = 0, .state_in = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5 };
    const saves: kda.Saves = undefined;
    Capture.count = 0;
    try std.testing.expectError(error.KdaShape, kda.chain(&ks, stream, c, 0, saves, null));
    c.heads = 0;
    try std.testing.expectError(error.KdaShape, kda.chain(&ks, stream, c, 1, saves, null));
    c.heads = 1;
    c.a_stride = 127;
    try std.testing.expectError(error.KdaShape, kda.chain(&ks, stream, c, 1, saves, null));
    c.a_stride = 128;
    c.b_off = 383;
    try std.testing.expectError(error.KdaShape, kda.chain(&ks, stream, c, 1, saves, null));
    c.b_off = 384;
    c.p_stride = 384;
    try std.testing.expectError(error.KdaShape, kda.chain(&ks, stream, c, 1, saves, null));
    try std.testing.expectError(error.KdaShape, kda.replay(&ks, stream, 0, 0, 0, 0, 0, 1, 0, 0));
    try std.testing.expectError(error.KdaShape, kda.replayLayers(&ks, stream, undefined, 1, 1, 0));
    try std.testing.expectEqual(@as(usize, 0), Capture.count);
}
