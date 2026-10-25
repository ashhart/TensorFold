//! The real Metal drawer's ABI runs under owned recording objects; no sampler arithmetic or Metal call runs here.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const L = @import("core").lanes;
const R = struct {
    const Self = @This();
    pub const Size = struct {
        width: usize,
        height: usize = 1,
        depth: usize = 1,
        pub fn of(x: usize, y: usize, z: usize) @This() {
            return .{ .width = x, .height = y, .depth = z };
        }
    };
    pub const Buffer = struct {
        bytes: []align(16) u8,
        pub fn deviceId(_: @This()) usize {
            return 1;
        }
        pub fn length(b: @This()) usize {
            return b.bytes.len;
        }
        pub fn contents(b: @This()) [*]u8 {
            return b.bytes.ptr;
        }
    };
    pub const Device = struct { id: usize = 1 };
    pub const CompileOptions = struct {
        pub fn mlx() @This() {
            return .{};
        }
    };
    var pipes: usize = 0;
    var fail_pipe: ?usize = null;
    var calls: usize = 0;
    var selected: usize = 0;
    var views: [8]struct { b: Buffer, off: usize } = undefined;
    var inline_bytes: [8][4096]u8 = undefined;
    var sizes: [8]usize = @splat(0);
    var positions: [2][2]u32 = undefined;
    var samples: usize = 0;
    pub const Library = struct {
        text: []const u8,
        pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !@This() {
            return .{ .text = text };
        }
        pub fn deinit(_: @This()) void {}
    };
    pub const Pipeline = struct {
        index: usize,
        pub fn init(_: Device, lib: Library, name: []const u8, _: bool) !@This() {
            const names = [_][]const u8{ "tf_sample_full", "fz_argmax", "tf_grammar_mask" };
            for (names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
                if (fail_pipe == i) return error.PlantedDrawPipeline;
                try t.expect(std.mem.indexOf(u8, lib.text, n) != null);
                pipes += 1;
                return .{ .index = i };
            };
            return error.BadSymbol;
        }
        pub fn maxThreads(p: @This()) usize {
            return if (p.index == 2) 256 else 1024;
        }
        pub fn simdWidth(_: @This()) usize {
            return 32;
        }
        pub fn deinit(_: @This()) void {
            pipes -= 1;
        }
    };
    pub const Encoder = struct {
        pub fn setPipeline(_: @This(), pipe: Pipeline) void {
            selected = pipe.index;
            calls += 1;
        }
        pub fn setBuffer(_: @This(), buffer: Buffer, off: usize, i: usize) void {
            views[i] = .{ .b = buffer, .off = off };
            calls += 1;
        }
        pub fn setBytes(_: @This(), bytes: []const u8, i: usize) void {
            @memcpy(inline_bytes[i][0..bytes.len], bytes);
            sizes[i] = bytes.len;
            calls += 1;
        }
        pub fn dispatchThreads(_: @This(), grid: Size, threads: Size) void {
            std.debug.assert(selected == 2 and grid.width == 2 and grid.height == 2 and threads.width == 256);
            calls += 1;
        }
        pub fn dispatchGroups(_: @This(), grid: Size, threads: Size) void {
            std.debug.assert(grid.width == 2 and grid.height == 1 and threads.width == 1024);
            if (selected == 0) {
                const rule = L.gpu_rule.rule(.{ .seed = 0x0123456789abcdef, .temperature = 0.5, .top_k = 7, .top_p = 0.75, .min_p = 0.125 });
                std.debug.assert(sizes[1] == 16 and sizes[3] == 32 and sizes[4] == 8);
                std.debug.assert(std.mem.bytesToValue(u32, inline_bytes[1][0..4]) == rule.seedHalves()[0]);
                std.debug.assert(std.mem.bytesToValue(f32, inline_bytes[3][0..4]) == rule.inv_t);
                std.debug.assert(std.mem.bytesToValue(f32, inline_bytes[3][12..16]) == rule.min_log);
                std.debug.assert(std.mem.bytesToValue(u32, inline_bytes[4][0..4]) == 7);
                for (0..2) |r| positions[samples][r] = std.mem.bytesToValue(u32, views[2].b.bytes[views[2].off + 4 * r ..][0..4]);
                samples += 1;
            }
            const dst = views[if (selected == 0) 5 else 1];
            for (0..grid.width) |r| std.mem.writeInt(u32, dst.b.bytes[dst.off + 4 * r ..][0..4], @intCast(5 + r), .little);
            calls += 1;
        }
        pub fn barrier(_: @This()) void {
            calls += 1;
        }
    };
    pub const Session = struct {
        pub fn encoder(_: *@This()) !Encoder {
            return .{};
        }
    };
};
const B = struct {
    pub const Buffer = R.Buffer;
    device: R.Device = .{},
    commands: ?*R.Session,
    held: usize = 0,
    fail_drain: bool = false,
    writes: usize = 0,
    pub fn allocate(b: *@This(), n: usize) !Buffer {
        b.held += n;
        return .{ .bytes = try a.alignedAlloc(u8, .@"16", n) };
    }
    pub fn release(b: *@This(), x: Buffer) void {
        b.held -= x.bytes.len;
        a.free(x.bytes);
    }
    pub fn write(b: *@This(), x: Buffer, off: usize, bytes: []const u8) !void {
        @memcpy(x.bytes[off..][0..bytes.len], bytes);
        b.writes += 1;
    }
    pub fn drain(b: *@This()) !void {
        if (b.fail_drain) return error.PlantedDrawDrain;
    }
};
test "Qwen4 drawer recording: Metal rules, absolute positions, output offsets, masks and failed-drain ownership" {
    const D = @import("metal_draw.zig").Drawer(R, L, B);
    var session: R.Session = .{};
    var b: B = .{ .commands = &session };
    R.fail_pipe = null;
    R.samples = 0;
    R.calls = 0;
    try t.expectEqual(@as(usize, 0), R.pipes);
    const input = try b.allocate(16 + 2 * 33 * 2);
    defer b.release(input);
    const tensor: @import("arena.zig").Arena(B).Tensor = .{ .buffer = input, .offset = 16, .rows = 2, .width = 33, .kind = .bf16 };
    var d = try D.load(a, &b, 33, 4, 2);
    const rule: L.Sampling = .{ .seed = 0x0123456789abcdef, .temperature = 0.5, .top_k = 7, .top_p = 0.75, .min_p = 0.125 };
    try d.draw(tensor, 16, &.{ 7, 19 }, rule, &.{ 0xffffffff, 1, 0xffffffff, 1 });
    try d.draw(tensor, 0, &.{ 2, 3 }, rule, &.{});
    try t.expectEqualSlices(u32, &.{ 7, 19 }, &R.positions[0]);
    try t.expectEqualSlices(u32, &.{ 2, 3 }, &R.positions[1]);
    const calls = R.calls;
    const writes = b.writes;
    try t.expectError(error.Qwen4DrawPosition, d.draw(tensor, 0, &.{ 0, 1 }, rule, &.{}));
    try t.expectError(error.Qwen4EmptyDrawMask, d.draw(tensor, 0, &.{ 1, 2 }, rule, &.{ 0, 2, 0, 2 }));
    try t.expectEqual(calls, R.calls);
    try t.expectEqual(writes, b.writes);
    try d.draw(tensor, 16, &.{ 7, 19 }, null, &.{});
    var ids: [2]u32 = undefined;
    try d.read(16, &ids);
    try t.expectEqualSlices(u32, &.{ 5, 6 }, &ids);
    const held = b.held;
    b.fail_drain = true;
    try t.expectError(error.PlantedDrawDrain, d.close());
    try t.expectEqual(held, b.held);
    try t.expectEqual(@as(usize, 3), R.pipes);
    b.fail_drain = false;
    try d.close();
    try t.expectEqual(@as(usize, 0), R.pipes);
    for (0..3) |i| {
        R.fail_pipe = i;
        try t.expectError(error.PlantedDrawPipeline, D.load(a, &b, 33, 4, 2));
        try t.expectEqual(input.bytes.len, b.held);
        try t.expectEqual(@as(usize, 0), R.pipes);
    }
    R.fail_pipe = null;
    b.commands = null;
    try t.expectError(error.Qwen4CommandSessionMissing, D.load(a, &b, 33, 4, 2));
}
