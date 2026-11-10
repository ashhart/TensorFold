//! A committed one-row conv tail advances identically with a real rollback destination or a null one.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64 = 0, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int = 0, len: c_int = 0, row0: c_int = 0, rows: c_int = 0, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
const nl = 2;
const channels = (2 * 16 + 48) * 128;
const tail_rows = 3;
const tail_bytes = nl * tail_rows * channels * 2;
const rec_bytes = nl * 48 * 128 * 128 * 4;
const geo: Geo = .{ .hid = 0, .nk = 16, .nv = 48, .qh = 0, .kvh = 0, .nl = nl };
const Out = struct { tail: [2][]const u8, before: [2][]const u8 };

fn values(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*x| x.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
    return out;
}

fn launch(gpu: check.Gpu, s: cuda.Stream, fun: cuda.Function, initial: [2][]u16, qkv: []const u16, omit: bool, a: std.mem.Allocator) !Out {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var segs: [2]Seg = undefined;
    var tails: [2]u64 = undefined;
    var backups: [2]u64 = undefined;
    for (&segs, initial, 0..) |*sg, data, st| {
        const tail = try rig.scratch(tail_bytes);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = tail, .len = tail_bytes }).upload(0, std.mem.sliceAsBytes(data));
        tails[st] = tail;
        backups[st] = if (omit) 0 else try rig.output(rec_bytes + tail_bytes);
        sg.* = .{ .conv = tail, .before = backups[st], .row0 = @intCast(st), .rows = 1 };
    }
    const segs_d = try rig.input(std.mem.sliceAsBytes(&segs));
    const qkv_d = try rig.input(std.mem.sliceAsBytes(qkv));
    const parents = [_]c_int{ -1, -1 };
    const params = [_]c_int{ 0, 0, 2, 2, 1 };
    const parents_d = try rig.input(std.mem.sliceAsBytes(&parents));
    const params_d = try rig.input(std.mem.sliceAsBytes(&params));
    const seen = try a.dupe(u8, try rig.inputBytes());
    for (0..nl) |layer| {
        var args: cuda.Args = .{};
        inline for (.{ geo, qkv_d, segs_d, parents_d, @as(c_int, @intCast(layer)), params_d }) |x| args.add(x);
        try cuda.launch.launch(fun, .{ .grid = .{ .x = (channels + 255) / 256, .y = 2 }, .block = .{ .x = 256 } }, s, &args);
    }
    var out: Out = undefined;
    for (0..2) |st| {
        out.tail[st] = try a.dupe(u8, try rig.bytes(tails[st], tail_bytes));
        out.before[st] = if (backups[st] == 0) &.{} else try a.dupe(u8, try rig.bytes(backups[st] + rec_bytes, tail_bytes));
        if (backups[st] != 0) for (try rig.bytes(backups[st], rec_bytes)) |byte|
            try check.expect(byte == 0xff, "conv backup overwrote the unrelated F32 state", .{});
    }
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, seen, try rig.inputBytes()), "conv backup changed a guard or input", .{});
    return out;
}

pub fn run(gpu: check.Gpu, s: cuda.Stream, fun: cuda.Function) !void {
    for (0..3) |seed| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var rng = std.Random.DefaultPrng.init(0xc0a7 + seed);
        const qkv = try values(a, rng.random(), 2 * channels);
        var initial: [2][]u16 = undefined;
        var expected: [2][]u16 = undefined;
        for (&initial, &expected, 0..) |*src, *want, st| {
            src.* = try values(a, rng.random(), tail_bytes / 2);
            want.* = try a.alloc(u16, tail_bytes / 2);
            for (0..nl) |layer| for (0..tail_rows) |row| {
                const from = if (row + 1 < tail_rows) src.*[(layer * tail_rows + row + 1) * channels ..][0..channels] else qkv[st * channels ..][0..channels];
                @memcpy(want.*[(layer * tail_rows + row) * channels ..][0..channels], from);
            };
            try check.expect(!std.mem.eql(u16, src.*, want.*), "conv reference has no shift witness", .{});
        }
        const backed = try launch(gpu, s, fun, initial, qkv, false, a);
        for (0..2) |st| {
            try check.sameBytes("conv populated backup", backed.before[st], std.mem.sliceAsBytes(initial[st]));
            try check.sameBytes("conv populated advance", backed.tail[st], std.mem.sliceAsBytes(expected[st]));
        }
        for (0..5) |_| {
            const bare = try launch(gpu, s, fun, initial, qkv, true, a);
            for (0..2) |st| try check.sameBytes("conv null-before advance", bare.tail[st], backed.tail[st]);
        }
    }
    std.debug.print("PASS qwen35 conv_commit: null/populated before keeps exact BF16 tail shifts; 3 seeds, 5 repeats, guards\n", .{});
}
