//! #741's Grouped interface on #837's pointer tables and independent per-expert projection widths.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const experts = @import("experts.zig");
const grouped = @import("grouped.zig");
const memory = @import("memory.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");

const Stage = extern struct {
    items: u64,
    counts: u64,
    members: u64,
    x: u64,
    su0: u64,
    su1: u64,
    sv0: u64,
    sv1: u64,
    x0: u64,
    x1: u64,
    z: u64,
    out: u64,
    k: i32,
    n: i32,
    p: i32,
    sk: i32,
    slots: i32,
    stride: i32,
    skip: i32,
    epi: i32,
    act_mode: i32,
    limit: f32,
};

pub const UpContract = enum { bf16, combined_fp32 };

pub const Grouped = struct {
    up_contract: UpContract = .bf16,
    d: *const Driver,
    kernels: *const ex.Experts,
    storage: ?memory.DeviceBuffer = null,

    pub fn deinit(g: *Grouped) void {
        if (g.storage) |*b| b.free();
        g.storage = null;
    }

    pub fn bytes(k: usize, n: usize, items: usize, sk: usize, mats: usize) !usize {
        const p = try std.math.mul(usize, items, 64);
        if (p > std.math.maxInt(i32) or k > std.math.maxInt(i32) or n > std.math.maxInt(i32))
            return error.Exl3Shape;
        const x = try std.math.mul(usize, k, 4);
        const z = try std.math.mul(usize, try std.math.mul(usize, n, sk * mats), 4);
        return std.math.mul(usize, p, try std.math.add(usize, x, z));
    }

    pub fn run(g: *Grouped, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32, _: bool) !void {
        if (epi == .up and g.up_contract != .bf16) return error.UnsupportedEpilogue;
        const w = l.exl3 orelse return error.BadExperts;
        if (w.dims != l.dims or w.width != l.width or w.experts != l.experts or l.relu2 or
            in.stride > std.math.maxInt(i32) or in.slots > std.math.maxInt(i32)) return error.BadExperts;
        const up = epi == .up;
        const k = if (up) w.dims else w.width;
        const n = if (up) w.width else w.dims;
        if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or in.stride < k) return error.Exl3Shape;
        const sk = try ex.plan(k, n, up);
        const mats: usize = if (up) 2 else 1;
        const size = try bytes(k, n, items, sk, mats);
        if (items == 0) return;
        if (g.storage == null or g.storage.?.len < size) {
            const replacement = try memory.DeviceBuffer.alloc(g.d, size);
            if (g.storage) |*old| old.free();
            g.storage = replacement;
        }
        const pairs = items * 64;
        const x0 = g.storage.?.ptr;
        const x1 = x0 + pairs * k * 2;
        const z = x1 + pairs * k * 2;
        const a: Stage = .{ .items = p.items, .counts = p.counts, .members = p.members, .x = in.x, .su0 = if (up) w.suh_g else w.suh_d, .su1 = if (up) w.suh_u else w.suh_d, .sv0 = if (up) w.svh_g else w.svh_d, .sv1 = if (up) w.svh_u else w.svh_d, .x0 = x0, .x1 = x1, .z = z, .out = out, .k = @intCast(k), .n = @intCast(n), .p = @intCast(pairs), .sk = @intCast(sk), .slots = @intCast(if (up) in.slots else 0), .stride = @intCast(in.stride), .skip = skip, .epi = switch (epi) {
            .up => 2,
            .down_f32 => 0,
            .down_bf16 => 3,
        }, .act_mode = w.act_mode, .limit = w.limit };
        var rot: launch.Args = .{};
        rot.add(a);
        try launch.launch(g.kernels.plan_rotate, .{ .grid = .{ .x = @intCast(items), .y = @intCast(k / 128), .z = @intCast(mats) }, .block = .{ .x = 32 } }, s, &rot);
        var args: launch.Args = .{};
        for ([_]u64{ x0, x1, if (up) w.gate else w.down, if (up) w.up else w.down, if (up) w.gate_k2 else w.down_k2, if (up) w.up_k2 else w.down_k2, p.items, p.counts, p.members, z }) |v| args.add(v);
        for ([_]usize{ k, n, pairs, sk, 64, in.slots, items, mats, if (up) w.gu_stride else 0, 1 }) |v|
            args.add(@as(i32, @intCast(v)));
        args.add(skip);
        try launch.launch(g.kernels.grouped[@backingInt(w.codebook)], .{ .grid = .{ .x = @intCast(items), .y = @intCast(n / 128), .z = @intCast(mats * sk * 4) }, .block = .{ .x = 128 } }, s, &args);
        var finish: launch.Args = .{};
        finish.add(a);
        try launch.launch(g.kernels.plan_finish, .{ .grid = .{ .x = @intCast(items), .y = @intCast(n / 128) }, .block = .{ .x = 32 } }, s, &finish);
    }
};

test "Grouped scratch includes every pair and rejects overflowing geometry" {
    try std.testing.expectEqual(@as(usize, 64 * (128 * 4 + 256 * 4 * 2 * 4)), try Grouped.bytes(128, 256, 1, 4, 2));
    try std.testing.expectError(error.Exl3Shape, Grouped.bytes(std.math.maxInt(usize), 256, 1, 4, 2));
    try std.testing.expectError(error.Exl3Shape, Grouped.bytes(128, 256, std.math.maxInt(i32), 4, 2));
}

test "Grouped's bf16 up refuses a call that asks for the combined fp32 activation pipeline" {
    var g: Grouped = .{ .d = undefined, .kernels = undefined, .up_contract = .combined_fp32 };
    const layer: experts.Layer = .{ .format = .exl3, .up = 0, .down = 0, .dims = 128, .width = 128, .experts = 1 };
    try std.testing.expectError(error.UnsupportedEpilogue, g.run(undefined, .up, undefined, layer, undefined, 0, 0, -1, false));
    g.up_contract = .bf16;
    try std.testing.expectError(error.BadExperts, g.run(undefined, .up, undefined, layer, undefined, 0, 0, -1, false));
}
