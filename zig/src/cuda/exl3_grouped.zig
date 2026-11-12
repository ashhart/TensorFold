//! #741's Grouped interface on #837's pointer tables and independent per-expert projection widths.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const experts = @import("experts.zig");
const grouped = @import("grouped.zig");
const memory = @import("memory.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");

/// plan_rotate's and plan_finish's arguments (exl3_plan.cuh's PlanStage), in its field order.
pub const Stage = extern struct {
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
    alpha: f32,
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

    pub fn bytes(k: usize, n: usize, pairs: usize, tile: usize, sk: usize, mats: usize) !usize {
        _ = try tileBlocks(tile);
        if (pairs > std.math.maxInt(i32) or k > std.math.maxInt(i32) or n > std.math.maxInt(i32))
            return error.Exl3Shape;
        const x = try std.math.mul(usize, k, 4);
        const z = try std.math.mul(usize, try std.math.mul(usize, n, try std.math.mul(usize, sk, mats)), 4);
        return std.math.mul(usize, pairs, try std.math.add(usize, x, z));
    }

    pub fn run(g: *Grouped, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, pairs: usize, tile: usize, skip: i32) !void {
        const mt = try tileBlocks(tile);
        if (epi == .up and g.up_contract != .bf16) return error.UnsupportedEpilogue;
        const w = l.exl3 orelse return error.BadExperts;
        if (w.dims != l.dims or w.width != l.width or w.experts != l.experts or l.relu2 or
            in.stride > std.math.maxInt(i32) or in.slots > std.math.maxInt(i32)) return error.BadExperts;
        try checkAct(w, l.after);
        const up = epi == .up;
        const k = if (up) w.dims else w.width;
        const n = if (up) w.width else w.dims;
        if (k == 0 or n == 0 or k % 128 != 0 or n % 128 != 0 or in.stride < k or items > std.math.maxInt(i32))
            return error.Exl3Shape;
        const sk = try ex.plan(k, n, up);
        const mats: usize = if (up) 2 else 1;
        const size = try bytes(k, n, pairs, tile, sk, mats);
        if (items == 0 or pairs == 0) return;
        if (g.storage == null or g.storage.?.len < size) {
            const replacement = try memory.DeviceBuffer.alloc(g.d, size);
            if (g.storage) |*old| old.free();
            g.storage = replacement;
        }
        const x0 = g.storage.?.ptr;
        const x1 = x0 + pairs * k * 2;
        const z = x1 + pairs * k * 2;
        const a: Stage = .{ .items = p.items, .counts = p.counts, .members = p.members, .x = in.x, .su0 = if (up) w.suh_g else w.suh_d, .su1 = if (up) w.suh_u else w.suh_d, .sv0 = if (up) w.svh_g else w.svh_d, .sv1 = if (up) w.svh_u else w.svh_d, .x0 = x0, .x1 = x1, .z = z, .out = out, .k = @intCast(k), .n = @intCast(n), .p = @intCast(pairs), .sk = @intCast(sk), .slots = @intCast(if (up) in.slots else 0), .stride = @intCast(in.stride), .skip = skip, .epi = switch (epi) {
            .up => 2,
            .down_f32 => 0,
            .down_bf16 => 3,
        }, .act_mode = w.act_mode, .limit = w.limit, .alpha = w.alpha };
        var rot: launch.Args = .{};
        rot.add(a);
        try launch.launch(g.kernels.plan_rotate, .{ .grid = .{ .x = @intCast(items), .y = @intCast(k / 128), .z = @intCast(mats) }, .block = .{ .x = 32 } }, s, &rot);
        var args: launch.Args = .{};
        for ([_]u64{ x0, x1, if (up) w.gate else w.down, if (up) w.up else w.down, if (up) w.gate_k2 else w.down_k2, if (up) w.up_k2 else w.down_k2, p.items, p.counts, p.members, z }) |v| args.add(v);
        for ([_]usize{ k, n, pairs, sk, tile, in.slots, items, mats, if (up) w.gu_stride else 0, 1 }) |v|
            args.add(@as(i32, @intCast(v)));
        args.add(skip);
        try launch.launch(g.kernels.grouped[@backingInt(w.codebook)], .{ .grid = .{ .x = @intCast(items), .y = @intCast(n / 128), .z = @intCast(mats * sk * mt) }, .block = .{ .x = 128 } }, s, &args);
        var finish: launch.Args = .{};
        finish.add(a);
        try launch.launch(if (l.after) g.kernels.plan_finish_after else g.kernels.plan_finish, .{ .grid = .{ .x = @intCast(items), .y = @intCast(n / 128) }, .block = .{ .x = 32 } }, s, &finish);
    }
};

/// Member-tile blocks a grouped launch spans: 16-member decode items or 64-member prompt items.
fn tileBlocks(tile: usize) !usize {
    return switch (tile) {
        16 => 1,
        64 => 4,
        else => error.TileNotBuilt,
    };
}

/// The activations plan_finish runs (0 bf16, 1 fp32, 2 SwiGLU-OAI); SwiGLU-OAI needs its clamp and alpha.
fn checkAct(w: ex.Layer, after: bool) !void {
    if (w.act_mode < 0 or w.act_mode > 2) return error.BadExperts;
    if (w.act_mode == 2 and !(w.limit > 0 and w.alpha > 0)) return error.BadExperts;
    if (after and w.act_mode == 0) return error.OptionNotBuilt; // bf16 activations: no contract entry covers it
    if (after and w.act_mode == 2) return error.OptionNotBuilt; // SwiGLU-OAI's branch clamps the gate before silu
}

test "Grouped's SwiGLU-OAI takes act_mode 2 only with a positive clamp and alpha" {
    var w: ex.Layer = std.mem.zeroes(ex.Layer);
    try checkAct(w, false);
    w.act_mode = 2;
    try std.testing.expectError(error.BadExperts, checkAct(w, false));
    w.limit = 7;
    try std.testing.expectError(error.BadExperts, checkAct(w, false));
    w.alpha = 1.702;
    try checkAct(w, false);
    w.limit = -1;
    try std.testing.expectError(error.BadExperts, checkAct(w, false));
    w.act_mode = 3;
    try std.testing.expectError(error.BadExperts, checkAct(w, false));
}

test "Grouped takes the clamp-after order at fp32 activations only: not at bf16, nor on SwiGLU-OAI" {
    var w: ex.Layer = std.mem.zeroes(ex.Layer);
    w.limit = 7;
    w.alpha = 1.702;
    w.act_mode = 1;
    try checkAct(w, true);
    w.act_mode = 0;
    try std.testing.expectError(error.OptionNotBuilt, checkAct(w, true));
    w.act_mode = 2;
    try std.testing.expectError(error.OptionNotBuilt, checkAct(w, true));
}

test "Grouped scratch scales linearly with pairs, refuses unsupported tiles, and rejects overflowing geometry" {
    const one = try Grouped.bytes(2048, 512, 1, 16, 4, 2);
    try std.testing.expectEqual(@as(usize, 2048 * 4 + 512 * 4 * 2 * 4), one);
    try std.testing.expectEqual(144 * one, try Grouped.bytes(2048, 512, 144, 16, 4, 2));
    try std.testing.expectEqual(144 * one, try Grouped.bytes(2048, 512, 144, 64, 4, 2));
    for ([_]usize{ 0, 8, 15, 17, 32, 48, 128 }) |bad| {
        try std.testing.expectError(error.TileNotBuilt, Grouped.bytes(2048, 512, 144, bad, 4, 2));
    }
    try std.testing.expectError(error.Exl3Shape, Grouped.bytes(std.math.maxInt(usize), 256, 1, 16, 4, 2));
    try std.testing.expectError(error.Exl3Shape, Grouped.bytes(128, 256, @as(usize, std.math.maxInt(i32)) + 1, 16, 4, 2));
}

test "Grouped's bf16 up refuses a call that asks for the combined fp32 activation pipeline" {
    var g: Grouped = .{ .d = undefined, .kernels = undefined, .up_contract = .combined_fp32 };
    const layer: experts.Layer = .{ .format = .exl3, .up = 0, .down = 0, .dims = 128, .width = 128, .experts = 1 };
    for ([_]usize{ 0, 8, 32, 128 }) |bad| {
        try std.testing.expectError(error.TileNotBuilt, g.run(undefined, .up, undefined, layer, undefined, 0, 1, 1, bad, -1));
    }
    try std.testing.expectError(error.UnsupportedEpilogue, g.run(undefined, .up, undefined, layer, undefined, 0, 0, 0, 16, -1));
    g.up_contract = .bf16;
    try std.testing.expectError(error.BadExperts, g.run(undefined, .up, undefined, layer, undefined, 0, 0, 0, 16, -1));
}

test "Grouped.run launches the plan tile's grid and compact pair stride, dropping no member at tile 64" {
    const abi = @import("abi.zig");
    const Trace = struct {
        var n: usize = 0;
        var grids: [3][3]u32 = undefined;
        var rot: Stage = undefined;
        var fin: Stage = undefined;
        var ptrs: [10]u64 = undefined;
        var ints: [11]i32 = undefined;
        fn launchFn(_: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            grids[n] = .{ gx, gy, gz };
            if (n == 0) rot = @as(*const Stage, @ptrCast(@alignCast(args.?[0].?))).*;
            if (n == 1) {
                for (&ptrs, 0..) |*p, i| p.* = @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
                for (&ints, 0..) |*v, i| v.* = @as(*const i32, @ptrCast(@alignCast(args.?[10 + i].?))).*;
            }
            if (n == 2) fin = @as(*const Stage, @ptrCast(@alignCast(args.?[0].?))).*;
            n += 1;
            return 0;
        }
    };
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launchFn;
    var ks: ex.Experts = undefined;
    ks.plan_rotate = .{ .d = &d, .handle = @ptrFromInt(1) };
    ks.plan_finish = .{ .d = &d, .handle = @ptrFromInt(2) };
    ks.plan_finish_after = .{ .d = &d, .handle = @ptrFromInt(3) };
    for (&ks.grouped, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(10 + i) };
    const base: u64 = 0x4000_0000;
    var g: Grouped = .{ .d = &d, .kernels = &ks, .storage = .{ .d = &d, .ptr = base, .len = 1 << 28 } };
    const w: ex.Layer = .{ .gate = 0x10, .up = 0x20, .down = 0x30, .gate_k2 = 0x40, .up_k2 = 0x50, .down_k2 = 0x60, .suh_g = 1, .suh_u = 2, .suh_d = 3, .svh_g = 4, .svh_u = 5, .svh_d = 6, .dims = 2048, .width = 512, .experts = 257, .codebook = .mul1 };
    const l: experts.Layer = .{ .format = .exl3, .exl3 = w, .up = 0, .down = 0, .dims = 2048, .width = 512, .experts = 257 };
    const pl: grouped.Plan = .{ .members = 0x100, .items = 0x200, .counts = 0x300, .rank = 0, .hist = 0 };
    const s: Stream = .{ .d = &d, .handle = null };
    const grp: experts.Grouped = .{ .exl3 = &g };
    for ([_]struct { tile: usize, pairs: usize, items: usize, epi: experts.Epi, prompt: bool }{
        .{ .tile = 16, .pairs = 9, .items = 9, .epi = .up, .prompt = false },
        .{ .tile = 16, .pairs = 144, .items = 153, .epi = .down_f32, .prompt = false },
        .{ .tile = 64, .pairs = 512, .items = 136, .epi = .up, .prompt = false },
        .{ .tile = 64, .pairs = 576, .items = 266, .epi = .down_bf16, .prompt = true },
    }) |c| {
        Trace.n = 0;
        const up = c.epi == .up;
        const k: usize = if (up) 2048 else 512;
        const n: usize = if (up) 512 else 2048;
        const sk: usize = try ex.plan(k, n, up);
        const mats: usize = if (up) 2 else 1;
        const in: experts.Rows = .{ .x = 0x9000, .stride = k, .slots = if (up) 9 else 0 };
        if (c.prompt) try grp.prompt(s, c.epi, in, l, pl, 0xa000, c.items, c.pairs, c.tile, -1) else try grp.decode(s, c.epi, in, l, pl, 0xa000, c.items, c.pairs, c.tile, -1);
        try std.testing.expectEqual(@as(usize, 3), Trace.n);
        try std.testing.expectEqual(@as(u32, @intCast(mats * sk * (c.tile / 16))), Trace.grids[1][2]);
        try std.testing.expectEqual(@as(i32, @intCast(c.pairs)), Trace.rot.p);
        try std.testing.expectEqual(@as(i32, @intCast(c.pairs)), Trace.fin.p);
        try std.testing.expectEqual(@as(i32, @intCast(c.pairs)), Trace.ints[2]);
        try std.testing.expectEqual(@as(i32, @intCast(c.tile)), Trace.ints[4]);
        try std.testing.expectEqual(base + c.pairs * k * 2, Trace.rot.x1);
        try std.testing.expectEqual(base + c.pairs * k * 4, Trace.rot.z);
        try std.testing.expectEqual(Trace.rot.x1, Trace.ptrs[1]);
        try std.testing.expectEqual(Trace.rot.z, Trace.ptrs[9]);
        if (c.tile == 64 and up) {
            var seen: [64]usize = @splat(0);
            const mt: u32 = @intCast((@as(usize, @intCast(Trace.ints[4])) + 15) / 16);
            for (0..Trace.grids[1][2]) |zz| {
                const mtile = zz % mt;
                for (0..16) |tx| seen[mtile * 16 + tx] += 1;
            }
            for (seen) |hits| try std.testing.expectEqual(mats * sk, hits);
        }
    }
}
