//! DeepSeek-V4.1's Metal layer on this GPU: float64 stage by stage, rows at once == one at a time, a stack's steps.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");
const check_moe = @import("check_moe.zig");

const testing = std.testing;
const limit = 1280;

test {
    _ = @import("ref.zig");
    _ = @import("forward_test.zig");
    _ = @import("zerocopy_test.zig");
    _ = @import("prompt_test.zig");
}

test "dsv41 metal: every stage of every layer kind within its float64 bound" {
    const a = testing.allocator;
    const r = try fixture.Rig.init(a, testing.io, limit, 41);
    defer r.deinit();
    const p0 = 1101;
    const rows = 3;
    const H = r.c.hidden;
    fixture.fillPast(&r.c, &r.slot, p0, 7);
    fixture.fillInput(r, rows, 11);
    var bad: usize = 0;
    for (r.layers, 0..) |*w, li| {
        const x = try a.dupe(u16, r.io[0].slice(u16, rows * 4 * H));
        defer a.free(x);
        const pre = try a.dupe(f32, r.io[2].slice(f32, rows * 4));
        defer a.free(pre);
        const carry = if (r.slot.regions[li].carry) |cb| try a.dupe(f32, cb.slice(f32, 4 * r.c.head_dim)) else null;
        defer if (carry) |v| a.free(v);
        const ctx = r.ctx(&r.slot, p0, rows);
        try r.s.writeRows(&r.c, p0, rows);
        const pass = fixture.Pass.begin(r.queue);
        const s = r.streams();
        const pr = r.pres();
        try pkg.layer.encode(&ctx, pass.e, w, .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
        try pass.finish();
        var pb: probe.Probe = .{ .a = a, .r = r, .w = w, .rows = rows, .p0 = p0 };
        defer pb.deinit();
        try check_attn.check(&pb, .{ .x = x, .pre = pre, .carry = carry });
        try check_moe.check(&pb, pr[1], s[1]);
        bad += pb.report();
        @memcpy(r.io[0].slice(u16, rows * 4 * H), r.io[1].slice(u16, rows * 4 * H));
        @memcpy(r.io[2].slice(f32, rows * 4), r.io[3].slice(f32, rows * 4));
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// Token rows [n][4][hidden] and pre mixes [n][4], the inputs of n consecutive decode steps.
const Inputs = struct { x: []u16, pre: []f32 };

fn inputs(a: std.mem.Allocator, r: *const fixture.Rig, n: usize, seed: u64) !Inputs {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const x = try a.alloc(u16, n * 4 * r.c.hidden);
    for (0..n) |i| for (0..r.c.hidden) |d| {
        const v = @import("ref.zig").toBf((rng.float(f32) * 2 - 1) * 2);
        for (0..4) |k| x[(i * 4 + k) * r.c.hidden + d] = v;
    };
    const pre = try a.alloc(f32, n * 4);
    for (0..n) |i| for (0..4) |k| {
        pre[i * 4 + k] = if (k == 0) 1 else 0;
    };
    return .{ .x = x, .pre = pre };
}

/// The stack over `n` steps from p0 in launches of `chunk` rows on `slot`: each step's output stream and pre mix.
fn runChunks(a: std.mem.Allocator, r: *fixture.Rig, slot: *const pkg.kv.Slot, in: Inputs, p0: u32, chunk: usize) !Inputs {
    const H = r.c.hidden;
    const n = in.pre.len / 4;
    const out: Inputs = .{ .x = try a.alloc(u16, n * 4 * H), .pre = try a.alloc(f32, n * 4) };
    const last = r.layers.len % 2;
    var at: usize = 0;
    while (at < n) : (at += chunk) {
        const k = @min(chunk, n - at);
        @memcpy(r.io[0].slice(u16, k * 4 * H), in.x[at * 4 * H ..][0 .. k * 4 * H]);
        @memcpy(r.io[2].slice(f32, k * 4), in.pre[at * 4 ..][0 .. k * 4]);
        const p: u32 = p0 + @as(u32, @intCast(at));
        try r.s.writeRows(&r.c, p, @intCast(k));
        const ctx = r.ctx(slot, p, @intCast(k));
        const pass = fixture.Pass.begin(r.queue);
        try pkg.layer.encodeStack(&ctx, pass.e, r.layers, r.streams(), r.pres());
        try pass.finish();
        @memcpy(out.x[at * 4 * H ..][0 .. k * 4 * H], r.io[last].slice(u16, k * 4 * H));
        @memcpy(out.pre[at * 4 ..][0 .. k * 4], r.io[2 + last].slice(f32, k * 4));
    }
    return out;
}

fn reset(r: *fixture.Rig, to: *const pkg.kv.Slot, from: *const pkg.kv.Slot) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    to.copyFrom(cb, from);
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CommandBufferFailed;
}

/// Bytes of every region that differ between two slots.
fn stateDiff(a: *const pkg.kv.Slot, b: *const pkg.kv.Slot) usize {
    var n: usize = 0;
    for (a.regions, b.regions) |x, y| {
        const pairs = [_][2]?mtl.Buffer{ .{ x.window, y.window }, .{ x.pool, y.pool }, .{ x.keys, y.keys }, .{ x.carry, y.carry } };
        for (pairs) |pr| if (pr[0]) |bx| {
            const len = bx.length();
            for (bx.contents()[0..len], pr[1].?.contents()[0..len]) |u, v| n += @intFromBool(u != v);
        };
    }
    return n;
}

/// One stream's `n` steps decoded in launches of `chunk` rows and one row at a time: outputs and state bit for bit.
fn compareChunks(a: std.mem.Allocator, r: *fixture.Rig, pristine: *const pkg.kv.Slot, p0: u32, n: usize, chunk: usize, seed: u64) !usize {
    var one = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer one.deinit();
    var many = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer many.deinit();
    try reset(r, &one, pristine);
    try reset(r, &many, pristine);
    const in = try inputs(a, r, n, seed);
    defer a.free(in.x);
    defer a.free(in.pre);
    const x1 = try runChunks(a, r, &one, in, p0, 1);
    defer a.free(x1.x);
    defer a.free(x1.pre);
    const xn = try runChunks(a, r, &many, in, p0, chunk);
    defer a.free(xn.x);
    defer a.free(xn.pre);
    var bad: usize = 0;
    for (x1.x, xn.x) |u, v| bad += @intFromBool(u != v);
    for (x1.pre, xn.pre) |u, v| bad += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    const st = stateDiff(&one, &many);
    const word = if (bad + st == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal stack ({d} layers): {d} steps from {d}, {d} rows a launch vs one: {d} output and {d} state bytes differ\n", .{ word, r.layers.len, n, p0, chunk, bad, st });
    return bad + st;
}

test "dsv41 metal: R rows in one launch equal the same rows one at a time, R = 1..16" {
    const a = testing.allocator;
    const r = try fixture.Rig.init(a, testing.io, limit, 42);
    defer r.deinit();
    var pristine = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer pristine.deinit();
    const p0 = 1101;
    fixture.fillPast(&r.c, &pristine, p0, 8);
    var bad: usize = 0;
    for (1..17) |rows| bad += try compareChunks(a, r, &pristine, p0, rows, rows, 100 + rows);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal: the stack's 8 decode steps, 4 rows a launch equal one row a launch" {
    const a = testing.allocator;
    const r = try fixture.Rig.init(a, testing.io, limit, 43);
    defer r.deinit();
    var pristine = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer pristine.deinit();
    const p0 = 1104;
    fixture.fillPast(&r.c, &pristine, p0, 9);
    try testing.expectEqual(@as(usize, 0), try compareChunks(a, r, &pristine, p0, 8, 4, 77));
}

test "dsv41 metal: shapes the kernels do not fix are refused (attention, an Engram head of 128)" {
    const tiny = host.config.tiny();
    try testing.expectError(error.UnsupportedAttention, pkg.pipelines.check(&tiny));
    var c = fixture.forwardConfig();
    try pkg.pipelines.check(&c);
    c.engram_head_dim = 128;
    try testing.expectError(error.UnsupportedEngram, pkg.pipelines.check(&c));
}

test "dsv41 metal: planted faults turn their stage checks red" {
    const a = testing.allocator;
    const r = try fixture.Rig.init(a, testing.io, limit, 44);
    defer r.deinit();
    const p0 = 1101;
    const rows = 3;
    fixture.fillPast(&r.c, &r.slot, p0, 10);
    fixture.fillInput(r, rows, 12);
    const w = &r.layers[1];
    const x = try a.dupe(u16, r.io[0].slice(u16, rows * 4 * r.c.hidden));
    defer a.free(x);
    const pre = try a.dupe(f32, r.io[2].slice(f32, rows * 4));
    defer a.free(pre);
    const carry = try a.dupe(f32, r.slot.regions[1].carry.?.slice(f32, 4 * r.c.head_dim));
    defer a.free(carry);
    const ctx = r.ctx(&r.slot, p0, rows);
    try r.s.writeRows(&r.c, p0, rows);
    const pass = fixture.Pass.begin(r.queue);
    try pkg.layer.encode(&ctx, pass.e, w, .{ .x = r.streams()[0], .pre = r.pres()[0], .x_out = r.streams()[1], .pre_out = r.pres()[1] });
    try pass.finish();
    // the sink of head 0, one mirror copy of row 0's window key and the carry's first score, each nudged
    const sink = @as([*]f32, @ptrCast(@alignCast(w.arena.contents() + w.sink.off)));
    sink[0] += 4;
    const win = r.slot.regions[1].window.slice(u16, 2 * @as(usize, pkg.kv.windowSlots(&r.c)) * r.c.head_dim);
    win[pkg.kv.windowRows(&r.c, p0)[1] * r.c.head_dim] ^= 1;
    carry[2 * r.c.head_dim] += 0.25;
    var pb: probe.Probe = .{ .a = a, .r = r, .w = w, .rows = rows, .p0 = p0 };
    defer pb.deinit();
    try check_attn.check(&pb, .{ .x = x, .pre = pre, .carry = carry });
    var red: usize = 0;
    for (pb.stages.items) |s| for ([_][]const u8{ "csa2 attention", "window write", "compressor pool (ratio 2)" }) |n| {
        if (std.mem.eql(u8, s.name, n) and s.bad > 0) red += 1;
        if (std.mem.eql(u8, s.name, n)) std.debug.print("control {s}: {d} of {d} values red (worst x{d:.1})\n", .{ n, s.bad, s.n, s.worst_ratio });
    };
    std.debug.print("{s} dsv41 metal controls: {d} of 3 planted faults red (sink, window mirror, carry)\n", .{ if (red == 3) "PASS" else "FAIL", red });
    try testing.expectEqual(@as(usize, 3), red);
}
