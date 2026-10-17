//! Persistence is separate from lesson rejection: stage writer inputs and preserve accepted live values on any error.
const std = @import("std");
const neuron = @import("neuron.zig");
const rounding = @import("pocket_round.zig");
pub fn save(p: anytype, t: anytype, host: anytype) !usize {
    const P = @TypeOf(p.*);
    const n = p.up.items.len;
    p.save_pending = true;
    if (@hasDecl(@TypeOf(host), "saveBegin")) try host.saveBegin();
    const parts: usize = if (P.swiglu) 3 else 2;
    const buffer = try p.gpa.alloc(f32, 2 * parts * n);
    defer p.gpa.free(buffer);
    const up = buffer[0..n];
    const down = buffer[n..][0..n];
    const gate = buffer[2 * n ..][0..if (P.swiglu) n else 0];
    const want = buffer[parts * n ..];
    const want_up = want[0..n];
    const want_down = want[n..][0..n];
    const want_gate = want[2 * n ..][0..if (P.swiglu) n else 0];
    try roundedOnce(p, want_gate, want_up, want_down);
    @memcpy(down, p.down.items);
    const bytes = if (P.swiglu) blk: {
        @memcpy(gate, p.gate.items);
        neuron.upRow(P.of.neuron, p.up.items, up);
        const written = try host.pocketSaveSwiglu(p.moe.items, P.neurons, gate, up, down);
        neuron.fromUp(P.of.neuron, up, up);
        break :blk written;
    } else blk: {
        @memcpy(up, p.up.items);
        break :blk try host.pocketSave(p.moe.items, P.neurons, up, down);
    };
    if (!same(up, want_up) or !same(down, want_down) or !same(gate, want_gate)) {
        p.saved_mismatch = true;
        return error.PocketSavedDifferently;
    }
    try adopt(p, t, host, gate, up, down);
    if (@hasDecl(@TypeOf(host), "saveFinish")) try host.saveFinish();
    p.saved_mismatch = false;
    p.save_pending = false;
    p.snap.valid = false;
    return bytes;
}

/// Every slot of the live pocket rounded once, as asWritten rounds the open fact's: what the writer must hand back.
fn roundedOnce(p: anytype, gate: []f32, up: []f32, down: []f32) !void {
    const P = @TypeOf(p.*);
    const n = P.neurons * p.hidden;
    @memcpy(down, p.down.items);
    if (P.swiglu) {
        @memcpy(gate, p.gate.items);
        neuron.upRow(P.of.neuron, p.up.items, up);
    } else @memcpy(up, p.up.items);
    for (0..p.used.items.len) |j| {
        const g: []f32 = if (P.swiglu) gate[j * n ..][0..n] else &.{};
        try rounding.layer(P.of, p.gpa, p.hidden, P.neurons, g, up[j * n ..][0..n], down[j * n ..][0..n]);
    }
    if (P.swiglu) neuron.fromUp(P.of.neuron, up, up);
}

/// The written form taken as the live pocket, each slot it moved handed to the forward when the host takes uploads.
fn adopt(p: anytype, t: anytype, host: anytype, gate: []const f32, up: []const f32, down: []const f32) !void {
    const P = @TypeOf(p.*);
    const n = P.neurons * p.hidden;
    for (0..p.used.items.len) |j| {
        const s = j * n;
        const moved = !same(up[s..][0..n], p.up.items[s..][0..n]) or !same(down[s..][0..n], p.down.items[s..][0..n]) or (P.swiglu and !same(gate[s..][0..n], p.gate.items[s..][0..n]));
        if (!moved) continue;
        @memcpy(p.up.items[s..][0..n], up[s..][0..n]);
        @memcpy(p.down.items[s..][0..n], down[s..][0..n]);
        if (P.swiglu) @memcpy(p.gate.items[s..][0..n], gate[s..][0..n]);
        if (comptime @hasDecl(@TypeOf(host), if (P.swiglu) "pocketSwiglu" else "pocket")) try p.upload(t, host, j);
    }
}

fn same(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

test "pocket save: a narrow group that rounds to other bits again saves the writer's form, which the live pocket takes" {
    const gpa = std.testing.allocator;
    const d = 64;
    const F = @import("families.zig").nemotron_h;
    const T = struct {
        pub fn config(_: *const @This()) struct { hidden: usize } {
            return .{ .hidden = d };
        }
    };
    const Host = struct {
        uploads: *usize,
        pub fn pocket(h: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {
            h.uploads.* += 1;
        }
        pub fn pocketSave(_: @This(), _: []const usize, _: usize, up: []f32, down: []f32) !usize {
            try @import("shards.zig").round(F, std.testing.allocator, d, &.{up}, down);
            return up.len * 4;
        }
    };
    const P = @import("pocket.zig").Pocket(T, F);
    var p: P = .{ .gpa = gpa, .mode = .reply, .hidden = d, .layers = 8 };
    defer p.deinit();
    try p.moe.appendSlice(gpa, &.{3});
    try p.up.appendNTimes(gpa, 0, P.neurons * d);
    try p.down.appendNTimes(gpa, 0, P.neurons * d);
    try p.used.append(gpa, 1);
    try p.taken.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply });
    for (p.neuron(p.up.items, 0, 0, d), 0..) |*u, i| u.* = @sin(@as(f32, @floatFromInt(i)) * 0.7);
    // each output's column: neuron 0's value and the 63 others' shared one, 2.9% apart, as a real lesson left it
    for (0..P.neurons) |k| for (0..d) |o| {
        p.down.items[k * d + o] = if (k == 0) 2.962537e-4 else 3.0485913e-4;
    };
    var t: T = .{};
    var uploads: usize = 0;
    try p.asWritten(&t, Host{ .uploads = &uploads });
    try std.testing.expectEqual(@as(usize, 1), uploads);
    const once = try gpa.dupe(f32, p.down.items);
    defer gpa.free(once);
    const twice = try gpa.dupe(f32, p.down.items);
    defer gpa.free(twice);
    const up = try gpa.dupe(f32, p.up.items);
    defer gpa.free(up);
    try @import("shards.zig").round(F, gpa, d, &.{up}, twice);
    try std.testing.expect(!same(once, twice));
    _ = try p.save(&t, Host{ .uploads = &uploads });
    try std.testing.expect(!p.saved_mismatch and !p.save_pending);
    try std.testing.expect(same(p.down.items, twice));
    try std.testing.expectEqual(@as(usize, 2), uploads);
}
