//! Pure logit selection, verify-batch logit planning, and logprob-row helpers for Inkling 2 Mini.
const std = @import("std");
const lanes = @import("lanes");
const fwd = @import("forward.zig");

pub const ring = 4096;
pub const SlotReq = struct { rows: usize, sampling: ?lanes.Sampling, has_masks: bool, has_logprobs: bool };
pub const BatchPlan = struct { mode: fwd.LogitMode, write_logits: bool };

/// Waits and releases flight command buffers after copying any failure text into `err_buf`.
pub fn settleFlight(cbs: anytype, landed: *u64, end: u64, err_buf: []u8) ?[]const u8 {
    defer for (cbs) |cb| cb.release();
    for (cbs) |cb| cb.wait();
    for (cbs) |cb| if (cb.failure()) |msg| {
        const text = std.mem.span(msg);
        const n = @min(text.len, err_buf.len);
        @memcpy(err_buf[0..n], text[0..n]);
        return err_buf[0..n];
    };
    landed.* = @max(landed.*, end);
    return null;
}

pub fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
    const t = sampling orelse return null;
    return if (t.temperature > 0) t else null;
}

pub fn planVerifyBatch(slots: []const SlotReq) BatchPlan {
    var any_logits = false;
    var any_argmax = false;
    for (slots) |slot| {
        const can_argmax = sampled(slot.sampling) == null and !slot.has_masks;
        if (can_argmax) any_argmax = true;
        if (!can_argmax or slot.has_logprobs) any_logits = true;
    }
    const mode: fwd.LogitMode = if (!any_logits)
        .argmax
    else if (any_argmax)
        .both
    else
        .logits;
    return .{ .mode = mode, .write_logits = any_logits };
}

/// The first index of the greatest logit, as draw's scan picks it: a NaN is picked only at index 0.
pub fn firstGreatest(l: []const f32) u32 {
    if (l[0] != l[0]) return 0;
    const V = @Vector(16, f32);
    const least: V = @splat(-std.math.inf(f32));
    var top = least;
    var i: usize = 0;
    while (i + 16 <= l.len) : (i += 16) {
        const v: V = l[i..][0..16].*;
        top = @max(top, @select(f32, v != v, least, v));
    }
    var best = @reduce(.Max, top);
    for (l[i..]) |w| if (w > best) {
        best = w;
    };
    const want: V = @splat(best);
    i = 0;
    while (i + 16 <= l.len) : (i += 16) {
        const v: V = l[i..][0..16].*;
        if (@reduce(.Or, v == want)) break;
    }
    for (l[i..], i..) |w, id| if (w == best) return @intCast(id);
    return 0;
}

pub fn firstLogprobRow(l: []const f32, tokens: []const u32, next: usize, count: u8) lanes.LogprobRow {
    std.debug.assert(tokens.len == ring and next > 0);
    return logprobRow(l, tokens[(next - 1) % ring], count);
}

pub fn logprobRow(l: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, w);
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, w) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = l[pick] - lse, .count = 0 };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        if (filled == top and w <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and w > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = w - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "a greedy pick is the scan's: the first of equal maxima, and a NaN only at index 0" {
    const scan = struct {
        fn pick(l: []const f32) u32 {
            var best: usize = 0;
            for (l, 0..) |w, id| if (w > l[best]) {
                best = id;
            };
            return @intCast(best);
        }
    }.pick;
    var l: [203]f32 = undefined;
    for (0..64) |round| {
        for (&l, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 37 + round * 11) % 101)) - 50);
        // Equal maxima, NaNs, both infinities and zeros of both signs, at places that move with the round.
        l[(round * 7) % l.len] = 60;
        l[(round * 13 + 5) % l.len] = 60;
        if (round % 3 == 0) l[(round * 5 + 1) % l.len] = std.math.nan(f32);
        if (round % 4 == 0) l[(round * 17 + 2) % l.len] = std.math.inf(f32);
        if (round % 5 == 0) l[(round * 3) % l.len] = -std.math.inf(f32);
        if (round % 7 == 0) l[0] = std.math.nan(f32);
        try std.testing.expectEqual(scan(&l), firstGreatest(&l));
        try std.testing.expectEqual(scan(l[0..17]), firstGreatest(l[0..17]));
    }
    var z = [_]f32{ -0.0, 0.0, -std.math.inf(f32) };
    try std.testing.expectEqual(@as(u32, 0), firstGreatest(&z));
    z = .{ -std.math.inf(f32), std.math.nan(f32), -std.math.inf(f32) };
    try std.testing.expectEqual(@as(u32, 0), firstGreatest(&z));
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [ring]u32 = @splat(0);
    tokens[2] = 3;
    const row = firstLogprobRow(&logits, &tokens, 3, 0);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

test "settleFlight checks every earlier command buffer before advancing landed and releases all parts" {
    const Mock = struct {
        const Self = @This();
        buf: [64]u8 = @splat(0),
        has_msg: bool = false,
        waits: usize = 0,
        releases: usize = 0,
        const Ref = struct {
            m: *Self,
            pub fn wait(r: @This()) void {
                r.m.waits += 1;
            }
            pub fn release(r: @This()) void {
                r.m.releases += 1;
                // Clobber the backing buffer on release to prove settleFlight copied failure text first.
                @memset(&r.m.buf, 'X');
                r.m.buf[r.m.buf.len - 1] = 0;
                r.m.has_msg = false;
            }
            pub fn failure(r: @This()) ?[*:0]const u8 {
                return if (r.m.has_msg) @ptrCast(&r.m.buf) else null;
            }
        };
        fn setMsg(self: *Self, text: ?[]const u8) void {
            if (text) |msg| {
                std.debug.assert(msg.len < self.buf.len);
                @memcpy(self.buf[0..msg.len], msg);
                self.buf[msg.len] = 0;
                self.has_msg = true;
            } else {
                self.has_msg = false;
            }
        }
    };
    const slots = @typeInfo(@TypeOf(@as(fwd.Flight, undefined).cbs)).array.len;
    try std.testing.expectEqual(@as(usize, 5), slots);
    var landed: u64 = 10;
    var err_buf: [128]u8 = undefined;
    var mocks: [slots]Mock = @splat(.{});
    var refs: [slots]Mock.Ref = undefined;
    for (&mocks, &refs) |*m, *r| r.* = .{ .m = m };

    mocks[1].setMsg("earlier part failed");
    try std.testing.expectEqualStrings("earlier part failed", settleFlight(&refs, &landed, 11, &err_buf).?);
    try std.testing.expectEqual(@as(u8, 'X'), mocks[1].buf[0]);
    try std.testing.expectEqual(@as(u64, 10), landed);
    for (mocks) |m| try std.testing.expect(m.waits == 1 and m.releases == 1);

    mocks[slots - 1].setMsg("final part failed");
    try std.testing.expectEqualStrings("final part failed", settleFlight(&refs, &landed, 11, &err_buf).?);
    try std.testing.expectEqual(@as(u8, 'X'), mocks[slots - 1].buf[0]);
    try std.testing.expectEqual(@as(u64, 10), landed);
    for (mocks) |m| try std.testing.expect(m.waits == 2 and m.releases == 2);

    var tiny: [7]u8 = undefined;
    mocks[0].setMsg("earlier part failed");
    try std.testing.expectEqualStrings("earlier", settleFlight(&refs, &landed, 11, &tiny).?);
    try std.testing.expectEqual(@as(u64, 10), landed);
    for (mocks) |m| try std.testing.expect(m.waits == 3 and m.releases == 3);

    try std.testing.expectEqual(@as(?[]const u8, null), settleFlight(&refs, &landed, 11, &err_buf));
    try std.testing.expectEqual(@as(u64, 11), landed);
    for (mocks) |m| try std.testing.expect(m.waits == 4 and m.releases == 4);
}
