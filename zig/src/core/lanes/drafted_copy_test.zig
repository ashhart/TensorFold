//! A drafter whose hold costs a pass of its own: no hold for a stream whose next round is a copy.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const dr = @import("drafter.zig");
const Proposer = @import("proposer.zig").Proposer;
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;

/// Copies the fake target's own continuation (it lands whole) whenever the context's length is a multiple of three.
const Copier = struct {
    out: [3]u32 = undefined,
    match: i64 = 0,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        c.match = 0;
        if (context.len % 3 != 0 or most < 2) return &.{};
        var history: [128]u32 = undefined;
        @memcpy(history[0..context.len], context);
        const n: usize = @intCast(@min(3, most));
        for (c.out[0..n], 0..) |*t, k| {
            t.* = fake.next(history[0 .. context.len + k], null, 0);
            history[context.len + k] = t.*;
        }
        c.match = 100;
        return c.out[0..n];
    }

    fn lastMatch(ptr: *anyopaque) i64 {
        return @as(*Copier, @ptrCast(@alignCast(ptr))).match;
    }

    const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
};

/// Each stream's last hold position, and the copy windows verified at a position a hold was asked for.
const Watch = struct {
    var inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined;
    var inner_verify: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var at: [4]?u64 = @splat(null);
    var streams: [4]?*sm.Stream = @splat(null);
    var wasted: u32 = 0;
    var copies: u32 = 0;
    var blocks: u32 = 0; // the streams' blocks held, over every hold

    fn slot(s: *sm.Stream) usize {
        for (&streams, 0..) |*x, i| if (x.* == null or x.* == s) {
            x.* = s;
            return i;
        };
        unreachable;
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        blocks += @intCast(items.len);
        // a joining stream's first hold comes before its first token is read: no copy can be foreseen for it
        for (items) |it| at[slot(it.stream)] = if (it.stream.emitted().len > 0) it.position else null;
        return inner_hold(ptr, items);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| if (w.held == 0 and w.tokens.len > 0) {
            copies += 1;
            if (at[slot(w.stream)] == w.positions[0]) wasted += 1;
        };
        return inner_verify(ptr, windows, out);
    }
};

const Run = struct { tokens: [3][64]u32 = undefined, lens: [3]usize = @splat(0), holds: u32 = 0, wasted: u32 = 0, copies: u32 = 0 };

fn run(n: usize, drafts: bool, copy_first: bool) !Run {
    Watch.at = @splat(null);
    Watch.streams = @splat(null);
    Watch.wasted = 0;
    Watch.copies = 0;
    Watch.blocks = 0;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true, .copy_first = copy_first } };
    defer head.deinit();
    var hooked = head.drafter();
    var table = hooked.vtable.*;
    Watch.inner_hold = table.hold;
    table.hold = Watch.hold;
    hooked.vtable = &table;
    var wrapped = try Drafted.init(gpa, target.backend(), hooked);
    defer wrapped.deinit();
    // the round loop's windows, before the wrapper turns held drafts into host tokens
    var seen = if (drafts) wrapped.backend() else target.backend();
    var verified = seen.vtable.*;
    Watch.inner_verify = verified.verify;
    verified.verify = Watch.verify;
    seen.vtable = &verified;
    const m: @import("config.zig").Model = .{ .exact_width = 16, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
    var cfg = try Config.init(gpa, if (drafts) wrapped.facts(m) else m, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, seen, clock.clock());
    defer engine.deinit();
    var copiers: [3]Copier = @splat(.{});
    var streams: [3]sm.Stream = undefined;
    for (streams[0..n], copiers[0..n], 0..) |*s, *c, i| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = dt.long[0 .. 12 + i], .max_new = 40, .eos = &.{96}, .drafts = drafts, .proposer = .{ .ptr = c, .vtable = &Copier.vtable } });
    defer for (streams[0..n]) |*s| s.deinit(gpa);
    for (streams[0..n]) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .holds = Watch.blocks, .wasted = Watch.wasted, .copies = Watch.copies };
    for (streams[0..n], 0..) |*s, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
    }
    return out;
}

test "a copy ahead takes the round: with copy_first no hold is asked for it, and replies equal one-token decode" {
    for ([_]usize{ 1, 3 }) |n| {
        const plain = try run(n, false, false);
        const before = try run(n, true, false);
        const after = try run(n, true, true);
        for (0..n) |i| {
            try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], before.tokens[i][0..before.lens[i]]);
            try std.testing.expectEqualSlices(u32, plain.tokens[i][0..plain.lens[i]], after.tokens[i][0..after.lens[i]]);
        }
        try std.testing.expect(before.copies > 0 and after.copies > 0);
        try std.testing.expect(before.wasted > 0); // a hold of one draft beside each copy, dropped by the copy round
        try std.testing.expectEqual(@as(u32, 0), after.wasted);
        try std.testing.expect(after.holds < before.holds);
    }
}
