//! Speed-up mode's slot commands: rank 0 sends each slot operation before it runs it; rank 1 runs the same, in order.
const std = @import("std");
const eng = @import("engine.zig");
const slots_mod = @import("slots.zig");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const Engine = eng.Engine;
const Slots = slots_mod.Slots;
const Win = slots_mod.Win;
const Draft = slots_mod.Draft;

pub const Kind = enum(u32) { begin = 1, chunk, window, keep, draft, release, save, restore, drop, persist, load, forget };

/// Rank 0 of a pair sends the command; one Mac sends nothing.
pub fn send(e: *Engine, kind: Kind, words: []const u32) !void {
    const ep = e.ep orelse return;
    if (ep.rank != 0) return error.FollowsPeer;
    try ep.ctl.sendCommand(@backingInt(kind), words);
}

/// A window command: rank 0's last window digest, then each window's slot, pending token, held count and host drafts.
pub fn windowWords(out: *std.ArrayList(u32), gpa: std.mem.Allocator, digest: u64, wins: []const Win) !void {
    out.clearRetainingCapacity();
    try out.appendSlice(gpa, &.{ @truncate(digest), @truncate(digest >> 32), @intCast(wins.len) });
    for (wins) |w| {
        try out.appendSlice(gpa, &.{ w.slot, w.pending, w.held, @intCast(w.tokens.len) });
        try out.appendSlice(gpa, w.tokens);
    }
}

/// A window command's windows into `out` (their drafts point into `words`) and rank 0's digest.
pub fn readWindows(words: []const u32, out: []Win) !struct { digest: u64, n: usize } {
    if (words.len < 3 or words[2] == 0 or words[2] > out.len) return error.CommandOutOfStep;
    const n = words[2];
    var at: usize = 3;
    for (out[0..n]) |*w| {
        if (at + 4 > words.len or at + 4 + words[at + 3] > words.len) return error.CommandOutOfStep;
        w.* = .{ .slot = words[at], .pending = words[at + 1], .held = words[at + 2], .tokens = words[at + 4 ..][0..words[at + 3]] };
        at += 4 + words[at + 3];
    }
    if (at != words.len) return error.CommandOutOfStep;
    return .{ .digest = @as(u64, words[0]) | @as(u64, words[1]) << 32, .n = n };
}

/// A draft command: each stream's slot, whether it absorbs the prompt's last row, its depth and its next tokens.
pub fn draftWords(out: *std.ArrayList(u32), gpa: std.mem.Allocator, reqs: []const Draft) !void {
    out.clearRetainingCapacity();
    try out.append(gpa, @intCast(reqs.len));
    for (reqs) |r| {
        try out.appendSlice(gpa, &.{ r.slot, @intFromBool(r.prompt), r.depth, @intCast(r.follow.len) });
        try out.appendSlice(gpa, r.follow);
    }
}

/// A draft command's streams into `out` (their tokens point into `words`).
pub fn readDrafts(words: []const u32, out: []Draft) !usize {
    if (words.len < 1 or words[0] == 0 or words[0] > out.len) return error.CommandOutOfStep;
    var at: usize = 1;
    for (out[0..words[0]]) |*d| {
        if (at + 4 > words.len or at + 4 + words[at + 3] > words.len) return error.CommandOutOfStep;
        d.* = .{ .slot = words[at], .prompt = words[at + 1] != 0, .depth = words[at + 2], .follow = words[at + 4 ..][0..words[at + 3]] };
        at += 4 + words[at + 3];
    }
    if (at != words.len) return error.CommandOutOfStep;
    return words[0];
}

/// Rank 1: run rank 0's requests and slot commands in order until rank 0 closes or this Mac stops following.
pub fn follow(e: *Engine, sl: *Slots) !void {
    const ep = e.ep orelse return;
    defer sl.flush() catch {}; // the open command buffer's pool is this thread's
    var wins: [st.max_rows]Win = undefined;
    var drafts: [st.max_rows]Draft = undefined;
    var dummy: u8 = 0;
    const quiet: eng.Out = .{ .ctx = &dummy, .prefilled = eng.Quiet.prefilled, .tokens = eng.Quiet.tokens, .cancelled = eng.Quiet.cancelled };
    while (try ep.ctl.waitCommand()) |cmd| switch (cmd) {
        .request => |r| _ = e.generate(r.prompt, r.max_tokens, r.eos, r.depth, quiet) catch |err| switch (err) {
            error.ContextFull, error.EmptyPrompt => continue, // refused before its first step, on rank 0 too
            else => return err,
        },
        .lanes => |l| try apply(sl, l.kind, l.words, &wins, &drafts),
    };
}

/// One slot command, as rank 0 ran it.
pub fn apply(sl: *Slots, kind: u32, w: []const u32, wins: []Win, drafts: []Draft) !void {
    const k = std.enums.fromInt(Kind, kind) orelse return error.CommandOutOfStep;
    const need: usize = switch (k) {
        .begin, .keep, .restore, .forget => 2,
        .chunk, .window, .save, .persist => 3,
        .load => 4,
        .draft, .release, .drop => 1,
    };
    if (w.len < need) return error.CommandOutOfStep;
    switch (k) {
        .begin => try sl.begin(w[0], w[2..], w[1] != 0),
        .chunk => try sl.chunk(w[0], w[1], w[2]),
        .window => {
            const r = try readWindows(w, wins);
            if (r.digest != sl.digest) { // the last window's picks differ between the Macs: their caches have parted
                std.log.err("speed-up mode: window {d}'s picks differ from rank 0's", .{sl.windows});
                return error.PairOutOfStep;
            }
            try sl.window(wins[0..r.n]);
        },
        .keep => try sl.keep(w[0], w[1]),
        .draft => try sl.draftAll(drafts[0..try readDrafts(w, drafts)]),
        .release => sl.release(w[0]),
        .save => _ = try sl.save(w[1], w[2], w[0]),
        .restore => try sl.restore(w[0], w[1]),
        .drop => sl.drop(w[0]),
        .persist, .load => {
            learned(sl, k, w) catch |err| { // replied as failed: rank 0 forgets the state and the pair stays in step
                std.log.warn("speed-up mode: this Mac's half of a learned state failed ({s})", .{@errorName(err)});
                return sl.e.ep.?.ctl.reply(false);
            };
            try sl.e.ep.?.ctl.reply(true);
        },
        .forget => forgetHalf(sl, @as(u64, w[0]) | @as(u64, w[1]) << 32),
    }
}

/// Rank 1's half of a forgotten learned state removed (no reply: rank 0 goes on).
fn forgetHalf(sl: *Slots, key: u64) void {
    var buf: [1100]u8 = undefined;
    _ = std.c.unlink(snapshot.path(&buf, sl.learned orelse return, key, 1) catch return);
}

/// Rank 1's half of a learned state (--learn): snapshot w[0] written to its file, or read back from it as w[0].
fn learned(sl: *Slots, k: Kind, w: []const u32) !void {
    var buf: [1100]u8 = undefined;
    const file = try snapshot.path(&buf, sl.learned orelse return error.NotLearning, @as(u64, w[1]) | @as(u64, w[2]) << 32, 1);
    if (k == .persist) return sl.writeSnap(w[0], file);
    _ = try sl.readSnap(w[0], w[3], file);
}

test "draft commands carry every stream's head request" {
    const gpa = std.testing.allocator;
    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(gpa);
    const reqs = [_]Draft{ .{ .slot = 3, .prompt = true, .follow = &.{9}, .depth = 2 }, .{ .slot = 1, .prompt = false, .follow = &.{ 4, 5 }, .depth = 0 } };
    try draftWords(&words, gpa, &reqs);
    var back: [st.max_rows]Draft = undefined;
    const n = try readDrafts(words.items, &back);
    try std.testing.expectEqual(@as(usize, 2), n);
    for (reqs, back[0..n]) |a, b| {
        try std.testing.expectEqual(a.slot, b.slot);
        try std.testing.expectEqual(a.prompt, b.prompt);
        try std.testing.expectEqual(a.depth, b.depth);
        try std.testing.expectEqualSlices(u32, a.follow, b.follow);
    }
    try std.testing.expectError(error.CommandOutOfStep, readDrafts(words.items[0 .. words.items.len - 1], &back));
}

test "window commands carry every window's rows and the digest" {
    const gpa = std.testing.allocator;
    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(gpa);
    const wins = [_]Win{
        .{ .slot = 2, .pending = 11, .held = 2, .tokens = &.{} },
        .{ .slot = 0, .pending = 12, .held = 0, .tokens = &.{ 5, 6, 7 } },
    };
    try windowWords(&words, gpa, 0x0123456789abcdef, &wins);
    var back: [st.max_rows]Win = undefined;
    const r = try readWindows(words.items, &back);
    try std.testing.expectEqual(@as(u64, 0x0123456789abcdef), r.digest);
    try std.testing.expectEqual(@as(usize, 2), r.n);
    for (wins, back[0..2]) |a, b| {
        try std.testing.expectEqual(a.slot, b.slot);
        try std.testing.expectEqual(a.pending, b.pending);
        try std.testing.expectEqual(a.held, b.held);
        try std.testing.expectEqualSlices(u32, a.tokens, b.tokens);
    }
    try std.testing.expectError(error.CommandOutOfStep, readWindows(words.items[0 .. words.items.len - 1], &back));
    words.items[2] = 0;
    try std.testing.expectError(error.CommandOutOfStep, readWindows(words.items, &back));
}
