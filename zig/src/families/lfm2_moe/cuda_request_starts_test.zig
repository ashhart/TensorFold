//! LFM2.5-24B-A2B message-start chunk/tap planning and the native capture/hook protocol, without a GPU.
const std = @import("std");
const be = @import("cuda_backend.zig");
const mdl = @import("cuda_model.zig");
const lanes = @import("lanes");
const cut = be.Lfm.cut;
const t = std.testing;

/// A pass's chunk ends and the points it keeps a state at (taps and cut-point ends).
const Walk = struct { ends: []usize, kept: []usize };
fn walk(from: usize, len: usize, marks: []const u32, chunks: []const u32) !Walk {
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(t.allocator);
    var kept: std.ArrayList(usize) = .empty;
    errdefer kept.deinit(t.allocator);
    var at = from;
    while (at < len) {
        const c = cut(at, len, marks, chunks);
        try ends.append(t.allocator, c.end);
        for (c.taps()) |tap| try kept.append(t.allocator, tap);
        for ([_][]const u32{ marks, chunks }) |points| {
            if (std.mem.indexOfScalar(u32, points, @intCast(c.end)) != null) {
                try kept.append(t.allocator, c.end);
                break;
            }
        }
        at = c.end;
    }
    return .{ .ends = try ends.toOwnedSlice(t.allocator), .kept = try kept.toOwnedSlice(t.allocator) };
}

test "message starts keep a state at each, grid-aligned or not (the resume point main misses)" {
    const len = 9000;
    const starts = [_]u32{ 700, 1024, 2100, 4096, 5700, 8192 };
    const w = try walk(0, len, &.{}, &starts);
    defer t.allocator.free(w.ends);
    defer t.allocator.free(w.kept);
    for (starts) |s| try t.expect(std.mem.indexOfScalar(usize, w.kept, s) != null);
    try t.expectEqual(starts.len, w.kept.len); // the grid alone keeps nothing
}

test "message starts past the tap limit end their chunk so each still keeps a state" {
    const len = 3000;
    const starts = [_]u32{ 100, 200, 300, 400, 500, 600 };
    const w = try walk(0, len, &.{}, &starts);
    defer t.allocator.free(w.ends);
    defer t.allocator.free(w.kept);
    for (starts) |s| try t.expect(std.mem.indexOfScalar(usize, w.kept, s) != null);
    try t.expectEqual(starts.len, w.kept.len);
}

/// A cut-invariant prompt-state fold: the value after p tokens depends only on [0,p), never on cuts.
const State = u64;
const seed: State = 0x243f6a8885a308d3;
fn fold(s: State, tok: u32) State {
    return s *% 6364136223846793005 +% (@as(u64, tok) +% 1);
}

/// The opaque callback context a store passes as saveState's owner: it records each kept position
/// and the tap-buffer state the pass hands it (modeling saveState reading the captured snapshot).
const Saver = struct {
    kept: std.ArrayList(u32) = .empty,
    states: std.AutoHashMapUnmanaged(u32, State) = .empty,
    st: State = seed, // the snapshot for the next hook call

    fn at(ptr: *anyopaque, _: *lanes.Stream, pos: u32) void {
        const self: *Saver = @ptrCast(@alignCast(ptr));
        self.kept.append(t.allocator, pos) catch return;
        self.states.put(t.allocator, pos, self.st) catch {};
    }
    fn deinit(self: *Saver, a: std.mem.Allocator) void {
        self.kept.deinit(a);
        self.states.deinit(a);
    }
};

fn isEnd(at: usize, points: []const u32, marks: []const u32) bool {
    const u: u32 = @intCast(at);
    return std.mem.indexOfScalar(u32, points, u) != null or std.mem.indexOfScalar(u32, marks, u) != null;
}

/// A host replay of chunk()'s cut -> tapRows -> forward -> hook protocol over a fake device.
/// It uses the real cut()/cutPoints lookahead; forward's conv state is the fold above. `boundary`
/// models a kernel that is NOT cut-invariant, to show the equality checks below are not vacuous.
const Pass = struct {
    s: *lanes.Stream,
    tokens: []const u32,
    base: State, // committed state at `from` (the seed cold, the restored snapshot when resumed)
    from: usize,
    capture: bool, // a hook is registered: ask forward for tap snapshots (H19)
    boundary: bool = false,
    saver: ?*Saver = null,
    ends: std.ArrayList(usize) = .empty,
    final: State = seed,
    captures: usize = 0,

    fn run(p: *Pass, a: std.mem.Allocator) !void {
        var at = p.from;
        var cur = p.base; // state after `at` tokens
        while (at < p.tokens.len) {
            const points = lanes.cutPoints(p.s);
            const c = cut(at, p.tokens.len, p.s.reuse.marks, points);
            var rows: [mdl.max_taps]usize = undefined;
            const taps = be.Lfm.tapRows(at, &c, p.capture, &rows);
            p.captures += taps.len;
            var tap_pos: [mdl.max_taps]usize = undefined;
            var tap_state: [mdl.max_taps]State = undefined;
            var nt: usize = 0;
            var row = at;
            while (row < c.end) : (row += 1) {
                cur = fold(cur, p.tokens[row]);
                if (p.boundary) cur +%= @as(u64, @intCast(at + 1));
                if (p.capture) for (taps) |r| if (at + r == row) {
                    tap_pos[nt] = row + 1;
                    tap_state[nt] = cur;
                    nt += 1;
                };
            }
            const done = c.end == p.tokens.len;
            if (done) p.final = cur;
            try p.ends.append(a, c.end);
            if (p.capture) if (p.saver) |sv| {
                for (tap_pos[0..nt], tap_state[0..nt]) |pos, st| {
                    sv.st = st;
                    Saver.at(@ptrCast(sv), p.s, @intCast(pos));
                }
                if (isEnd(c.end, points, p.s.reuse.marks)) {
                    sv.st = cur; // the committed last row of the chunk
                    Saver.at(@ptrCast(sv), p.s, @intCast(c.end));
                }
            };
            at = c.end;
        }
    }
};

fn promptOf(comptime n: usize) [n]u32 {
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i * 7 + 3);
    return ids;
}

/// Unplanned message starts (irregular, not the 1024 grid), four inside the first chunk.
const message_starts = [_]u32{ 100, 200, 300, 400, 500, 1100, 1200, 2100 };

test "a resume at an interior message-start tap reproduces the fresh cache-off pass past it" {
    const a = t.allocator;
    const ids = promptOf(2500);

    var cold_s = try lanes.Stream.init(a, .{ .id = "cold", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &message_starts });
    defer cold_s.deinit(a);
    var cold_saver: Saver = .{};
    defer cold_saver.deinit(a);
    cold_s.reuse.hook = .{ .ptr = &cold_saver, .at = Saver.at };
    var cold: Pass = .{ .s = &cold_s, .tokens = &ids, .base = seed, .from = 0, .capture = true, .saver = &cold_saver };
    defer cold.ends.deinit(a);
    try cold.run(a);

    // every start is kept; 100..400 are interior taps, the fifth (500) ends their chunk
    for (message_starts) |m| try t.expect(cold_saver.states.contains(m));
    try t.expectEqual(@as(usize, 500), cold.ends.items[0]);
    for ([_]u32{ 100, 200, 300, 400 }) |m| {
        try t.expect(m < cold.ends.items[0]); // captured inside the chunk, not at an end
    }
    try t.expectEqualSlices(u32, &.{ 100, 200, 300, 400, 500, 1100, 1200, 2100 }, cold_saver.kept.items);

    // fresh cache-off pass: no hook, so no captures, yet the same cut plan and final logits (H19)
    var fresh_s = try lanes.Stream.init(a, .{ .id = "fresh", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &message_starts });
    defer fresh_s.deinit(a);
    var fresh: Pass = .{ .s = &fresh_s, .tokens = &ids, .base = seed, .from = 0, .capture = false };
    defer fresh.ends.deinit(a);
    try fresh.run(a);
    try t.expectEqual(@as(usize, 0), fresh.captures);
    try t.expect(std.mem.eql(usize, cold.ends.items, fresh.ends.items));
    try t.expectEqual(cold.final, fresh.final);

    // resume at interior tap 200: restore its snapshot, replay from there with the same starts
    const m: u32 = 200;
    const base = cold_saver.states.get(m).?;
    var resume_s = try lanes.Stream.init(a, .{ .id = "resume", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &message_starts });
    defer resume_s.deinit(a);
    resume_s.reuse.saved = @ptrFromInt(1); // opaque; the restored `base` models its content
    resume_s.reuse.at = m;
    var resume_saver: Saver = .{};
    defer resume_saver.deinit(a);
    resume_s.reuse.hook = .{ .ptr = &resume_saver, .at = Saver.at };
    var resumed: Pass = .{ .s = &resume_s, .tokens = &ids, .base = base, .from = m, .capture = true, .saver = &resume_saver };
    defer resumed.ends.deinit(a);
    try resumed.run(a);

    // chunk ends may differ after a resume (H09): compare states and logits, not end positions
    var checked: usize = 0;
    var it = resume_saver.states.iterator();
    while (it.next()) |e| {
        try t.expect(e.key_ptr.* > m);
        try t.expectEqual(cold_saver.states.get(e.key_ptr.*).?, e.value_ptr.*);
        checked += 1;
    }
    try t.expect(checked > 0);
    try t.expectEqual(cold.final, resumed.final);
    try t.expectEqual(fresh.final, resumed.final); // resumed == fresh cache-off output past m
}

test "candidate cold equals main's logits; main misses the message starts (control that fails)" {
    const a = t.allocator;
    const ids = promptOf(2500);

    var cand_s = try lanes.Stream.init(a, .{ .id = "candidate", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &message_starts });
    defer cand_s.deinit(a);
    var cand_saver: Saver = .{};
    defer cand_saver.deinit(a);
    cand_s.reuse.hook = .{ .ptr = &cand_saver, .at = Saver.at };
    var cand: Pass = .{ .s = &cand_s, .tokens = &ids, .base = seed, .from = 0, .capture = true, .saver = &cand_saver };
    defer cand.ends.deinit(a);
    try cand.run(a);

    // main has no message-start cuts, only its grid marks: the same state model, different chunks
    const grid_marks = [_]u32{ 1024, 2048 };
    var main_s = try lanes.Stream.init(a, .{ .id = "main", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &.{} });
    defer main_s.deinit(a);
    main_s.reuse.marks = &grid_marks;
    var main_saver: Saver = .{};
    defer main_saver.deinit(a);
    main_s.reuse.hook = .{ .ptr = &main_saver, .at = Saver.at };
    var main: Pass = .{ .s = &main_s, .tokens = &ids, .base = seed, .from = 0, .capture = true, .saver = &main_saver };
    defer main.ends.deinit(a);
    try main.run(a);

    // cold == main: cut-invariant state/logits, even though their chunk ends genuinely differ
    try t.expectEqual(main.final, cand.final);
    try t.expect(!std.mem.eql(usize, main.ends.items, cand.ends.items));
    try t.expectEqualSlices(u32, &.{ 1024, 2048 }, main_saver.kept.items);
    // the control that fails on main: no message start is kept there, all are kept by the candidate
    for (message_starts) |m| {
        try t.expect(cand_saver.states.contains(m));
        try t.expect(!main_saver.states.contains(m));
    }
}

test "control: a boundary-dependent state fails the cold-vs-main equality an invariant fold passes" {
    const a = t.allocator;
    const ids = promptOf(2500);
    const grid_marks = [_]u32{ 1024, 2048 };

    var cand_s = try lanes.Stream.init(a, .{ .id = "dep-cand", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &message_starts });
    defer cand_s.deinit(a);
    var cand: Pass = .{ .s = &cand_s, .tokens = &ids, .base = seed, .from = 0, .capture = false, .boundary = true };
    defer cand.ends.deinit(a);
    try cand.run(a);

    var main_s = try lanes.Stream.init(a, .{ .id = "dep-main", .prompt = &ids, .max_new = 1, .drafts = false, .chunks = &.{} });
    defer main_s.deinit(a);
    main_s.reuse.marks = &grid_marks;
    var main: Pass = .{ .s = &main_s, .tokens = &ids, .base = seed, .from = 0, .capture = false, .boundary = true };
    defer main.ends.deinit(a);
    try main.run(a);

    // a real cut-dependence is detected; the equality in the previous test is therefore meaningful
    try t.expect(main.final != cand.final);
}

test "only a hook asks forward for tap snapshots; the cut plan is unchanged either way" {
    const ids = promptOf(2500);
    const few = [_]u32{ 100, 200, 300, 400, 500 };
    const c = cut(0, ids.len, &.{}, &few);
    var rows: [mdl.max_taps]usize = undefined;
    const hooked = be.Lfm.tapRows(0, &c, true, &rows);
    try t.expectEqualSlices(usize, &.{ 99, 199, 299, 399 }, hooked); // tap row = t - at - 1
    const cache_off = be.Lfm.tapRows(0, &c, false, &rows);
    try t.expectEqual(@as(usize, 0), cache_off.len); // H19: no hook, no capture work
    try t.expectEqual(@as(usize, 500), c.end); // the fifth start still ends the chunk
}

test "lfm: cold and reused passes both bound chunks by the shared cut points" {
    var prompt: [2000]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    const local_starts = [_]u32{ 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
    var s = try lanes.Stream.init(t.allocator, .{
        .id = "cold-and-reused", .prompt = &prompt, .max_new = 1, .drafts = false, .chunks = &local_starts,
    });
    defer s.deinit(t.allocator);
    const points = lanes.cutPoints(&s);
    try t.expectEqualSlices(u32, &local_starts, points);
    const cold = try walk(0, prompt.len, &.{}, points);
    defer t.allocator.free(cold.ends);
    defer t.allocator.free(cold.kept);
    for (local_starts) |at| try t.expect(std.mem.indexOfScalar(usize, cold.kept, at) != null);
    // A state kept at 128 resumes there; with no store marks the starts after it still end chunks.
    const reused = try walk(128, prompt.len, &.{}, points);
    defer t.allocator.free(reused.ends);
    defer t.allocator.free(reused.kept);
    for (local_starts) |at| if (at > 128) try t.expect(std.mem.indexOfScalar(usize, reused.kept, at) != null);
}
