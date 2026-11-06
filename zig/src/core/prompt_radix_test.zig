//! The prompt cache's tests over host memory: a resumed pass keeps exactly the pages and states it should.

const std = @import("std");
const radix = @import("prompt_radix.zig");
const Allocator = std.mem.Allocator;
const Store = radix.Store;
const Plan = radix.Plan;
const Saved = radix.Saved;

/// Pages and a family for the tests: a state is the position it was kept at, a page is 4 tokens.
const Fake = struct {
    gpa: Allocator,
    refs: [64]u32 = @splat(0),
    live: usize = 0,
    fail_save: bool = false,
    fail_restore: bool = false,

    const State = struct { at: u32 };

    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn pages(f: *Fake) radix.Pages {
        return .{ .ptr = f, .bytes = 10, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }
    fn family(f: *Fake) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = dropState } };
    }
    fn retain(ptr: *anyopaque, id: u32) void {
        of(ptr).refs[id] += 1;
    }
    fn release(ptr: *anyopaque, id: u32) void {
        of(ptr).refs[id] -= 1;
    }
    fn holders(ptr: *anyopaque, id: u32) u32 {
        return of(ptr).refs[id];
    }
    fn available(ptr: *anyopaque) usize {
        var n: usize = 0;
        for (of(ptr).refs) |r| n += @intFromBool(r == 0);
        return n;
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 100;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (f.fail_save) return error.CopyFailed;
        const st = try f.gpa.create(State);
        st.* = .{ .at = at };
        f.live += 1;
        return st;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, _: Saved) anyerror!void {
        if (of(ptr).fail_restore) return error.CopyFailed;
    }
    fn dropState(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        f.gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
        f.live -= 1;
    }
    /// `out.len` free pages taken by a stream.
    fn lane(f: *Fake, out: []u32) void {
        var i: usize = 0;
        for (out) |*p| {
            while (f.refs[i] != 0) i += 1;
            p.* = @intCast(i);
            f.refs[i] = 1;
        }
    }
    fn leave(f: *Fake, mine: []const u32) void {
        for (mine) |id| f.refs[id] -= 1;
    }
};

const rules: radix.Rules = .{ .page = 4, .min_gap = 8, .min_prompt = 0 };

fn seq(comptime n: usize, from: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = from + @as(u32, @intCast(i));
    return out;
}

/// A pass over `prompt` as a backend runs it: begin, then a keep at each mark, the stream taking the tree's pages.
fn pass(f: *Fake, s: *Store, a: Allocator, prompt: []const u32, history: u32, shared: []const u32, mine: *std.ArrayList(u32)) !Plan {
    mine.clearRetainingCapacity();
    const plan = try s.begin(a, prompt, history, shared, null, mine);
    var fresh: [32]u32 = undefined;
    const more = (prompt.len + 3) / 4 - mine.items.len;
    f.lane(fresh[0..more]);
    try mine.appendSlice(a, fresh[0..more]);
    for (plan.marks) |m| {
        var path: [32]u32 = undefined;
        const n = m / 4;
        const kept = s.keep(prompt, m, null, mine.items[0..n], path[0..n]);
        for (path[0..kept.shared], 0..) |id, i| if (mine.items[i] != id) {
            f.refs[id] += 1;
            f.refs[mine.items[i]] -= 1;
            mine.items[i] = id;
        };
    }
    return plan;
}

fn newStore(f: *Fake, r: radix.Rules) !Store {
    return Store.init(std.testing.allocator, f.family(), f.pages(), r);
}

test "a conversation resumes where its history ended, and the pages below the resume point are shared" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var mine: std.ArrayList(u32) = .empty;
    const t1 = seq(24, 0);
    var plan = try pass(&f, &s, a, &t1, 20, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqualSlices(u32, &.{20}, plan.marks);
    const first = try a.dupe(u32, mine.items);
    f.leave(mine.items);
    const t2 = seq(40, 0); // the next turn: the history grew
    plan = try pass(&f, &s, a, &t2, 36, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 20), plan.from);
    try std.testing.expectEqualSlices(u32, first[0..5], mine.items[0..5]);
    try std.testing.expectEqualSlices(u32, &.{36}, plan.marks);
    f.leave(mine.items);
    var edited = seq(40, 0); // an earlier turn edited: nothing resumes past it
    edited[9] = 999;
    plan = try pass(&f, &s, a, &edited, 36, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u64, 1), s.tree.counts.hits);
    try std.testing.expectEqual(@as(u64, 2), s.tree.counts.misses);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "marks: the history, the stable prefix, shared blocks and the last page, apart, past the resume point, before the end" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 8, .min_prompt = 0, .tail = true });
    defer s.deinit();
    const prompt = seq(48, 0);
    var prev = seq(30, 0);
    prev[28] = 77;
    // history 41 floors to 40, the prefix shared with the last prompt ends at 28, a block at 12, the last page at 44
    try std.testing.expectEqualSlices(u32, &.{ 12, 28, 40 }, try s.marks(a, &prompt, 0, 41, &.{12}, &prev));
    try std.testing.expectEqualSlices(u32, &.{40}, try s.marks(a, &prompt, 28, 41, &.{12}, &prev));
    try std.testing.expectEqualSlices(u32, &.{44}, try s.marks(a, &prompt, 0, 0, &.{}, &.{})); // no history: the last page
    try std.testing.expectEqualSlices(u32, &.{40}, try s.marks(a, &prompt, 0, 41, &.{36}, &.{})); // a block next to the history
}

test "a shared cut outlives the conversation that made it, and a state its conversation moved past goes first" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 16, .min_prompt = 0 });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 3 });
    var mine: std.ArrayList(u32) = .empty;
    // a system block of 16 tokens and a question; then the conversation's next turn
    const p1 = seq(48, 0);
    _ = try pass(&f, &s, a, &p1, 36, &.{16}, &mine);
    f.leave(mine.items);
    const p2 = seq(72, 0);
    _ = try pass(&f, &s, a, &p2, 64, &.{16}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(usize, 3), s.tree.snaps);
    // another conversation on the same block: three states do not hold a fourth, and the first turn's history goes
    var p3 = seq(48, 500);
    @memcpy(p3[0..16], p1[0..16]);
    const plan = try pass(&f, &s, a, &p3, 36, &.{16}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u32, 16), plan.from);
    try std.testing.expectEqual(@as(usize, 3), s.tree.snaps);
    try std.testing.expectEqual(@as(u32, 16), s.find(&p1).?.end);
    try std.testing.expectEqual(@as(u32, 64), s.find(&p2).?.end);
    try std.testing.expectEqual(@as(u64, 1), s.tree.counts.evicted);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "an entry that was resumed outlives one that was not, though it is older" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 2 });
    var mine: std.ArrayList(u32) = .empty;
    const first = seq(24, 0);
    _ = try pass(&f, &s, a, &first, 20, &.{}, &mine);
    f.leave(mine.items);
    const again = seq(26, 0); // resumes it and keeps nothing
    try std.testing.expectEqual(@as(u32, 20), (try pass(&f, &s, a, &again, 0, &.{}, &mine)).from);
    f.leave(mine.items);
    const second = seq(24, 100);
    _ = try pass(&f, &s, a, &second, 20, &.{}, &mine);
    f.leave(mine.items);
    const third = seq(24, 200);
    _ = try pass(&f, &s, a, &third, 20, &.{}, &mine);
    f.leave(mine.items);
    try std.testing.expect(s.find(&again) != null);
    try std.testing.expect(s.find(&seq(26, 100)) == null);
    try std.testing.expect(s.find(&seq(26, 200)) != null);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "a state past the limits is refused, and a failed copy keeps no state" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    var mine: std.ArrayList(u32) = .empty;
    const t = seq(24, 0);
    s.limit(.{ .pages = 2, .snaps = 2 });
    try std.testing.expectEqualSlices(u32, &.{}, (try pass(&f, &s, a, &t, 20, &.{}, &mine)).marks);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u64, 1), s.tree.counts.refused);
    s.limit(.{ .pages = 40, .snaps = 2 });
    f.fail_save = true;
    _ = try pass(&f, &s, a, &t, 20, &.{}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(usize, 0), s.tree.snaps);
    try std.testing.expectEqual(@as(u64, 1), s.tree.counts.failed);
    f.fail_save = false;
    _ = try pass(&f, &s, a, &t, 20, &.{}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(usize, 1), s.tree.snaps);
    // the same state again takes no second copy
    var path: [5]u32 = undefined;
    const held = [_]u32{ 0, 1, 2, 3, 4 };
    _ = s.keep(&t, 20, null, &held, &path);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    // a restore that fails drops the entry and the pass starts from the start
    f.fail_restore = true;
    const longer = seq(30, 0);
    const plan = try pass(&f, &s, a, &longer, 24, &.{}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqual(@as(u64, 2), s.tree.counts.failed);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "reclaim frees the pages of the least valuable leaf nobody holds, and spares the pages a stream holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 4 });
    var mine: std.ArrayList(u32) = .empty;
    const x = seq(24, 0);
    _ = try pass(&f, &s, a, &x, 20, &.{}, &mine);
    f.leave(mine.items);
    const y = seq(24, 100);
    _ = try pass(&f, &s, a, &y, 20, &.{}, &mine);
    const y_pages = try a.dupe(u32, mine.items[0..5]);
    f.leave(mine.items);
    var rest: [54]u32 = undefined;
    f.lane(&rest); // the pool is full
    try std.testing.expectEqual(@as(usize, 0), Fake.available(&f));
    try std.testing.expect(s.reclaim(3));
    try std.testing.expectEqual(@as(usize, 1), s.tree.snaps);
    try std.testing.expect(s.find(&seq(26, 0)) == null);
    for (y_pages) |id| f.refs[id] += 1; // a stream holds the other's pages
    try std.testing.expect(!s.reclaim(100));
    try std.testing.expectEqual(@as(usize, 1), s.tree.snaps);
    for (y_pages) |id| f.refs[id] -= 1;
    s.limit(.{ .pages = 40, .snaps = 0 });
    try std.testing.expectEqual(@as(usize, 0), s.tree.held_pages);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

test "a history every turn resumes again after one kept a longer state is a branch point: its turns never supersede it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 3 });
    var mine: std.ArrayList(u32) = .empty;
    const history = seq(24, 0); // history 20
    _ = try pass(&f, &s, a, &history, 20, &.{}, &mine);
    f.leave(mine.items);
    var turn = seq(32, 0);
    for (1..3) |k| { // two turns on that history, each keeping its own state at 28
        for (turn[20..], 0..) |*t, i| t.* = @intCast(100 * k + i);
        try std.testing.expectEqual(@as(u32, 20), (try pass(&f, &s, a, &turn, 28, &.{}, &mine)).from);
        f.leave(mine.items);
    }
    try std.testing.expectEqual(@as(usize, 3), s.tree.snaps);
    const other = seq(24, 500); // another prompt: three states hold no fourth, and the first turn's goes
    _ = try pass(&f, &s, a, &other, 20, &.{}, &mine);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u64, 1), s.tree.counts.evicted);
    for (turn[20..], 0..) |*t, i| t.* = @intCast(300 + i); // the next turn still resumes the history
    try std.testing.expectEqual(@as(u32, 20), (try pass(&f, &s, a, &turn, 28, &.{}, &mine)).from);
    f.leave(mine.items);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

/// A golden pass: where it resumed and kept, then what the tree held and had counted.
const Row = struct { from: u32, marks: []const u32, snaps: usize, pages: usize, bytes: u64, hits: u64, misses: u64, kept: u64, evicted: u64, refused: u64 };

/// Two conversations over a shared system block, a growing and an edited turn, a third prompt that evicts, a refusal.
fn goldenRun(a: Allocator, f: *Fake, s: *Store, out: *std.ArrayList(Row)) !void {
    var mine: std.ArrayList(u32) = .empty;
    var edited = seq(42, 0);
    edited[25] = 999;
    var other: [32]u32 = undefined;
    @memcpy(other[0..13], &seq(13, 0));
    @memcpy(other[13..], &seq(19, 200));
    const Pass = struct { prompt: []const u32, history: u32, shared: []const u32 };
    const passes = [_]Pass{
        .{ .prompt = &seq(30, 0), .history = 0, .shared = &.{12} },
        .{ .prompt = &seq(42, 0), .history = 30, .shared = &.{12} },
        .{ .prompt = &other, .history = 0, .shared = &.{12} },
        .{ .prompt = &edited, .history = 30, .shared = &.{12} },
        .{ .prompt = &seq(56, 0), .history = 42, .shared = &.{12} },
        .{ .prompt = &seq(30, 300), .history = 20, .shared = &.{} },
        .{ .prompt = &seq(92, 400), .history = 88, .shared = &.{} },
    };
    for (passes) |p| {
        const plan = try pass(f, s, a, p.prompt, p.history, p.shared, &mine);
        const c = s.tree.counts;
        try out.append(a, .{ .from = plan.from, .marks = plan.marks, .snaps = s.tree.snaps, .pages = s.tree.held_pages, .bytes = s.tree.held(), .hits = c.hits, .misses = c.misses, .kept = c.kept, .evicted = c.evicted, .refused = c.refused });
        f.leave(mine.items);
    }
}

test "golden: lookahead 0 and no grid, as HIP's families and Qwen's CUDA path keep, resume and evict the same" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 16, .min_prompt = 0, .tail = true });
    defer s.deinit();
    s.limit(.{ .pages = 20, .snaps = 4 });
    var rows: std.ArrayList(Row) = .empty;
    try goldenRun(a, &f, &s, &rows);
    // the store before it learned a lookahead and a grid kept exactly these
    const want = [_]Row{
        .{ .from = 0, .marks = &.{28}, .snaps = 1, .pages = 7, .bytes = 170, .hits = 0, .misses = 1, .kept = 1, .evicted = 0, .refused = 0 },
        .{ .from = 28, .marks = &.{}, .snaps = 1, .pages = 7, .bytes = 170, .hits = 1, .misses = 1, .kept = 1, .evicted = 0, .refused = 0 },
        .{ .from = 0, .marks = &.{28}, .snaps = 2, .pages = 11, .bytes = 310, .hits = 1, .misses = 2, .kept = 2, .evicted = 0, .refused = 0 },
        .{ .from = 0, .marks = &.{28}, .snaps = 3, .pages = 12, .bytes = 420, .hits = 1, .misses = 3, .kept = 3, .evicted = 0, .refused = 0 },
        .{ .from = 28, .marks = &.{40}, .snaps = 4, .pages = 15, .bytes = 550, .hits = 2, .misses = 3, .kept = 4, .evicted = 0, .refused = 0 },
        .{ .from = 0, .marks = &.{20}, .snaps = 4, .pages = 20, .bytes = 600, .hits = 2, .misses = 4, .kept = 5, .evicted = 1, .refused = 0 },
        .{ .from = 0, .marks = &.{}, .snaps = 4, .pages = 20, .bytes = 600, .hits = 2, .misses = 5, .kept = 5, .evicted = 1, .refused = 1 },
    };
    try std.testing.expectEqual(want.len, rows.items.len);
    for (want, rows.items) |w, r| {
        try std.testing.expectEqualSlices(u32, w.marks, r.marks);
        var got = r;
        got.marks = w.marks;
        try std.testing.expectEqual(w, got);
    }
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "lookahead 1: a state resumes when the token past it repeats, and not when it differs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 8, .min_prompt = 0, .lookahead = 1 });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var mine: std.ArrayList(u32) = .empty;
    const t1 = seq(24, 0);
    var plan = try pass(&f, &s, a, &t1, 20, &.{}, &mine);
    try std.testing.expectEqualSlices(u32, &.{20}, plan.marks);
    f.leave(mine.items);
    const t2 = seq(32, 0); // the next turn repeats token 20
    plan = try pass(&f, &s, a, &t2, 28, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 20), plan.from);
    f.leave(mine.items);
    var t3 = seq(32, 0); // a conversation parting exactly at token 20 cannot use the state there
    t3[20] = 999;
    plan = try pass(&f, &s, a, &t3, 28, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u32, 20), s.find(&seq(21, 0)).?.end); // its lookahead, the one token to prefill
    try std.testing.expect(s.find(&seq(20, 0)) == null); // no token past the state: nothing to resume
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

/// Pages a second conversation takes from a first's state at 20 when it keeps at 20 too, parting from it at `branch`.
fn sharedAfter(branch: usize, lookahead: u32) !usize {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 8, .min_prompt = 0, .lookahead = lookahead });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var mine: std.ArrayList(u32) = .empty;
    _ = try pass(&f, &s, a, &seq(24, 0), 20, &.{}, &mine);
    f.leave(mine.items);
    var second = seq(24, 0);
    second[branch] = 999;
    var own: [5]u32 = undefined;
    f.lane(&own);
    var path: [5]u32 = undefined;
    const kept = s.keep(&second, 20, null, &own, &path);
    f.leave(&own);
    try std.testing.expect(kept.held);
    try std.testing.expectEqual(s.tree.snaps, f.live);
    var taken: usize = 0;
    for (own, path) |o, p| taken += @intFromBool(o != p);
    return taken;
}

test "lookahead 1: a conversation shares only the pages whose next token it still repeats" {
    // parting at the state's end: the last page's rows read the token past it, so it is not shared
    try std.testing.expectEqual(@as(usize, 5), try sharedAfter(20, 0));
    try std.testing.expectEqual(@as(usize, 4), try sharedAfter(20, 1));
    // parting at an inner page edge, and one token after it
    try std.testing.expectEqual(@as(usize, 4), try sharedAfter(16, 0));
    try std.testing.expectEqual(@as(usize, 3), try sharedAfter(16, 1));
    try std.testing.expectEqual(@as(usize, 4), try sharedAfter(17, 1));
}

test "planned: marks and resumes sit on the grid, never on a page between" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 8, .min_prompt = 0, .tail = true, .planned = true, .grid = 8 });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    // a page store would cut at 12 and 40; the grid of 8 moves the block's cut to 8, the tail joins the history's 40
    try std.testing.expectEqualSlices(u32, &.{ 8, 40 }, try s.marks(a, &seq(48, 0), 0, 41, &.{12}, &.{}));
    var mine: std.ArrayList(u32) = .empty;
    var plan = try pass(&f, &s, a, &seq(24, 0), 20, &.{}, &mine);
    try std.testing.expectEqualSlices(u32, &.{16}, plan.marks); // history 20, its grid point 16
    f.leave(mine.items);
    plan = try pass(&f, &s, a, &seq(40, 0), 36, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 16), plan.from);
    f.leave(mine.items);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "planned: a keep off the grid is refused, one on it kept" {
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 8, .min_prompt = 0, .planned = true, .grid = 8 });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var own: [4]u32 = undefined;
    f.lane(&own);
    defer f.leave(&own);
    var path: [4]u32 = undefined;
    const prompt = seq(24, 0);
    try std.testing.expect(!s.keep(&prompt, 12, null, own[0..3], path[0..3]).held);
    try std.testing.expectEqual(@as(usize, 0), s.tree.snaps);
    try std.testing.expect(s.keep(&prompt, 16, null, &own, &path).held);
    try std.testing.expectEqual(@as(usize, 1), s.tree.snaps);
}

test "the tree's deepest state is the deepest at a multiple of the step it is asked for" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, rules);
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var mine: std.ArrayList(u32) = .empty;
    const prompt = seq(24, 0);
    _ = try pass(&f, &s, a, &prompt, 12, &.{}, &mine); // a state at 12, off a grid of 8
    f.leave(mine.items);
    try std.testing.expectEqual(@as(u32, 12), s.tree.deepest(&prompt, 20, 4).?.end);
    try std.testing.expect(s.tree.deepest(&prompt, 20, 8) == null);
}

test "GLM's shape, a grid of four pages and lookahead 1: a chat resumes at its history's grid point, sharing its pages" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = std.testing.allocator };
    var s = try newStore(&f, .{ .page = 4, .min_gap = 16, .min_prompt = 0, .lookahead = 1, .planned = true, .grid = 16 });
    defer s.deinit();
    s.limit(.{ .pages = 40, .snaps = 8 });
    var mine: std.ArrayList(u32) = .empty;
    var plan = try pass(&f, &s, a, &seq(40, 0), 36, &.{}, &mine);
    try std.testing.expectEqualSlices(u32, &.{32}, plan.marks);
    const first = try a.dupe(u32, mine.items[0..8]);
    f.leave(mine.items);
    plan = try pass(&f, &s, a, &seq(56, 0), 52, &.{}, &mine);
    try std.testing.expectEqual(@as(u32, 32), plan.from);
    try std.testing.expectEqualSlices(u32, first, mine.items[0..8]); // the resume adopted the first turn's pages
    try std.testing.expectEqualSlices(u32, &.{48}, plan.marks);
    f.leave(mine.items);
    try std.testing.expectEqual(@as(usize, 2), s.tree.snaps);
    try std.testing.expectEqual(@as(usize, 12), s.tree.held_pages);
    try std.testing.expectEqual(s.tree.snaps, f.live);
}

test "a store refuses rules it cannot keep: no page, or a planned grid that is no multiple of the page" {
    var f: Fake = .{ .gpa = std.testing.allocator };
    try std.testing.expectError(error.BadPromptRadixRules, newStore(&f, .{ .page = 0 }));
    try std.testing.expectError(error.BadPromptRadixRules, newStore(&f, .{ .page = 4, .planned = true }));
    try std.testing.expectError(error.BadPromptRadixRules, newStore(&f, .{ .page = 4, .planned = true, .grid = 6 }));
    var s = try newStore(&f, .{ .page = 4, .planned = true, .grid = 8 });
    s.deinit();
}
