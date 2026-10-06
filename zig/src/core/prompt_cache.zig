//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A family's copy of one state.
pub const Saved = *anyopaque;

/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Bytes `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
    };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.free(e);
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or std.mem.indexOfScalar(u32, starts, at) != null);
    }

    /// The longest kept state `prompt` resumes exactly: its tokens a prefix of the prompt, at least one row left.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            best = e;
        }
        return best;
    }

    /// Before a prompt pass: restore the longest state `prompt` resumes (from 0: none or a failed copy) and plan its marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque) !Plan {
        const l = try s.lookup(a, prompt, history_len, shared, starts);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        return .{ .from = from, .marks = l.marks };
    }

    /// begin without the restore, for backends that restore inside their own prompt pass and then call `resumed`.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        const e = s.find(prompt, starts) orelse {
            s.counts.misses += 1;
            return .{ .marks = try s.marks(a, prompt, 0, history_len, shared, starts, &.{}) };
        };
        return .{ .entry = e, .marks = try s.marks(a, prompt, e.at, history_len, shared, starts, e.last) };
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (at - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// The prompt pass stands at `at`: keep its state for `prompt`, evicting to fit; refused (counted) past the budget.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque) void {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return;
        s.clock += 1;
        for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n])) {
            e.used = s.clock; // the same state again: no copy
            return;
        };
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return;
        }
        while (s.held + bytes > s.budget) {
            s.remove(s.victim());
            s.counts.evicted += 1;
        }
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        e.* = .{ .tokens = tokens, .at = at, .saved = saved, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = last };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += bytes;
        s.counts.kept += 1;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) void {
        s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
    }

    /// The entry to free first: one a later-born entry extends (that conversation moved on), oldest first; else the oldest.
    fn victim(s: *const Store) usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            const moved_on = for (s.entries.items) |o| {
                if (o != e and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on and (best == null or e.used < s.entries.items[best.?].used)) best = i;
        }
        if (best) |i| return i;
        var oldest: usize = 0;
        for (s.entries.items, 0..) |e, i| if (e.used < s.entries.items[oldest].used) {
            oldest = i;
        };
        return oldest;
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

/// A family over host memory for tests: its live state is a position and a running sum of the prompt's tokens.
const Fake = struct {
    gpa: Allocator,
    at: u32 = 0,
    sum: u64 = 0,
    fail_save: bool = false,
    fail_restore: bool = false,
    live: usize = 0,

    const State = struct { at: u32, sum: u64 };

    fn snapshots(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn } };
    }
    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (f.fail_save) return error.CopyFailed;
        if (at != f.at) return error.NotAtMark;
        const st = try f.gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const f = of(ptr);
        if (f.fail_restore) return error.CopyFailed;
        const st: *State = @ptrCast(@alignCast(saved));
        f.at, f.sum = .{ st.at, st.sum };
    }
    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        f.gpa.destroy(st);
        f.live -= 1;
    }

    /// A prompt pass from `plan.from`, keeping at each mark; returns the sum a fresh pass would give.
    fn pass(f: *Fake, s: *Store, prompt: []const u32, plan: Plan) u64 {
        if (plan.from == 0) f.* = .{ .gpa = f.gpa, .fail_save = f.fail_save, .fail_restore = f.fail_restore, .live = f.live };
        var mi: usize = 0;
        for (prompt[plan.from..]) |t| {
            f.sum = f.sum *% 31 +% t;
            f.at += 1;
            if (mi < plan.marks.len and plan.marks[mi] == f.at) {
                s.keep(prompt, f.at, null);
                mi += 1;
            }
        }
        return f.sum;
    }
};

fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}

test "a growing conversation resumes each turn where the last one's history ended, and equals a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // history 5, then a two-token generation prompt
    var p = try s.begin(a, &t1, 5, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{5}, p.marks);
    try std.testing.expectEqual(fresh(&t1), f.pass(&s, &t1, p));
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 9, 7, 7, 6, 6, 9, 9 }; // the reply and a tool result, history 10
    p = try s.begin(a, &t2, 10, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqualSlices(u32, &.{10}, p.marks);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
    const edited = [_]u32{ 1, 2, 3, 8, 5, 9, 7, 7, 6, 6, 9, 9 }; // an earlier turn edited: nothing resumes past it
    p = try s.begin(a, &edited, 10, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, p));
    try std.testing.expectEqual(@as(u64, 1), s.counts.hits);
    try std.testing.expectEqual(@as(u64, 2), s.counts.misses);
}

test "an entry keys its lookahead tokens: a prompt that differs right after the state does not resume it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null)); // keeps [1 2 3 4] + 5
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 6, 9 }, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 4), s.find(&.{ 1, 2, 3, 4, 5, 6 }, &.{}).?.at);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4 }, &.{}) == null); // the lookahead token must be in the prompt
}

test "planned families resume and keep only at the request's chunk starts" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const p = try s.begin(a, &t1, 7, &.{}, &.{ 4, 6 }, null);
    try std.testing.expectEqualSlices(u32, &.{6}, p.marks); // history 7 floored to the start at 6
    _ = f.pass(&s, &t1, p);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{4}) == null); // 6 is not one of this prompt's starts
    try std.testing.expectEqual(@as(u32, 6), s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{ 4, 6 }).?.at);
}

test "eviction frees a conversation's superseded state first, then the oldest; a state past the budget is refused" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 330); // fake bytes: 100 + at
    defer s.deinit();
    const other = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 3, &.{}, &.{}, null)); // 103 bytes
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null)); // 104: 207 held
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 6, &.{}, &.{}, null)); // 106 more: 313
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    const t3 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    _ = f.pass(&s, &t3, try s.begin(a, &t3, 8, &.{}, &.{}, null)); // 108: t1's state (extended by later turns) goes first
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.find(&.{ 5, 5, 5, 5 }, &.{}) != null);
    try std.testing.expectEqual(@as(u32, 8), s.find(&t3, &.{}).?.at);
    const big: [300]u32 = @splat(7);
    _ = f.pass(&s, &big, try s.begin(a, &big, 299, &.{}, &.{}, null)); // 399 > 330: refused, nothing evicted
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a failed copy keeps nothing and a failed restore prefills from the start" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa, .fail_save = true };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    f.fail_save = false;
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    f.fail_restore = true;
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const p = try s.begin(a, &t2, 6, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len); // the entry that failed is gone
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
}

test "marks: the stable prefix with the last prompt and shared blocks, past the resume point, before the end" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_gap = 2 }, 1 << 20);
    defer s.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const prev = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0, 0 };
    try std.testing.expectEqualSlices(u32, &.{ 3, 7, 10 }, try s.marks(a, &prompt, 0, 10, &.{ 3, 12, 0 }, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 7, 10, &.{3}, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &prompt, 0, 0, &.{}, &.{}, &.{})); // a raw prompt keeps nothing
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 0, 10, &.{9}, &.{}, &.{})); // a block next to the history
}
