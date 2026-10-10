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
