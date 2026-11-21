//! Main's pool, traced past kv_pool_class_test's script: adopt, share, drop, runs, reclaims and a shuffled stack.
const std = @import("std");
const kv = @import("kv_pool.zig");
const t = std.testing.allocator;

/// The walk's store: pages it holds itself, released last first when a promise or hold asks it to reclaim.
const Store = struct {
    p: *kv.Pool,
    held: std.ArrayList(u32) = .empty,
    asked: u32 = 0, // reclaims the pool asked for
    notified: u32 = 0, // outermost changes the pool reported

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.asked += 1;
        while (s.p.available() < want) s.p.release(s.held.pop() orelse return false);
        return true;
    }

    fn notify(ctx: *anyopaque, _: *const kv.Pool) void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.notified += 1;
    }
};

const Run = struct { start: u32, k: usize };
const ops = 12;

/// A page the store holds and `l` does not map yet (a lease maps a page once).
fn pick(l: *const kv.Lease, held: []const u32, r: std.Random) ?u32 {
    if (held.len == 0) return null;
    const id = held[r.uintLessThan(usize, held.len)];
    return if (std.mem.indexOfScalar(u32, l.ids.items, id) == null) id else null;
}

/// 400 seeded steps on 32 pages (`shuffled`: as gemma4's shufflePages), traced step by step, then the free stack.
fn walk(a: std.mem.Allocator, seed: u64, shuffled: bool, out: *std.ArrayList(u32), did: *[ops + 1]u32) !void {
    var p = try kv.Pool.init(a, 32, 4096);
    defer p.deinit();
    var s: Store = .{ .p = &p };
    defer s.held.deinit(a);
    try p.joinReclaim(.{ .ctx = &s, .call = Store.reclaim });
    p.notify = .{ .ctx = &s, .call = Store.notify };
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    if (shuffled) r.shuffle(u32, p.free[0..p.nfree]);
    var ls: [4]kv.Lease = @splat(.{});
    var kept: [ls.len]usize = @splat(0); // a lease's adopted pages, its table's head: never trimmed (open's need)
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(a);
    for (0..400) |_| {
        const li = r.uintLessThan(usize, ls.len);
        const l = &ls[li];
        const op = r.uintLessThan(u32, ops);
        var got: u32 = 0; // 1 + an id or start, 1 for a success, 0 for nothing done
        switch (op) {
            0 => {
                got = 1;
                p.promise(l, 1 + r.uintLessThan(usize, 6)) catch {
                    got = 0;
                };
            },
            1 => if (l.promised > 0) {
                try p.cover(l, l.ids.items.len + 1 + r.uintLessThan(usize, l.promised), null);
                got = 1 + l.ids.getLast();
            },
            2 => if (l.ids.items.len == 0 and l.promised > 0) {
                try p.coverRun(l, 1 + r.uintLessThan(usize, l.promised), @as(usize, 1) << r.uintLessThan(u6, 3), null);
                got = 1 + l.ids.items[0];
            },
            3 => if (l.promised > 0) {
                const k = 1 + r.uintLessThan(usize, @min(l.promised, 4));
                if (p.claimRun(l, k)) |start| {
                    try runs.append(a, .{ .start = start, .k = k });
                    got = 1 + start;
                }
            },
            4 => if (runs.pop()) |run| {
                p.releaseRun(run.start, run.k);
                got = 1 + run.start;
            },
            5 => if (l.ids.items.len > kept[li]) {
                p.trim(l, kept[li] + r.uintLessThan(usize, l.ids.items.len - kept[li]));
                got = 1;
            },
            6 => if (l.ids.items.len > 0 or l.promised > 0) {
                p.finish(l);
                kept[li] = 0;
                got = 1;
            },
            7 => if (l.promised > 0) {
                p.unpromise(l);
                got = 1;
            },
            8 => {
                var ids: [3]u32 = undefined;
                const n = 1 + r.uintLessThan(usize, ids.len);
                if (p.hold(ids[0..n])) {
                    try s.held.appendSlice(a, ids[0..n]);
                    got = 1 + ids[0];
                }
            },
            9 => if (s.held.items.len > 0) {
                const id = s.held.orderedRemove(r.uintLessThan(usize, s.held.items.len));
                if (r.boolean()) p.release(id) else try p.drop(id);
                got = 1 + id;
            },
            10 => if (l.ids.items.len == 0) if (pick(l, s.held.items, r)) |id| { // a resume's retained page, first
                p.retain(id);
                try p.adopt(l, &.{id});
                kept[li] = 1;
                got = 1 + id;
            },
            else => if (l.ids.items.len > 0) if (pick(l, s.held.items, r)) |id| { // a keep's shared copy
                p.share(l, r.uintLessThan(usize, l.ids.items.len), id);
                got = 1 + id;
            },
        }
        did[op] += @intFromBool(got > 0);
        try out.appendSlice(a, &.{ op, got });
        for (&ls) |*x| {
            try out.appendSlice(a, &.{ @intCast(x.promised), @intCast(x.ids.items.len) });
            try out.appendSlice(a, x.ids.items);
        }
        try out.append(a, @intCast(s.held.items.len));
        try out.appendSlice(a, s.held.items);
        const c = p.census();
        try out.appendSlice(a, &.{ @intCast(c.free), @intCast(c.promised), @intCast(c.kept), @intCast(c.leased) });
        try out.appendSlice(a, &.{ @intCast(p.available()), @intCast(p.reclaimable()), s.asked, s.notified });
    }
    did[ops] += s.asked;
    for (&ls) |*x| p.finish(x);
    for (runs.items) |run| p.releaseRun(run.start, run.k);
    for (s.held.items) |id| p.release(id);
    try std.testing.expect(p.idle() and p.nfree == p.total());
    try out.appendSlice(a, p.free[0..p.nfree]);
}

test "kv pool trace: adopt, share, drop, runs, reclaims and a shuffled stack hand out main's ids in main's order" {
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(t);
    var did: [ops + 1]u32 = @splat(0); // each operation's effective steps, then the reclaims asked for
    for ([_]u64{ 1, 2, 3 }) |seed| for ([_]bool{ false, true }) |shuffled| {
        try walk(t, 0x2415_0000 + seed, shuffled, &out, &did);
    };
    for (did) |n| try std.testing.expect(n >= 8); // every operation took effect several times, reclaims too
    try std.testing.expectEqual(@as(usize, 90429), out.items.len); // main ddd2df87c's to f03e5aac1's kv_pool.zig
    try std.testing.expectEqual(@as(u64, 0x9289b00642a64c9), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out.items)));
}
