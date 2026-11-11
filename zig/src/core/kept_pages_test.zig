//! Byte fixtures for immutable KV pages and the prompt store that owns their lifetimes.
const std = @import("std");
const kp = @import("kept_pages.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;

const Fake = struct {
    pool: kp.Store,
    data: [][4]u32,
    live: []const u32 = &.{},
    fn init(count: usize) !Fake {
        var pool = try kp.Store.init(a, count, 16, 4);
        errdefer pool.deinit();
        return .{ .pool = pool, .data = try a.alloc([4]u32, count) };
    }
    fn deinit(f: *Fake) void {
        f.pool.deinit();
        a.free(f.data);
    }
    const State = struct { image: kp.Image, private: [16]u8 = @splat(27) };
    fn of(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }
    fn state(p: pc.Saved) *State {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 16 + std.mem.alignForward(u64, at, 4) * 4;
    }
    fn added(p: *anyopaque, old: ?pc.Saved, at: u32) u64 {
        return 16 + of(p).pool.added(if (old) |x| state(x).image else null, at);
    }
    fn save(p: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        return saveFrom(p, owner, at, null);
    }
    fn saveFrom(p: *anyopaque, _: ?*anyopaque, at: u32, old: ?pc.Saved) !pc.Saved {
        const f = of(p);
        const st = try a.create(State);
        errdefer a.destroy(st);
        const image = try f.pool.keep(if (old) |x| state(x).image else null, at);
        for (image.ids[image.fresh..], image.fresh..) |id, i| {
            @memset(&f.data[id], 0xdead);
            const begin = i * 4;
            @memcpy(f.data[id][0..@min(4, at - begin)], f.live[begin..@min(begin + 4, at)]);
        }
        st.* = .{ .image = image };
        return @ptrCast(st);
    }
    fn restore(p: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const f = of(p);
        const st = state(saved);
        for (0..st.image.at) |i| try std.testing.expectEqual(f.live[i], f.data[st.image.ids[i / 4]][i % 4]);
        try std.testing.expectEqual(@as([16]u8, @splat(27)), st.private);
    }
    fn drop(p: *anyopaque, saved: pc.Saved) void {
        of(p).pool.drop(state(saved).image);
        a.destroy(state(saved));
    }
    fn charge(p: *anyopaque, saved: pc.Saved) u64 {
        return 16 + of(p).pool.exclusive(state(saved).image);
    }
    fn shared(p: *anyopaque) u64 {
        return of(p).pool.sharedBytes();
    }
    fn snapshots(f: *Fake) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .save_from = saveFrom, .incremental = added, .restore = restore, .drop = drop, .charged = charge, .shared_bytes = shared } };
    }
};

test "kept pages: full prefix retained, tail private, earlier eviction leaves exact rows and charge" {
    var f = try Fake.init(512);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    var s = pc.Store.init(a, f.snapshots(), .{}, 1024);
    defer s.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    f.live = &tokens;
    try std.testing.expect(s.keep(&tokens, 6, null, &.{}, &.{}));
    const first = Fake.state(s.entries.items[0].saved);
    const page = first.image.ids[0];
    try std.testing.expect(s.keep(&tokens, 10, null, &.{}, &.{}));
    const last = Fake.state(s.entries.items[1].saved);
    try std.testing.expectEqual(page, last.image.ids[0]);
    try std.testing.expect(first.image.ids[1] != last.image.ids[1]);
    try std.testing.expectEqual(@as(u32, 2), f.pool.pool.holders(page));
    try std.testing.expectEqual(@as(u64, 96), s.held);
    try std.testing.expectEqual(@as(u64, 48), s.entries.items[1].bytes);
    s.remove(0);
    try std.testing.expectEqual(@as(u32, 1), f.pool.pool.holders(page));
    try std.testing.expectEqual(@as(u64, 64), s.held);
    try std.testing.expectEqual(@as(u64, 64), s.entries.items[0].bytes);
    try Fake.restore(&f, null, s.entries.items[0].saved);
    s.remove(0);
    try std.testing.expectEqual(@as(u64, 0), s.held);
    try std.testing.expectEqual(@as(usize, 512), f.pool.pool.available());
}

test "kept pages: cyclic sixteen conversations keep one KV prefix plus private state per mark" {
    var f = try Fake.init(66_048);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    const budget = 16 * (16_400 * 4 + 3 * 16);
    var s = pc.Store.init(a, f.snapshots(), .{}, budget);
    defer s.deinit();
    const tokens = try a.alloc([16_512]u32, 16);
    defer a.free(tokens);
    for (tokens, 0..) |*t, i| for (t, 0..) |*x, j| {
        x.* = @intCast(100_000 * i + j);
    };
    for (0..16) |i| {
        f.live = &tokens[i];
        for ([_]u32{ 16_380, 16_384, 16_388 }) |at| try std.testing.expect(s.keep(&tokens[i], at, null, &.{}, &.{}));
    }
    try std.testing.expectEqual(@as(usize, 48), s.entries.items.len);
    for (0..3) |round| for (0..16) |i| {
        f.live = &tokens[i];
        const at: u32 = @intCast(16_388 + round * 4);
        const plan = try s.begin(a, tokens[i][0 .. at + 4], 0, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try std.testing.expectEqual(at, plan.from);
        try std.testing.expect(s.keep(tokens[i][0 .. at + 4], at + 4, null, &.{}, &.{}));
        try std.testing.expect(s.held <= budget);
    };
    try std.testing.expectEqual(@as(u64, 48), s.counts.hits);
    try std.testing.expectEqual(@as(u64, 0), s.counts.misses);
}

test "kept pages: incompatible token, decoded mode or expired parent is never shared" {
    var f = try Fake.init(512);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    var s = pc.Store.init(a, f.snapshots(), .{}, 1024);
    defer s.deinit();
    const tokens = @as([12]u32, @splat(1));
    f.live = &tokens;
    try std.testing.expect(s.keep(&tokens, 8, null, &.{}, &.{}));
    const share = @import("prompt_cache_share.zig");
    try std.testing.expect(share.parent(&s, &tokens, 10, &.{}) != null);
    var changed = tokens;
    changed[2] = 2;
    try std.testing.expect(share.parent(&s, &changed, 10, &.{}) == null);
    try std.testing.expect(share.parent(&s, &tokens, 10, &.{.{ 2, 4 }}) == null);
    s.image_frame = 9;
    try std.testing.expect(share.parent(&s, &tokens, 10, &.{}) == null);
    s.image_frame = 0;
    s.entries.items[0].expired = true;
    try std.testing.expect(share.parent(&s, &tokens, 10, &.{}) == null);
}

test "kept pages: allocation failures release promises and every retained page" {
    try std.testing.checkAllAllocationFailures(a, allocationFailure, .{});
}
fn allocationFailure(gpa: std.mem.Allocator) !void {
    var s = try kp.Store.init(gpa, 8, 16, 4);
    defer s.deinit();
    const first = try s.keep(null, 6);
    defer s.drop(first);
    const next = try s.keep(first, 12);
    defer s.drop(next);
}

const Mapping = struct {
    bytes: [2048]u8 = @splat(0),
    private: [128]u8 = @splat(0),
    pages: [8][128]u8 = @splat(@splat(0)),
    save: bool,
    fn emit(m: *Mapping, span: kp.Layout.Span) !void {
        const storage = if (span.page) |i| m.pages[i][span.offset..][0..span.len] else m.private[span.offset..][0..span.len];
        if (m.save) @memcpy(storage, m.bytes[span.output..][0..span.len]) else @memcpy(m.bytes[span.output..][0..span.len], storage);
    }
};
test "kept pages: legacy byte image round trips every head, partial page, fixed state, taps and disk range" {
    const l = kp.Layout{ .fixed = 16, .groups = 4, .row = 4, .tokens = 8 };
    for ([_]u32{ 1, 7, 8, 9, 16, 19 }) |at| {
        const n = 16 + at * 16 + 32;
        var m = Mapping{ .save = true };
        for (m.bytes[0..n], 0..) |*b, i| b.* = @truncate(i * 7 + 13);
        const expected = m.bytes;
        try l.walk(at, 0, n, &m, Mapping.emit);
        @memset(&m.bytes, 0);
        m.save = false;
        try l.walk(at, 0, n, &m, Mapping.emit);
        try std.testing.expectEqualSlices(u8, expected[0..n], m.bytes[0..n]);
        for (0..n) |off| {
            @memset(&m.bytes, 0);
            const len = @min(n - off, 23);
            try l.walk(at, off, len, &m, Mapping.emit);
            try std.testing.expectEqualSlices(u8, expected[off..][0..len], m.bytes[0..len]);
        }
    }
}

test "kept pages: losing the only parent cannot exceed the store budget" {
    var f = try Fake.init(512);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    var s = pc.Store.init(a, f.snapshots(), .{}, 63);
    defer s.deinit();
    const tokens: [16]u32 = @splat(7);
    f.live = &tokens;
    try std.testing.expect(s.keep(&tokens, 8, null, &.{}, &.{}));
    try std.testing.expect(!s.keep(&tokens, 12, null, &.{}, &.{}));
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 8), s.entries.items[0].at);
    try Fake.restore(&f, null, s.entries.items[0].saved);
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
}

test "kept pages: another pass's promise cannot be spent by a shared keep" {
    var f = try Fake.init(512);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    var s = pc.Store.init(a, f.snapshots(), .{}, 128);
    defer s.deinit();
    const tokens: [16]u32 = @splat(7);
    f.live = &tokens;
    try std.testing.expect(s.keep(&tokens, 8, null, &.{}, &.{}));
    const G = struct {
        fn needed(_: *anyopaque, _: *const pc.Entry) bool {
            return false;
        }
    };
    s.reservation_guard = .{ .ctx = &f, .needed = G.needed, .promised = .{ .bytes = 80, .slots = 1 } };
    try std.testing.expect(!s.keep(&tokens, 12, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 8), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u64, 48), s.held);
    try Fake.restore(&f, null, s.entries.items[0].saved);
}

test "kept pages: a guarded keep refuses before losing a parent that cannot be replaced" {
    var f = try Fake.init(512);
    defer f.deinit();
    f.pool.bind(@import("prompt_radix.zig").pagesOf(@TypeOf(f.pool.pool), &f.pool.pool, 16));
    var s = pc.Store.init(a, f.snapshots(), .{}, 155);
    defer s.deinit();
    const parent: [32]u32 = @splat(1);
    const queued: [32]u32 = @splat(2);
    f.live = &parent;
    try std.testing.expect(s.keep(&parent, 8, null, &.{}, &.{}));
    f.live = &queued;
    try std.testing.expect(s.keep(&queued, 20, null, &.{}, &.{}));
    const G = struct {
        fn needed(_: *anyopaque, e: *const pc.Entry) bool {
            return e.tokens[0] == 2;
        }
    };
    s.reservation_guard = .{ .ctx = &f, .needed = G.needed, .exact = true };
    f.live = &parent;
    try std.testing.expect(!s.keep(&parent, 12, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 144), s.held);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    try Fake.restore(&f, null, s.entries.items[0].saved);
    f.live = &queued;
    try Fake.restore(&f, null, s.entries.items[1].saved);
}
