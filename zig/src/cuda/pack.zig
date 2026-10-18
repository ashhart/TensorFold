//! A load's permanent buffers bumped back to back in one growing mapping (a memory.Hook), not 2 MiB-rounded each.

const std = @import("std");
const memory = @import("memory.zig");
const arena = @import("arena.zig");

pub const Pack = struct {
    gpa: std.mem.Allocator,
    be: arena.Backend,
    alignment: usize = 512,
    prev: ?memory.Hook = null,
    open: bool = false,
    mutex: std.atomic.Mutex = .unlocked,
    /// mapped segments (base, len): one unless the backend hands out a range not following the last
    segments: std.ArrayList([2]u64) = .empty,
    cursor: u64 = 0,
    limit: u64 = 0,
    stats: struct { buffers: u64 = 0, bytes: u64 = 0, mapped: u64 = 0, freed: u64 = 0, outside: u64 = 0, fell_back: u64 = 0 } = .{},
    /// a backend whose chunks do not follow each other (cuMemAlloc): segments of `seg_min`, larger buffers their own
    fallback: ?arena.Backend = null,
    contiguous: bool = true,
    seg_min: usize = 256 << 20,

    pub fn init(gpa: std.mem.Allocator, be: arena.Backend) Pack {
        return .{ .gpa = gpa, .be = be };
    }

    /// Forgets the segments; their memory stays mapped (the backend's owner releases it).
    pub fn deinit(p: *Pack) void {
        p.uninstall();
        p.segments.deinit(p.gpa);
    }

    /// The first backend's chunk failed: pack on the fallback from here (none: the buffer goes to cuMemAlloc).
    fn fallBack(p: *Pack) bool {
        const f = p.fallback orelse return false;
        p.be = f;
        p.fallback = null;
        p.contiguous = false;
        p.cursor = 0;
        p.limit = 0;
        p.stats.fell_back += 1;
        return true;
    }

    /// Under DeviceBuffer.alloc / free, over whatever hook was there (the pack must outlive every buffer it hands out).
    pub fn install(p: *Pack) void {
        if (memory.currentHook()) |h| if (h.ctx == @as(*anyopaque, @ptrCast(p))) return;
        p.prev = memory.currentHook();
        memory.setHook(.{ .ctx = p, .alloc = hookAlloc, .free = hookFree });
    }

    pub fn uninstall(p: *Pack) void {
        if (memory.currentHook()) |h| if (h.ctx == @as(*anyopaque, @ptrCast(p))) memory.setHook(p.prev);
    }

    pub fn begin(p: *Pack) void {
        p.open = true;
    }

    pub fn end(p: *Pack) void {
        p.open = false;
    }

    fn lock(p: *Pack) void {
        while (!p.mutex.tryLock()) std.Thread.yield() catch {};
    }

    /// An address for `len` bytes while the phase is open; else the previous hook's (null: cuMemAlloc).
    pub fn alloc(p: *Pack, len: usize) ?u64 {
        if (len == 0) return null;
        p.lock();
        defer p.mutex.unlock();
        if (!p.open) {
            p.stats.outside += 1;
            const h = p.prev orelse return null;
            return h.alloc(h.ctx, len);
        }
        if (!p.contiguous) return p.segmented(len);
        const at = std.mem.alignForward(u64, p.cursor, p.alignment);
        if (at + len > p.limit) {
            const grow = std.mem.alignForward(usize, @intCast(at + len - p.limit), p.be.granularity);
            if (p.limit != 0) if (p.be.chunk(p.be.ctx, grow)) |base| {
                p.stats.mapped += grow;
                if (base == p.limit) {
                    p.segments.items[p.segments.items.len - 1][1] += grow;
                    p.limit += grow;
                    return p.place(len);
                }
                // a range not following the last (another user of the backend mapped in between): give it back
                p.be.release(p.be.ctx, base, grow);
                p.stats.mapped -= grow;
            };
            // the first segment, or a new one: the buffer's own pages to start it
            const n = std.mem.alignForward(usize, len, p.be.granularity);
            p.segments.ensureUnusedCapacity(p.gpa, 1) catch return null;
            const base = p.be.chunk(p.be.ctx, n) orelse return if (p.fallBack()) p.segmented(len) else null;
            p.segments.appendAssumeCapacity(.{ base, n });
            p.stats.mapped += n;
            p.cursor = base;
            p.limit = base + n;
        }
        return p.place(len);
    }

    /// Without contiguous growth: a `seg_min`+ buffer its own chunk, the rest bumped in segments.
    fn segmented(p: *Pack, len: usize) ?u64 {
        if (len >= p.seg_min) {
            const n = std.mem.alignForward(usize, len, p.be.granularity);
            p.segments.ensureUnusedCapacity(p.gpa, 1) catch return null;
            const base = p.be.chunk(p.be.ctx, n) orelse return null;
            p.segments.appendAssumeCapacity(.{ base, n });
            p.stats.mapped += n;
            p.stats.buffers += 1;
            p.stats.bytes += len;
            return base;
        }
        if (std.mem.alignForward(u64, p.cursor, p.alignment) + len > p.limit) {
            p.segments.ensureUnusedCapacity(p.gpa, 1) catch return null;
            const base = p.be.chunk(p.be.ctx, p.seg_min) orelse return null;
            p.segments.appendAssumeCapacity(.{ base, p.seg_min });
            p.stats.mapped += p.seg_min;
            p.cursor = base;
            p.limit = base + p.seg_min;
        }
        return p.place(len);
    }

    fn place(p: *Pack, len: usize) u64 {
        const at = std.mem.alignForward(u64, p.cursor, p.alignment);
        p.cursor = at + len;
        p.stats.buffers += 1;
        p.stats.bytes += len;
        return at;
    }

    fn owns(p: *const Pack, x: u64) bool {
        for (p.segments.items) |s| if (x >= s[0] and x < s[0] + s[1]) return true;
        return false;
    }

    /// A packed address stays mapped until close (true: ours); any other goes to the previous hook.
    pub fn free(p: *Pack, x: u64) bool {
        p.lock();
        defer p.mutex.unlock();
        if (p.owns(x)) {
            p.stats.freed += 1;
            return true;
        }
        const h = p.prev orelse return false;
        return h.free(h.ctx, x);
    }

    /// Mapped bytes past the packed ones: the segments' tails and alignment (what packing still costs).
    pub fn slack(p: *const Pack) u64 {
        return p.stats.mapped - p.stats.bytes;
    }

    fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
        const p: *Pack = @ptrCast(@alignCast(ctx));
        return p.alloc(len);
    }

    fn hookFree(ctx: *anyopaque, x: u64) bool {
        const p: *Pack = @ptrCast(@alignCast(ctx));
        return p.free(x);
    }
};

// -- tests: a fake backend handing out contiguous ranges, and a previous hook that counts ----------------------------

const Fake = struct {
    next: u64 = 1 << 40,
    chunks: u32 = 0,
    gap: bool = false,

    fn backend(f: *Fake, g: usize) arena.Backend {
        return .{ .ctx = f, .granularity = g, .chunk = chunk, .release = release, .name = "fake" };
    }
    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.gap) f.next += 4096; // another mapping in between
        const at = f.next;
        f.next += len;
        f.chunks += 1;
        return at;
    }
    fn release(_: *anyopaque, _: u64, _: usize) void {}
};

/// A backend whose every chunk fails (a VMM that maps nothing would otherwise pack 0 buffers).
const Broken = struct {
    calls: u32 = 0,
    fn backend(b: *Broken) arena.Backend {
        return .{ .ctx = b, .granularity = 2 << 20, .chunk = chunk, .release = Fake.release, .name = "broken" };
    }
    fn chunk(ctx: *anyopaque, _: usize) ?u64 {
        const b: *Broken = @ptrCast(@alignCast(ctx));
        b.calls += 1;
        return null;
    }
};

const Prev = struct {
    allocs: u32 = 0,
    frees: u32 = 0,
    fn hook(p: *Prev) memory.Hook {
        return .{ .ctx = p, .alloc = a, .free = f };
    }
    fn a(ctx: *anyopaque, _: usize) ?u64 {
        const p: *Prev = @ptrCast(@alignCast(ctx));
        p.allocs += 1;
        return 0x7000_0000;
    }
    fn f(ctx: *anyopaque, _: u64) bool {
        const p: *Prev = @ptrCast(@alignCast(ctx));
        p.frees += 1;
        return true;
    }
};

test "pack: permanent buffers back to back, 512-aligned, under a granularity page of slack in all" {
    const t = std.testing;
    var fake: Fake = .{};
    var prev: Prev = .{};
    var p = Pack.init(t.allocator, fake.backend(2 << 20));
    defer p.deinit();
    p.prev = prev.hook();
    p.begin();
    // the weights' mix: many tensors of odd sizes, a few large ones
    var sizes: [600]usize = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&sizes, 0..) |*s, i| s.* = if (i % 50 == 0) 300 << 20 else prng.random().intRangeAtMost(usize, 1000, 9 << 20);
    var total: u64 = 0;
    var last: u64 = 0;
    for (sizes) |s| {
        const x = p.alloc(s).?;
        try t.expectEqual(@as(u64, 0), x % 512);
        try t.expect(x >= last); // in order, never overlapping the one before
        last = x + s;
        total += s;
    }
    try t.expectEqual(@as(usize, 1), p.segments.items.len);
    try t.expect(p.slack() < (2 << 20) + 600 * 512);
    // one cuMemAlloc each would round each to 2 MiB: far more than the pack's slack
    var rounded: u64 = 0;
    for (sizes) |s| rounded += std.mem.alignForward(u64, s, 2 << 20);
    try t.expect(rounded - total > 100 * p.slack());
    // a packed buffer's free is ours and a no-op; after the phase the previous hook serves and frees
    const kept = p.stats.mapped;
    try t.expect(p.free(last - 1));
    p.end();
    try t.expectEqual(@as(?u64, 0x7000_0000), p.alloc(4096));
    try t.expect(p.free(0x7000_0000));
    try t.expectEqual(@as(u32, 1), prev.allocs);
    try t.expectEqual(@as(u32, 1), prev.frees);
    try t.expectEqual(kept, p.stats.mapped);
}

test "pack: a range that does not follow the last starts a new segment" {
    const t = std.testing;
    var fake: Fake = .{};
    var p = Pack.init(t.allocator, fake.backend(2 << 20));
    defer p.deinit();
    p.begin();
    _ = p.alloc(3 << 20).?;
    fake.gap = true;
    const x = p.alloc(3 << 20).?;
    try t.expectEqual(@as(usize, 2), p.segments.items.len);
    try t.expect(p.owns(x) and !p.owns(1));
}

test "pack: when the first backend's chunks fail it packs on the fallback; a boot never packs 0 buffers" {
    const t = std.testing;
    var broken: Broken = .{};
    var plain: Fake = .{ .gap = true }; // cuMemAlloc's chunks: never following each other
    var p = Pack.init(t.allocator, broken.backend());
    defer p.deinit();
    p.fallback = plain.backend(2 << 20);
    p.seg_min = 64 << 20;
    p.begin();
    var sizes: [800]usize = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    // the weights' small tensors (scales, norms, tables) between a few large ones
    for (&sizes, 0..) |*s, i| s.* = if (i % 40 == 0) 300 << 20 else prng.random().intRangeAtMost(usize, 1000, 2 << 20);
    var total: u64 = 0;
    var spans: std.ArrayList([2]u64) = .empty;
    defer spans.deinit(t.allocator);
    for (sizes) |s| {
        const x = p.alloc(s) orelse return error.NotPacked;
        try t.expectEqual(@as(u64, 0), x % 512);
        try spans.append(t.allocator, .{ x, x + s });
        total += s;
    }
    try t.expectEqual(@as(u32, 1), broken.calls); // tried once, then the fallback for good
    try t.expectEqual(@as(u64, 1), p.stats.fell_back);
    try t.expectEqual(@as(u64, sizes.len), p.stats.buffers);
    // no two buffers overlap
    std.mem.sort([2]u64, spans.items, {}, struct {
        fn lt(_: void, a: [2]u64, b: [2]u64) bool {
            return a[0] < b[0];
        }
    }.lt);
    for (spans.items[1..], spans.items[0 .. spans.items.len - 1]) |b, a| try t.expect(b[0] >= a[1]);
    // slack: the segments' tails and the large buffers' own rounding, a small part of 2 MiB-rounding each buffer
    var rounded: u64 = 0;
    for (sizes) |s| rounded += std.mem.alignForward(u64, s, 2 << 20);
    try t.expect(p.slack() * 10 < rounded - total);
}

const OwnedChunks = struct {
    live: usize = 0,
    fn chunk(ctx: *anyopaque, _: usize) ?u64 {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.live += 1;
        return 0x100000;
    }
    fn release(ctx: *anyopaque, _: u64, _: usize) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.live -= 1;
    }
};
fn failedDescriptor(contiguous: bool, len: usize) !void {
    var f = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var o: OwnedChunks = .{};
    var p = Pack.init(f.allocator(), .{ .ctx = &o, .granularity = 4096, .chunk = OwnedChunks.chunk, .release = OwnedChunks.release, .name = "owned" });
    defer p.deinit();
    p.contiguous = contiguous;
    p.seg_min = 4096;
    p.begin();
    f.fail_index = f.alloc_index;
    try std.testing.expectEqual(@as(?u64, null), p.alloc(len));
    try std.testing.expectEqual(@as(usize, 0), o.live);
    try std.testing.expectEqual(@as(u64, 0), p.stats.mapped);
    f.fail_index = std.math.maxInt(usize);
    try std.testing.expect(p.alloc(len) != null);
    for (p.segments.items) |s| p.be.release(p.be.ctx, s[0], @intCast(s[1]));
    try std.testing.expectEqual(@as(usize, 0), o.live);
}
test "Ownership rollback: pack contiguous descriptor OOM keeps chunk ownership" {
    try failedDescriptor(true, 512);
}
test "Ownership rollback: pack large descriptor OOM keeps chunk ownership" {
    try failedDescriptor(false, 8192);
}
test "Ownership rollback: pack small descriptor OOM keeps chunk ownership" {
    try failedDescriptor(false, 512);
}
