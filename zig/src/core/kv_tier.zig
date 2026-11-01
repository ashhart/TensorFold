//! A host tier for the page pool: logical pages whose canonical rows live in host RAM, cached by device frames.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Allocator = std.mem.Allocator;

/// No frame (for a page) or no page (for a frame).
pub const none: u32 = std.math.maxInt(u32);

/// What the ranks of one model agree on before they open a tier: the smallest of each.
pub const Shape = struct {
    host_pages: usize,
    frames: usize,

    pub fn agree(a: Shape, b: Shape) Shape {
        return .{ .host_pages = @min(a.host_pages, b.host_pages), .frames = @min(a.frames, b.frames) };
    }
};

/// A rank's shape, slots and digest (16-bit float pieces), told to the other ranks at open; a differing one is refused.
pub const Lockstep = struct {
    pub const len = 8;

    pub fn vector(t: *const Tier, slots: usize) [len]f32 {
        const d = t.digest();
        var v: [len]f32 = @splat(0);
        v[0] = @floatFromInt(t.pool.total());
        v[1] = @floatFromInt(t.count());
        v[2] = @floatFromInt(slots);
        for (0..4) |i| v[3 + i] = @floatFromInt((d >> @intCast(16 * i)) & 0xffff);
        return v;
    }

    pub fn agree(a: [len]f32, b: [len]f32) error{RanksDisagree}!void {
        if (!std.mem.eql(f32, &a, &b)) return error.RanksDisagree;
    }
};

/// A deliberate bug a test turns on to show that the check beside it fails without the rule: never set outside tests.
pub const Fault = enum { none, keep_generation, ignore_pins, ignore_dirty, unpinned_dirty, keep_failed_miss };

/// A page a call pinned: its frame, the frame's generation, and whether the family must copy the page in.
pub const Fill = struct { page: u32, frame: u32, gen: u32, miss: bool };

/// A page's frame as a device table saw it: current while the frame's generation is the same.
pub const Ticket = struct { frame: u32, gen: u32 };

/// A stream's reservation in both currencies: pages (the pool's lease) and the frames it may pin at once.
pub const Lease = struct {
    pages: kv.Lease = .{},
    frames: usize = 0,
    pins: std.ArrayList(u32) = .empty, // one entry per pin, a page id
    lent: std.ArrayList(u32) = .empty, // frames lent to the stream: mapped to no page, the family's to manage
};

pub const Census = struct { frames: usize, free: usize, resident: usize, pinned: usize, dirty: usize, promised: usize, lent: usize };

pub const Tier = struct {
    gpa: Allocator,
    pool: kv.Pool, // logical pages; the host store holds `pool.total()` of them
    host: []u8, // canonical rows, `pool.total() * page_bytes`, borrowed (pinned host memory in a server)
    frame_of: []u32, // per page: its frame or `none`
    dirty: []bool, // per page: the frame holds rows the host store lacks
    page_of: []u32, // per frame: its page or `none`
    gen: []u32, // per frame: changes whenever the frame stops holding the page it held
    stamp: []u64, // per frame: the LRU clock at its last use
    pinned: []u32, // per frame: calls reading it
    free_frames: []u32,
    lent_mark: []bool, // per frame: lent to a lease
    nfree: usize,
    promised: usize = 0, // frames promised, summed over leases
    nlent: usize = 0, // frames lent, summed over leases
    clock: u64 = 0,
    fault: Fault = .none,

    /// In place (the pool's release hook points here): `host.len / page_bytes` logical pages, `frames` frames.
    pub fn init(t: *Tier, gpa: Allocator, host: []u8, frames: usize, page_bytes: u64) !void {
        const npages: usize = @intCast(host.len / page_bytes);
        var pool = try kv.Pool.init(gpa, npages, page_bytes);
        errdefer pool.deinit();
        const frame_of = try gpa.alloc(u32, npages);
        errdefer gpa.free(frame_of);
        const dirty = try gpa.alloc(bool, npages);
        errdefer gpa.free(dirty);
        const page_of = try gpa.alloc(u32, frames);
        errdefer gpa.free(page_of);
        const gen = try gpa.alloc(u32, frames);
        errdefer gpa.free(gen);
        const stamp = try gpa.alloc(u64, frames);
        errdefer gpa.free(stamp);
        const pinned = try gpa.alloc(u32, frames);
        errdefer gpa.free(pinned);
        const free_frames = try gpa.alloc(u32, frames);
        errdefer gpa.free(free_frames);
        const lent_mark = try gpa.alloc(bool, frames);
        @memset(frame_of, none);
        @memset(dirty, false);
        @memset(page_of, none);
        @memset(gen, 0);
        @memset(stamp, 0);
        @memset(pinned, 0);
        @memset(lent_mark, false);
        for (free_frames, 0..) |*f, i| f.* = @intCast(frames - 1 - i); // frame 0 first
        t.* = .{ .gpa = gpa, .pool = pool, .host = host, .frame_of = frame_of, .dirty = dirty, .page_of = page_of, .gen = gen, .stamp = stamp, .pinned = pinned, .free_frames = free_frames, .lent_mark = lent_mark, .nfree = frames };
        t.pool.freed = .{ .ctx = t, .call = freed };
    }

    pub fn deinit(t: *Tier) void {
        t.pool.deinit();
        t.gpa.free(t.frame_of);
        t.gpa.free(t.dirty);
        t.gpa.free(t.page_of);
        t.gpa.free(t.gen);
        t.gpa.free(t.stamp);
        t.gpa.free(t.pinned);
        t.gpa.free(t.free_frames);
        t.gpa.free(t.lent_mark);
        t.* = undefined;
    }

    pub fn count(t: *const Tier) usize {
        return t.page_of.len;
    }

    pub fn shape(t: *const Tier) Shape {
        return .{ .host_pages = t.pool.total(), .frames = t.count() };
    }

    /// Promise `pages` pages and `n` frames to `l`, or neither: LaneMemory or FrameMemory, whichever does not fit.
    pub fn reserve(t: *Tier, l: *Lease, pages: usize, n: usize) error{ LaneMemory, FrameMemory }!void {
        if (t.promised + n > t.count()) return error.FrameMemory;
        try t.pool.promise(&l.pages, pages);
        t.promised += n;
        l.frames += n;
    }

    /// Whether `pages` and `n` frames could be promised now (the store's idle pages counted).
    pub fn fits(t: *const Tier, pages: usize, n: usize) bool {
        return t.promised + n <= t.count() and t.pool.fits(pages);
    }

    /// `l` covers `n` pages, as `Pool.cover`.
    pub fn cover(t: *Tier, l: *Lease, n: usize, table: ?kv.Table) !void {
        return t.pool.cover(&l.pages, n, table);
    }

    /// The frame of `id`, if it has one.
    pub fn resident(t: *const Tier, id: u32) ?Ticket {
        const f = t.frame_of[id];
        return if (f == none) null else .{ .frame = f, .gen = t.gen[f] };
    }

    /// Whether a table entry written from `tk` still names `id`'s rows.
    pub fn valid(t: *const Tier, id: u32, tk: Ticket) bool {
        return t.page_of[tk.frame] == id and t.gen[tk.frame] == tk.gen;
    }

    /// Pin `id` for a call: a frame if it has none (a miss: copy `host_page(id)` in); FrameMemory past the promise.
    pub fn pin(t: *Tier, l: *Lease, id: u32) error{ FrameMemory, OutOfMemory }!Fill {
        if (t.pool.refs[id] == 0) std.debug.panic("kv tier: page {d} pinned while free", .{id});
        if (l.pins.items.len + l.lent.items.len >= l.frames) return error.FrameMemory;
        try l.pins.ensureUnusedCapacity(t.gpa, 1);
        var miss = false;
        if (t.frame_of[id] == none) {
            const f = t.takeFrame() orelse return error.FrameMemory;
            t.page_of[f] = id;
            t.frame_of[id] = f;
            miss = true;
        }
        const f = t.frame_of[id];
        t.pinned[f] += 1;
        t.touch(id);
        l.pins.appendAssumeCapacity(id);
        return .{ .page = id, .frame = f, .gen = t.gen[f], .miss = miss };
    }

    /// Pin every page of `ids` into `out`, or none of them.
    pub fn fill(t: *Tier, l: *Lease, ids: []const u32, out: []Fill) !void {
        std.debug.assert(out.len >= ids.len);
        var done: usize = 0;
        errdefer {
            for (ids[0..done]) |id| t.unpin(l, id) catch {};
            for (out[0..done]) |o| t.undo(o); // a miss was never copied in: its frame must not stay mapped
        }
        for (ids, 0..) |id, i| {
            out[i] = try t.pin(l, id);
            done += 1;
        }
    }

    /// The frame a failed batch took for a miss goes back unmapped (generation moved on), unless something pins it.
    fn undo(t: *Tier, o: Fill) void {
        if (t.fault == .keep_failed_miss or !o.miss or t.frame_of[o.page] != o.frame or t.pinned[o.frame] != 0) return;
        t.unmap(o.frame);
        t.free_frames[t.nfree] = o.frame;
        t.nfree += 1;
    }

    /// One pin of `id` fewer. Dirty: the last pin of a page whose rows the host store lacks cannot go (sync it first).
    pub fn unpin(t: *Tier, l: *Lease, id: u32) error{ NotPinned, Dirty }!void {
        var i = l.pins.items.len;
        while (i > 0) : (i -= 1) if (l.pins.items[i - 1] == id) break;
        if (i == 0) return error.NotPinned;
        const f = t.frame_of[id];
        if (t.pinned[f] == 1 and t.dirty[id] and t.fault != .ignore_dirty) return error.Dirty;
        _ = l.pins.orderedRemove(i - 1);
        t.pinned[f] -= 1;
    }

    /// Lend `n` frames to `l` for its life: mapped to no page, never a victim, counted in its promise like pins.
    pub fn lend(t: *Tier, l: *Lease, n: usize) error{ FrameMemory, OutOfMemory }!void {
        if (l.pins.items.len + l.lent.items.len + n > l.frames) return error.FrameMemory;
        try l.lent.ensureUnusedCapacity(t.gpa, n);
        var taken: usize = 0;
        errdefer t.giveBackTo(l, l.lent.items.len - taken);
        while (taken < n) : (taken += 1) {
            const f = t.takeFrame() orelse return error.FrameMemory;
            t.gen[f] +%= 1;
            t.lent_mark[f] = true;
            l.lent.appendAssumeCapacity(f);
            t.nlent += 1;
        }
    }

    /// Lend `l` `n` consecutive free frames (a family's state run) for its life; the first frame, or FrameMemory.
    pub fn lendRun(t: *Tier, l: *Lease, n: usize) error{ FrameMemory, OutOfMemory }!u32 {
        if (n == 0 or l.pins.items.len + l.lent.items.len + n > l.frames) return error.FrameMemory;
        var start: usize = 0;
        var run: usize = 0;
        for (t.page_of, t.lent_mark, 0..) |p, m, f| {
            if (p == none and !m) run += 1 else run = 0;
            if (run == n) {
                start = f + 1 - n;
                break;
            }
        } else return error.FrameMemory;
        try l.lent.ensureUnusedCapacity(t.gpa, n);
        var kept: usize = 0;
        for (t.free_frames[0..t.nfree]) |f| if (f < start or f >= start + n) {
            t.free_frames[kept] = f;
            kept += 1;
        };
        t.nfree = kept;
        for (start..start + n) |f| {
            t.gen[f] +%= 1;
            t.lent_mark[f] = true;
            l.lent.appendAssumeCapacity(@intCast(f));
        }
        t.nlent += n;
        return @intCast(start);
    }

    /// The frames lent to `l` go back to the free frames, their generation moved on.
    pub fn giveBack(t: *Tier, l: *Lease) void {
        t.giveBackTo(l, 0);
    }

    fn giveBackTo(t: *Tier, l: *Lease, keep: usize) void {
        while (l.lent.items.len > keep) {
            const f = l.lent.pop().?;
            t.gen[f] +%= 1;
            t.lent_mark[f] = false;
            t.free_frames[t.nfree] = f;
            t.nfree += 1;
            t.nlent -= 1;
        }
    }

    /// The LRU clock moves on for `id` (a hit the device saw, or a call that read it).
    pub fn touch(t: *Tier, id: u32) void {
        const f = t.frame_of[id];
        if (f == none) return;
        t.clock += 1;
        t.stamp[f] = t.clock;
    }

    /// Rows were written into `id`'s frame, which `l` pins: the host store lacks them until `sync`.
    pub fn markDirty(t: *Tier, l: *const Lease, id: u32) error{NotPinned}!void {
        if (t.fault != .unpinned_dirty and std.mem.indexOfScalar(u32, l.pins.items, id) == null) return error.NotPinned;
        if (t.frame_of[id] == none) std.debug.panic("kv tier: page {d} written with no frame", .{id});
        t.dirty[id] = true;
    }

    /// The family copied `id`'s frame to `host_page(id)`: the frame may be dropped.
    pub fn sync(t: *Tier, id: u32) void {
        t.dirty[id] = false;
    }

    /// Where a page's canonical rows go (a flush writes here).
    pub fn host_page(t: *Tier, id: u32) []u8 {
        const b: usize = @intCast(t.pool.page_bytes);
        return t.host[id * b ..][0..b];
    }

    /// The canonical rows of `id` for a snapshot or a fill; Unsynced while its frame holds rows not flushed yet.
    pub fn canonical(t: *const Tier, id: u32) error{Unsynced}![]const u8 {
        if (t.dirty[id]) return error.Unsynced;
        const b: usize = @intCast(t.pool.page_bytes);
        return t.host[id * b ..][0..b];
    }

    /// `id`'s host rows were rewritten (a restored snapshot): its frame is dropped; Pinned while a call reads it.
    pub fn invalidate(t: *Tier, id: u32) error{Pinned}!void {
        const f = t.frame_of[id];
        if (f == none) return;
        if (t.pinned[f] > 0) return error.Pinned;
        t.dirty[id] = false; // the frame's rows are being replaced
        t.unmap(f);
        t.free_frames[t.nfree] = f;
        t.nfree += 1;
    }

    /// Drop up to `n` unpinned frames, least recently used first; the count dropped.
    pub fn evict(t: *Tier, n: usize) usize {
        var done: usize = 0;
        while (done < n) : (done += 1) {
            const f = t.victim() orelse break;
            t.unmap(f);
            t.free_frames[t.nfree] = f;
            t.nfree += 1;
        }
        return done;
    }

    /// Whether `l` can end: none of its pinned pages is left dirty and unpinned (a page only it holds goes with it).
    pub fn canFinish(t: *const Tier, l: *const Lease) bool {
        if (t.fault == .ignore_dirty) return true;
        for (l.pins.items) |id| {
            if (!t.dirty[id]) continue;
            const f = t.frame_of[id];
            var mine: usize = 0; // this lease's own pins of the page
            for (l.pins.items) |q| mine += @intFromBool(q == id);
            if (f != none and t.pinned[f] > mine) continue; // another stream still pins it
            const own = std.mem.indexOfScalar(u32, l.pages.ids.items, id) != null;
            if (!own or t.pool.refs[id] > 1) return false;
        }
        for (l.pages.ids.items) |id| if (t.dirty[id] and t.pool.refs[id] > 1) return false;
        return true;
    }

    /// The stream ended: its pins, frames, promise and pages go back (a page nobody else holds loses its frame).
    pub fn finish(t: *Tier, l: *Lease) void {
        if (t.fault == .none and !t.canFinish(l)) std.debug.panic("kv tier: a stream ended with unflushed rows pinned or shared", .{});
        for (l.pins.items) |id| {
            const f = t.frame_of[id];
            if (f != none) t.pinned[f] -= 1 else if (t.fault == .none) std.debug.panic("kv tier: pinned page {d} lost its frame", .{id});
        }
        l.pins.deinit(t.gpa);
        t.giveBack(l);
        l.lent.deinit(t.gpa);
        t.promised -= l.frames;
        t.pool.finish(&l.pages);
        l.* = .{};
    }

    pub fn census(t: *const Tier) Census {
        var c: Census = .{ .frames = t.count(), .free = t.nfree, .resident = 0, .pinned = 0, .dirty = 0, .promised = t.promised, .lent = t.nlent };
        for (t.page_of, t.pinned) |p, n| {
            c.resident += @intFromBool(p != none);
            c.pinned += @intFromBool(n > 0);
        }
        for (t.dirty) |d| c.dirty += @intFromBool(d);
        return c;
    }

    /// Nothing promised, pinned or dirty, and every frame free or holding a page still held (the store's).
    pub fn idle(t: *const Tier) bool {
        const c = t.census();
        return t.pool.idle() and c.promised == 0 and c.pinned == 0 and c.dirty == 0 and c.lent == 0 and c.free + c.resident == c.frames;
    }

    /// The mapping's own consistency; an error names the broken rule. Tests call it after every step.
    pub fn audit(t: *const Tier) error{Broken}!void {
        var mapped: usize = 0;
        for (t.page_of, 0..) |p, f| {
            if (p == none) {
                if (t.pinned[f] != 0) return error.Broken; // a pin on an empty frame
                continue;
            }
            mapped += 1;
            if (t.frame_of[p] != f) return error.Broken; // page and frame disagree
            if (t.pool.refs[p] == 0) return error.Broken; // a free page keeps a frame
        }
        if (mapped + t.nfree + t.nlent != t.count()) return error.Broken; // frames neither free, mapped nor lent, or two
        for (t.free_frames[0..t.nfree]) |f| if (t.page_of[f] != none or t.lent_mark[f]) return error.Broken;
        var marked: usize = 0;
        for (t.lent_mark, t.page_of) |m, p| {
            marked += @intFromBool(m);
            if (m and p != none) return error.Broken; // a lent frame holds a page
        }
        if (marked != t.nlent) return error.Broken;
        for (t.frame_of, 0..) |f, p| {
            if (f != none and t.page_of[f] != p) return error.Broken;
            if (t.dirty[p] and f == none) return error.Broken; // unflushed rows with no frame
        }
        if (t.promised > t.count()) return error.Broken;
    }

    /// A hash of what two ranks must hold alike: mapping, generations, LRU order, pins, holders and promises.
    pub fn digest(t: *const Tier) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.sliceAsBytes(t.frame_of));
        h.update(std.mem.sliceAsBytes(t.page_of));
        h.update(std.mem.sliceAsBytes(t.gen));
        h.update(std.mem.sliceAsBytes(t.stamp));
        h.update(std.mem.sliceAsBytes(t.pinned));
        h.update(std.mem.sliceAsBytes(t.free_frames[0..t.nfree]));
        h.update(std.mem.sliceAsBytes(t.pool.refs));
        h.update(std.mem.sliceAsBytes(t.pool.free[0..t.pool.nfree]));
        const counts = [_]u64{ t.promised, t.pool.promised, t.clock, t.nlent };
        h.update(std.mem.sliceAsBytes(&counts));
        return h.final();
    }

    /// A free frame, or the least recently used unpinned one dropped.
    fn takeFrame(t: *Tier) ?u32 {
        if (t.nfree > 0) {
            t.nfree -= 1;
            return t.free_frames[t.nfree];
        }
        const f = t.victim() orelse return null;
        t.unmap(f);
        return f;
    }

    /// The mapped frame with the oldest stamp, not pinned and with no unflushed rows (the lowest index among equals).
    fn victim(t: *const Tier) ?u32 {
        var best: ?u32 = null;
        for (t.page_of, 0..) |p, f| {
            if (p == none) continue;
            if (t.pinned[f] > 0 and t.fault != .ignore_pins) continue;
            if (t.dirty[p] and t.fault != .ignore_dirty) continue;
            if (best == null or t.stamp[f] < t.stamp[best.?]) best = @intCast(f);
        }
        return best;
    }

    /// The frame stops holding its page: the generation moves on so a table naming it is stale.
    fn unmap(t: *Tier, f: u32) void {
        const p = t.page_of[f];
        if (t.dirty[p] and t.fault != .ignore_dirty) std.debug.panic("kv tier: frame {d} dropped with page {d} unflushed", .{ f, p });
        t.frame_of[p] = none;
        t.dirty[p] = false;
        t.page_of[f] = none;
        if (t.fault != .keep_generation) t.gen[f] +%= 1;
    }

    /// The pool freed `id` (its last holder let go): its frame is free again.
    fn freed(ctx: *anyopaque, id: u32) void {
        const t: *Tier = @ptrCast(@alignCast(ctx));
        const f = t.frame_of[id];
        if (f == none) return;
        if (t.pinned[f] != 0 and t.fault == .none) std.debug.panic("kv tier: page {d} freed while pinned", .{id});
        t.dirty[id] = false; // the page is gone: rows nobody can read need no flush
        t.unmap(f);
        t.free_frames[t.nfree] = f;
        t.nfree += 1;
    }
};

test {
    _ = @import("kv_tier_test.zig");
    _ = @import("kv_tier_ranks_test.zig");
}
