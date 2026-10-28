//! Liquid's kept states in pages allocated once inside the prompt-cache budget (as GLM's): keeps allocate nothing.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const rows_page = @import("cuda_lanes.zig").page;

/// A snapshot page: the unit the budget is cut into.
pub const page: usize = 2 << 20;

/// The budget's pages, allocated once; states lease them and give them back.
pub const Pages = struct {
    pages: kv.Pool,
    arena: ?cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64) !Pages {
        const count = std.math.cast(u32, budget / page) orelse return error.SnapshotBudget;
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        return .{ .pages = pages, .arena = if (count > 0) try cuda.DeviceBuffer.alloc(d, @as(usize, count) * page) else null };
    }

    pub fn deinit(p: *Pages) void {
        if (p.arena) |*a| a.free();
        p.pages.deinit();
    }

    /// Bytes of pages no state holds.
    pub fn spare(p: *const Pages) u64 {
        return p.pages.available() * page;
    }

    fn take(p: *Pages, n: usize) !kv.Lease {
        if (n == 0 or n > p.pages.available() or p.arena == null) return error.NoSnapshotRoom;
        var lease: kv.Lease = .{};
        errdefer p.pages.finish(&lease);
        try p.pages.promise(&lease, n);
        try p.pages.cover(&lease, n, null);
        return lease;
    }
};

/// Region bytes [off, off + len) over a state's pages: (arena offset, region offset, length), a run of consecutive ids.
fn pageRuns(ids: []const u32, off: usize, len: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), held: u64, done: usize, n: usize) anyerror!void) !void {
    var done: usize = 0;
    while (done < len) {
        const at = off + done;
        var q = at / page;
        var n = @min(len - done, page - at % page);
        while (done + n < len and ids[q + 1] == ids[q] + 1) : (q += 1) n = @min(len - done, n + page);
        try f(ctx, @as(u64, ids[at / page]) * page + at % page, done, n);
        done += n;
    }
}

/// Copies between a slot's region at `dev` and a lease's pages.
const Copy = struct { d: *const cuda.Driver, s: cuda.Stream, arena: u64, dev: u64, out: bool };

fn copyRun(c: Copy, held: u64, done: usize, n: usize) anyerror!void {
    const src = if (c.out) c.dev + done else c.arena + held;
    const dst = if (c.out) c.arena + held else c.dev + done;
    try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, c.s.handle), "cuMemcpyDtoDAsync");
}

const Leased = struct { d: *const cuda.Driver, s: cuda.Stream, arena: u64, ids: []const u32, out: bool };

fn copy(c: Leased, dev: u64, off: usize, len: usize) anyerror!void {
    try pageRuns(c.ids, off, len, Copy{ .d = c.d, .s = c.s, .arena = c.arena, .dev = dev, .out = c.out }, copyRun);
}

/// `mdl` names `page_elems`, `kv_width`, `paged_layers`, a Model (`fixed_bytes`, `fixed(slot, i)`, arena, ids, caps).
pub fn States(comptime mdl: type) type {
    return struct {
        /// A kept state: its position and the snapshot pages that hold it, in order.
        pub const Saved = struct { at: u32, lease: kv.Lease };

        const page_bytes = mdl.page_elems * 2;
        const row_bytes = mdl.kv_width * 2;

        /// Bytes a state at `at` holds: whole snapshot pages.
        pub fn bytes(at: usize) u64 {
            const n = mdl.Model.fixed_bytes + at * (page_bytes / rows_page);
            return (n + page - 1) / page * page;
        }

        /// `slot`'s state at `at`: fixed regions (the first from `tap` if set), a copy a run of whole pages, then rows.
        pub fn each(m: *const mdl.Model, slot: usize, at: usize, tap: ?u64, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
            if (at > m.caps[slot]) return error.PastPages;
            var off: usize = 0;
            var i: usize = 0;
            while (m.fixed(slot, i)) |r| : (i += 1) { // a ring holds positions, not rows: whole
                try f(ctx, if (i == 0) tap orelse r[0] else r[0], off, @intCast(r[1]));
                off += @intCast(r[1]);
            }
            const ids = m.ids[slot * m.perSlot() ..];
            const whole = at / rows_page;
            var q: usize = 0;
            while (q < whole) {
                var n: usize = 1;
                while (q + n < whole and ids[q + n] == ids[q] + n) n += 1;
                try f(ctx, m.arena + @as(u64, ids[q]) * page_bytes, off, n * page_bytes);
                off += n * page_bytes;
                q += n;
            }
            const rows = at % rows_page;
            if (rows == 0) return;
            const last = m.arena + @as(u64, ids[whole]) * page_bytes;
            for (0..2 * mdl.paged_layers) |l| { // layer a's keys at block 2a, its values at 2a + 1
                try f(ctx, last + l * rows_page * row_bytes, off, rows * row_bytes);
                off += rows * row_bytes;
            }
        }

        /// `slot`'s state at `at` copied into leased pages, queued behind the chunk that ends there or taps it (`tap`).
        pub fn save(gpa: std.mem.Allocator, p: *Pages, m: *const mdl.Model, s: cuda.Stream, slot: usize, at: u32, tap: ?u64) !*Saved {
            if (at == 0 or at > m.context) return error.NotAtMark;
            const k = try gpa.create(Saved);
            errdefer gpa.destroy(k);
            k.* = .{ .at = at, .lease = try p.take(bytes(at) / page) };
            errdefer p.pages.finish(&k.lease);
            try each(m, slot, at, tap, Leased{ .d = m.d, .s = s, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = true }, copy);
            return k;
        }

        /// Make `slot`'s pages and fixed regions hold `k`'s state, queued on `s` before the next chunk.
        pub fn restore(p: *const Pages, m: *const mdl.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
            if (k.at > m.context) return error.PastContext;
            try each(m, slot, k.at, null, Leased{ .d = m.d, .s = s, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = false }, copy);
        }

        /// Give a state's pages back: every copy of them is queued on the one stream, ahead of any later one.
        pub fn drop(gpa: std.mem.Allocator, p: *Pages, k: *Saved) void {
            p.pages.finish(&k.lease);
            gpa.destroy(k);
        }
    };
}

const testing = std.testing;

const Run = struct {
    out: *std.ArrayList([3]usize),
    fn add(r: Run, held: u64, done: usize, n: usize) anyerror!void {
        try r.out.append(testing.allocator, .{ held, done, n });
    }
};

test "a region over snapshot pages: one copy a run of consecutive ids, split where the ids jump" {
    const ids = [_]u32{ 4, 5, 9, 10, 11, 2 };
    var out: std.ArrayList([3]usize) = .empty;
    defer out.deinit(testing.allocator);
    // bytes [page / 2, 5 page + 10): pages 4-5 (from mid-page), 9-11, then 2
    try pageRuns(&ids, page / 2, 4 * page + page / 2 + 10, Run{ .out = &out }, Run.add);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqual([3]usize{ 4 * page + page / 2, 0, page + page / 2 }, out.items[0]);
    try testing.expectEqual([3]usize{ 9 * page, page + page / 2, 3 * page }, out.items[1]);
    try testing.expectEqual([3]usize{ 2 * page, 4 * page + page / 2, 10 }, out.items[2]);
    out.clearRetainingCapacity();
    try pageRuns(&ids, 3 * page + 7, 100, Run{ .out = &out }, Run.add); // inside one page
    try testing.expectEqual([3]usize{ 10 * page + 7, 0, 100 }, out.items[0]);
}
