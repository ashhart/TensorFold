//! The data plane a round uses: one exchange step (a chunk to every rank, theirs back), and the exact collectives built on it.
const std = @import("std");
const canon = @import("canon.zig");

pub const Error = error{ PeerTimeout, TooLarge, Corrupt, OutOfMemory };

/// fabric.collective.Channel implements this over MCDMA one-sided writes and flag words; Mem does in one process.
pub const Exchange = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        rank: *const fn (ptr: *anyopaque) u32,
        size: *const fn (ptr: *anyopaque) u32,
        /// chunks[p] goes to rank p; out[p] views what rank p sent here, valid until this rank's next exchange.
        exchange: *const fn (ptr: *anyopaque, chunks: []const []const u8, out: [][]const u8) Error!void,
    };

    pub fn rank(x: Exchange) u32 {
        return x.vtable.rank(x.ptr);
    }
    pub fn size(x: Exchange) u32 {
        return x.vtable.size(x.ptr);
    }
    pub fn exchange(x: Exchange, chunks: []const []const u8, out: [][]const u8) Error!void {
        return x.vtable.exchange(x.ptr, chunks, out);
    }
};

pub const max_ranks = 64;

/// In-process ranks on threads: each send copies into the receiver's slot for the step's parity, then sets a flag word.
pub const Mem = struct {
    gpa: std.mem.Allocator,
    n: u32,
    slots: []std.ArrayList(u8),
    flags: []std.atomic.Value(u64),
    ends: []End,
    timeout_ns: u64 = 20 * std.time.ns_per_s,

    pub fn init(gpa: std.mem.Allocator, n: u32) !*Mem {
        const m = try gpa.create(Mem);
        m.* = .{ .gpa = gpa, .n = n, .slots = try gpa.alloc(std.ArrayList(u8), 2 * n * n), .flags = try gpa.alloc(std.atomic.Value(u64), 2 * n * n), .ends = try gpa.alloc(End, n) };
        for (m.slots) |*s| s.* = .empty;
        for (m.flags) |*f| f.* = .init(0);
        for (m.ends, 0..) |*e, i| e.* = .{ .mem = m, .me = @intCast(i) };
        return m;
    }

    pub fn deinit(m: *Mem) void {
        for (m.slots) |*s| s.deinit(m.gpa);
        m.gpa.free(m.slots);
        m.gpa.free(m.flags);
        m.gpa.free(m.ends);
        m.gpa.destroy(m);
    }

    fn at(m: *const Mem, parity: u64, src: u32, dst: u32) usize {
        return (@as(usize, @intCast(parity)) * m.n + src) * m.n + dst;
    }

    pub fn endpoint(m: *Mem, r: u32) Exchange {
        return .{ .ptr = &m.ends[r], .vtable = &End.vtable };
    }
};

const End = struct {
    mem: *Mem,
    me: u32,
    step: u64 = 0,

    const vtable: Exchange.VTable = .{ .rank = rank, .size = size, .exchange = exchange };

    fn rank(ptr: *anyopaque) u32 {
        const e: *End = @ptrCast(@alignCast(ptr));
        return e.me;
    }

    fn size(ptr: *anyopaque) u32 {
        const e: *End = @ptrCast(@alignCast(ptr));
        return e.mem.n;
    }

    fn exchange(ptr: *anyopaque, chunks: []const []const u8, out: [][]const u8) Error!void {
        const e: *End = @ptrCast(@alignCast(ptr));
        const m = e.mem;
        e.step += 1;
        const parity = e.step & 1;
        for (0..m.n) |i| {
            const p: u32 = @intCast(i);
            if (p == e.me) continue;
            const slot = &m.slots[m.at(parity, e.me, p)];
            slot.clearRetainingCapacity();
            try slot.appendSlice(m.gpa, chunks[p]);
            m.flags[m.at(parity, e.me, p)].store(e.step, .release);
        }
        out[e.me] = chunks[e.me];
        const start = std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds;
        for (0..m.n) |i| {
            const p: u32 = @intCast(i);
            if (p == e.me) continue;
            var spins: u32 = 0;
            while (m.flags[m.at(parity, p, e.me)].load(.acquire) != e.step) {
                spins += 1;
                if (spins % 1024 == 0) {
                    std.Thread.yield() catch {};
                    if (std.Io.Clock.awake.now(std.Options.debug_io).nanoseconds - start > m.timeout_ns) return error.PeerTimeout;
                } else std.atomic.spinLoopHint();
            }
            out[p] = m.slots[m.at(parity, p, e.me)].items;
        }
    }
};

/// Every rank waits for every other.
pub fn barrier(x: Exchange) Error!void {
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    var out: [max_ranks][]const u8 = undefined;
    try x.exchange(chunks[0..x.size()], out[0..x.size()]);
}

/// The root's bytes on every rank (the leader's round descriptor); returns the view this rank holds.
pub fn broadcast(x: Exchange, root: u32, bytes: []const u8, out: [][]const u8) Error![]const u8 {
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    if (x.rank() == root) {
        for (chunks[0..x.size()]) |*c| c.* = bytes;
    }
    try x.exchange(chunks[0..x.size()], out[0..x.size()]);
    return out[root];
}

/// Every rank's chunk at the root; the others receive nothing.
pub fn gather(x: Exchange, root: u32, mine: []const u8, out: [][]const u8) Error!void {
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    chunks[root] = mine;
    try x.exchange(chunks[0..x.size()], out[0..x.size()]);
}

/// One message: this rank's covering intervals' partial sums, columns [cols.begin, cols.end) of every row.
fn encodeParts(a: std.mem.Allocator, mine: canon.Range, slices: []const []const f32, rows: usize, width: usize, cols: canon.Range) Error![]u8 {
    var cover: [32]canon.Range = undefined;
    const nc = canon.cover(mine, &cover);
    const floats = rows * cols.len();
    const msg = try a.alloc(u8, 4 + nc * (8 + floats * 4));
    std.mem.writeInt(u32, msg[0..4], @intCast(nc), .little);
    var at: usize = 4;
    for (cover[0..nc]) |c| {
        std.mem.writeInt(u32, msg[at..][0..4], c.begin, .little);
        std.mem.writeInt(u32, msg[at + 4 ..][0..4], c.end, .little);
        at += 8;
        const dst: []align(1) f32 = @ptrCast(msg[at..][0 .. floats * 4]);
        for (0..rows) |row| for (0..cols.len()) |k| {
            dst[row * cols.len() + k] = canon.subtree(slices, mine.begin, c.begin, c.end, row * width + cols.begin + k);
        };
        at += floats * 4;
    }
    return msg;
}

/// Every received part, copied into `arena` so it outlives the exchange's views.
fn decodeParts(arena: std.mem.Allocator, got: []const []const u8, floats: usize) Error![]canon.Part {
    var parts: std.ArrayList(canon.Part) = .empty;
    for (got) |g| {
        if (g.len < 4) return error.Corrupt;
        const n = std.mem.readInt(u32, g[0..4], .little);
        var p: usize = 4;
        for (0..n) |_| {
            if (p + 8 + floats * 4 > g.len) return error.Corrupt;
            const r: canon.Range = .{ .begin = std.mem.readInt(u32, g[p..][0..4], .little), .end = std.mem.readInt(u32, g[p + 4 ..][0..4], .little) };
            const data = try arena.alloc(f32, floats);
            @memcpy(std.mem.sliceAsBytes(data), g[p + 8 ..][0 .. floats * 4]);
            try parts.append(arena, .{ .range = r, .data = data });
            p += 8 + floats * 4;
        }
    }
    return parts.items;
}

/// The canonical sum over `units` slices of rows x width floats (slices[k] is slice mine.begin + k): same bits on every rank.
pub fn allReduce(a: std.mem.Allocator, x: Exchange, units: u32, mine: canon.Range, slices: []const []const f32, rows: usize, width: usize, out: []f32) Error!void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const msg = try encodeParts(arena.allocator(), mine, slices, rows, width, .{ .begin = 0, .end = @intCast(width) });
    var chunks: [max_ranks][]const u8 = undefined;
    for (chunks[0..x.size()]) |*ch| ch.* = msg;
    var got: [max_ranks][]const u8 = undefined;
    try x.exchange(chunks[0..x.size()], got[0..x.size()]);
    canon.reduce(units, try decodeParts(arena.allocator(), got[0..x.size()], rows * width), out[0 .. rows * width]);
}

/// Rank p's columns of the canonical sum, as `allReduce` would give them; then `allGather` makes the whole.
pub fn reduceScatter(a: std.mem.Allocator, x: Exchange, units: u32, mine: canon.Range, slices: []const []const f32, rows: usize, width: usize, out: []f32) Error!canon.Range {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const n = x.size();
    var cols: [max_ranks]canon.Range = undefined;
    columns(n, width, cols[0..n]);
    var chunks: [max_ranks][]const u8 = undefined;
    for (chunks[0..n], cols[0..n]) |*ch, c| ch.* = try encodeParts(arena.allocator(), mine, slices, rows, width, c);
    var got: [max_ranks][]const u8 = undefined;
    try x.exchange(chunks[0..n], got[0..n]);
    const own = cols[x.rank()];
    canon.reduce(units, try decodeParts(arena.allocator(), got[0..n], rows * own.len()), out[0 .. rows * own.len()]);
    return own;
}

/// Every rank's reduced columns (`block`, rows x cols) assembled into rows x width on every rank.
pub fn allGather(x: Exchange, block: []const f32, rows: usize, width: usize, out: []f32) Error!void {
    const n = x.size();
    var cols: [max_ranks]canon.Range = undefined;
    columns(n, width, cols[0..n]);
    var chunks: [max_ranks][]const u8 = undefined;
    for (chunks[0..n]) |*ch| ch.* = std.mem.sliceAsBytes(block);
    var got: [max_ranks][]const u8 = undefined;
    try x.exchange(chunks[0..n], got[0..n]);
    for (got[0..n], cols[0..n]) |g, c| {
        if (g.len != rows * c.len() * 4) return error.Corrupt;
        const v: []align(1) const f32 = @ptrCast(g);
        for (0..rows) |row| @memcpy(out[row * width + c.begin ..][0..c.len()], v[row * c.len() ..][0..c.len()]);
    }
}

/// Each rank's rows (`own`, in row order, `width` floats each) assembled by `owner[row]` into every rank's `out`.
pub fn gatherRows(x: Exchange, own: []const f32, owner: []const u32, width: usize, out: []f32) Error!void {
    const n = x.size();
    var chunks: [max_ranks][]const u8 = undefined;
    for (chunks[0..n]) |*ch| ch.* = std.mem.sliceAsBytes(own);
    var got: [max_ranks][]const u8 = undefined;
    try x.exchange(chunks[0..n], got[0..n]);
    var next: [max_ranks]usize = @splat(0);
    for (owner, 0..) |p, row| {
        const v: []align(1) const f32 = @ptrCast(got[p]);
        if ((next[p] + 1) * width * 4 > got[p].len) return error.Corrupt;
        @memcpy(out[row * width ..][0..width], v[next[p] * width ..][0..width]);
        next[p] += 1;
    }
}

/// Contiguous column blocks, one a rank, as even as the width allows.
fn columns(n: u32, width: usize, out: []canon.Range) void {
    var ones: [max_ranks]u64 = @splat(1);
    canon.partition(@intCast(width), ones[0..n], out);
}

test "four threads exchange chunks, broadcast, gather and reduce to the same bits as one rank alone" {
    const gpa = std.testing.allocator;
    const units = 8;
    const width = 5;
    const rows = 3;
    var data: [units][rows * width]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(5);
    for (&data) |*s| for (s) |*v| {
        v.* = (prng.random().float(f32) - 0.5) * std.math.pow(f32, 10, @floatFromInt(prng.random().intRangeAtMost(i32, -2, 7)));
    };
    var solo: [rows * width]f32 = undefined;
    {
        const m = try Mem.init(gpa, 1);
        defer m.deinit();
        var all: [units][]const f32 = undefined;
        for (&all, 0..) |*s, i| s.* = &data[i];
        try allReduce(gpa, m.endpoint(0), units, .{ .begin = 0, .end = units }, &all, rows, width, &solo);
    }
    const Worker = struct {
        fn run(x: Exchange, d: *const [units][rows * width]f32, out: *[rows * width]f32, root_saw: *u64) void {
            var ranges: [4]canon.Range = undefined;
            canon.partition(units, &.{ 1, 1, 2, 0 }, &ranges);
            const mine = ranges[x.rank()];
            var mine_slices: [units][]const f32 = undefined;
            for (mine.begin..mine.end, 0..) |s, j| mine_slices[j] = &d[s];
            allReduce(std.testing.allocator, x, units, mine, mine_slices[0..mine.len()], rows, width, out) catch unreachable;
            var block: [rows * width]f32 = undefined;
            const cols = reduceScatter(std.testing.allocator, x, units, mine, mine_slices[0..mine.len()], rows, width, &block) catch unreachable;
            var whole: [rows * width]f32 = undefined;
            allGather(x, block[0 .. rows * cols.len()], rows, width, &whole) catch unreachable;
            std.debug.assert(std.mem.eql(u32, @ptrCast(&whole), @ptrCast(out)));
            var views: [4][]const u8 = undefined;
            const b = broadcast(x, 0, "round-7", &views) catch unreachable;
            std.debug.assert(std.mem.eql(u8, b, "round-7"));
            var r: [1]u8 = .{@intCast(x.rank())};
            gather(x, 0, &r, &views) catch unreachable;
            if (x.rank() == 0) root_saw.* = views[1][0] + 10 * @as(u64, views[2][0]) + 100 * @as(u64, views[3][0]);
            barrier(x) catch unreachable;
        }
    };
    const m = try Mem.init(gpa, 4);
    defer m.deinit();
    var outs: [4][rows * width]f32 = undefined;
    var saw: u64 = 0;
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Worker.run, .{ m.endpoint(@intCast(i)), &data, &outs[i], &saw });
    for (threads) |t| t.join();
    for (outs) |o| try std.testing.expectEqualSlices(u32, @ptrCast(&solo), @ptrCast(&o));
    try std.testing.expectEqual(@as(u64, 321), saw);
}
