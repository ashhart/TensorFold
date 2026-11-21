//! TF_DSV41_POOL_TOKENS on Metal: the seats share one pool of positions, each request a contiguous run of blocks.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const name = "TF_DSV41_POOL_TOKENS";

/// Positions a block holds: at every compress ratio a block's rows start 256-byte aligned in each pooled buffer.
pub const block: u32 = 256;

/// The largest compress ratio: a run gives up these positions so its last compressed row stays inside it.
pub const slack: u32 = 2;

/// A seat's blocks [start, start + blocks).
pub const Run = struct {
    start: u32,
    blocks: u32,

    pub fn end(r: Run) u32 {
        return r.start + r.blocks;
    }
};

/// A live run moved down to `to` (blocks [from, from + blocks) copied whole, lowest run first).
pub const Move = struct { seat: u32, from: u32, to: u32, blocks: u32 };

/// A placement: the run, the moves to apply before it is used, and its blocks to zero (in block order).
pub const Plan = struct { run: Run, moves: []const Move, zero: []const Run };

/// The positions a run of `blocks` blocks holds.
pub fn tokensOf(blocks: u32) u64 {
    return if (blocks == 0) 0 else @as(u64, blocks) * block - slack;
}

/// The blocks a run of `tokens` positions takes.
pub fn blocksFor(tokens: u64) u64 {
    return (tokens + slack + block - 1) / block;
}

pub const Pool = struct {
    a: Allocator,
    blocks: u32,
    /// each seat's run (null: the seat holds no positions)
    runs: []?Run,
    /// a block that may hold bytes other than zeros: a new run zeroes it before its request reads it
    dirty: []bool,
    moves: []Move,
    zero: []Run,
    order: []u32,

    /// `seats` seats over `blocks` blocks, every block zeroed (a new device buffer's state).
    pub fn init(a: Allocator, seats: u32, blocks: u32) !Pool {
        if (seats == 0 or blocks == 0) return error.EmptyPool;
        const runs = try a.alloc(?Run, seats);
        errdefer a.free(runs);
        @memset(runs, null);
        const dirty = try a.alloc(bool, blocks);
        errdefer a.free(dirty);
        @memset(dirty, false);
        const moves = try a.alloc(Move, seats);
        errdefer a.free(moves);
        const zero = try a.alloc(Run, blocks / 2 + 1);
        errdefer a.free(zero);
        const order = try a.alloc(u32, seats);
        return .{ .a = a, .blocks = blocks, .runs = runs, .dirty = dirty, .moves = moves, .zero = zero, .order = order };
    }

    pub fn deinit(p: *Pool) void {
        p.a.free(p.order);
        p.a.free(p.zero);
        p.a.free(p.moves);
        p.a.free(p.dirty);
        p.a.free(p.runs);
        p.* = undefined;
    }

    /// The most positions one run holds: the whole pool's.
    pub fn capacity(p: *const Pool) u64 {
        return tokensOf(p.blocks);
    }

    /// Blocks no seat holds.
    pub fn free(p: *const Pool) u32 {
        var used: u32 = 0;
        for (p.runs) |r| if (r) |x| {
            used += x.blocks;
        };
        return p.blocks - used;
    }

    /// Positions the seats' runs hold now.
    pub fn held(p: *const Pool) u64 {
        var n: u64 = 0;
        for (p.runs) |r| if (r) |x| {
            n += tokensOf(x.blocks);
        };
        return n;
    }

    /// Seat `seat`'s run of `tokens` positions: first fit, else the live runs slide down; LaneMemory: not now.
    pub fn take(p: *Pool, seat: u32, tokens: u64) error{ PastPool, SeatHeld, LaneMemory }!Plan {
        if (seat >= p.runs.len or p.runs[seat] != null) return error.SeatHeld;
        if (tokens == 0 or tokens > p.capacity()) return error.PastPool;
        const n: u32 = @intCast(blocksFor(tokens));
        var moves: []const Move = &.{};
        const at = p.firstFit(n) orelse blk: {
            if (p.free() < n) return error.LaneMemory;
            moves = p.compact();
            break :blk p.blocks - p.free();
        };
        const r: Run = .{ .start = at, .blocks = n };
        p.runs[seat] = r;
        return .{ .run = r, .moves = moves, .zero = p.claim(r) };
    }

    /// Seat `seat`'s run goes back; the blocks its first `written` positions reach are marked dirty.
    pub fn give(p: *Pool, seat: u32, written: u64) void {
        if (seat >= p.runs.len) return;
        const r = p.runs[seat] orelse return;
        p.runs[seat] = null;
        if (written == 0) return;
        const reach: u32 = @intCast(@min(r.blocks, (written + slack) / block + 1));
        @memset(p.dirty[r.start..][0..reach], true);
    }

    /// The lowest start of `n` free blocks.
    fn firstFit(p: *const Pool, n: u32) ?u32 {
        var at: u32 = 0;
        while (at + n <= p.blocks) {
            var next: ?u32 = null; // the end of the last run overlapping [at, at + n)
            for (p.runs) |r| if (r) |x| if (x.start < at + n and at < x.end()) {
                next = @max(next orelse 0, x.end());
            };
            at = next orelse return at;
        }
        return null;
    }

    /// Every live run slid down in start order, so the free blocks are one run at the top; vacated blocks go dirty.
    fn compact(p: *Pool) []const Move {
        var live: usize = 0;
        for (p.runs, 0..) |r, s| if (r != null) {
            var i = live;
            while (i > 0 and p.runs[p.order[i - 1]].?.start > r.?.start) : (i -= 1) p.order[i] = p.order[i - 1];
            p.order[i] = @intCast(s);
            live += 1;
        };
        var moved: usize = 0;
        var cursor: u32 = 0;
        for (p.order[0..live]) |s| {
            const r = p.runs[s].?;
            if (r.start != cursor) {
                p.moves[moved] = .{ .seat = s, .from = r.start, .to = cursor, .blocks = r.blocks };
                moved += 1;
                const lo = @max(cursor + r.blocks, r.start);
                @memset(p.dirty[lo..r.end()], true);
                p.runs[s] = .{ .start = cursor, .blocks = r.blocks };
            }
            cursor += r.blocks;
        }
        return p.moves[0..moved];
    }

    /// The dirty blocks of a new run, coalesced, now counted clean (its caller zeroes them).
    fn claim(p: *Pool, r: Run) []const Run {
        var n: usize = 0;
        var b = r.start;
        while (b < r.end()) {
            if (!p.dirty[b]) {
                b += 1;
                continue;
            }
            const lo = b;
            while (b < r.end() and p.dirty[b]) : (b += 1) p.dirty[b] = false;
            p.zero[n] = .{ .start = lo, .blocks = b - lo };
            n += 1;
        }
        return p.zero[0..n];
    }
};

/// The setting as a load takes it: the pool's blocks and the most positions one request uses.
pub const Setting = struct {
    tokens: u64,
    blocks: u32,
    /// the per-request window: --context, cut to the pool when it is smaller
    window: u32,
};

/// The setting checked against `--parallel` seats of `context` positions; null: unset (a window a seat, as before).
pub fn setting(tokens: ?u64, seats: u32, context: u32, problem: *[]const u8, buf: []u8) error{BadPoolTokens}!?Setting {
    const t = tokens orelse return null;
    const all = @as(u64, seats) * context;
    const words: ?[]const u8 = if (t < block)
        std.fmt.bufPrint(buf, "{s}={d} is less than one block of {d} positions; give at least {d}, or unset it for a context window a seat", .{ name, t, block, block }) catch name
    else if (t > all)
        std.fmt.bufPrint(buf, "{s}={d} is more than --parallel {d} x --context {d} = {d} positions; every seat would hold its own window: lower it or unset it", .{ name, t, seats, context, all }) catch name
    else
        null;
    if (words) |w| {
        problem.* = w;
        return error.BadPoolTokens;
    }
    return .{ .tokens = t, .blocks = @intCast(blocksFor(t)), .window = @intCast(@min(context, t)) };
}

/// The startup line's seat words: "N slots of C positions" unset (as before), else the pool's size and bytes.
pub fn seatWords(buf: []u8, slots: u32, limit: u32, s: ?Setting, block_bytes: u64) []const u8 {
    const x = s orelse return std.fmt.bufPrint(buf, "{d} slots of {d} positions", .{ slots, limit }) catch "slots";
    const mib = @as(u64, x.blocks) * block_bytes >> 20;
    return std.fmt.bufPrint(buf, "{d} slots sharing a pool of {d} positions ({s}={d}; a request up to {d}; {d} bytes a position, {d} MiB)", .{ slots, tokensOf(x.blocks), name, x.tokens, limit, block_bytes / block, mib }) catch "slots";
}

/// The refusal when the pool and the slots' own state pass the device's working set; null when they fit.
pub fn tooLarge(buf: []u8, tokens: u64, need: u64, used: u64, limit: u64) ?[]const u8 {
    if (used + need <= limit) return null;
    const gib = struct {
        fn of(b: u64) f64 {
            return @as(f64, @floatFromInt(b)) / (1 << 30);
        }
    }.of;
    return std.fmt.bufPrint(buf, "{s}={d} needs {d:.2} GiB for the pool and the slots beside {d:.2} GiB already allocated, past this GPU's {d:.2} GiB working set; lower {s}", .{ name, tokens, gib(need), gib(used), gib(limit), name }) catch name;
}

test {
    _ = @import("pool_test.zig");
}
