//! Recall's record in RAM: the ids a server read and wrote, segments ended by `mark`, found by their 8-token keys.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Ends a segment: no key spans it and no copy crosses it.
pub const mark: u32 = std.math.maxInt(u32);
/// Tokens a key covers: the shortest match the record finds.
pub const key_len: u32 = 8;
/// Ids a chunk holds (one chunk file's).
pub const chunk_tokens: u32 = 1 << 20;
/// Candidates a search checks at most, newest first: a lookup's time is bounded by it.
pub const max_steps: u32 = 64;
/// Tokens a match is extended back at most.
pub const max_match: u32 = 64;
/// Positions a record numbers before it stops taking ids (a reload numbers them from 0 again).
const limit: u64 = std.math.maxInt(u32) - 4 * @as(u64, chunk_tokens);

const Chunk = struct {
    ids: []u32,
    back: []u32, // each position's distance back to the previous position in its key's bucket (0: none)
};

/// A context's longest match: its length, and the position of the id that followed it.
pub const Match = struct { len: u32 = 0, at: u32 = 0 };

/// A continuation found for a context, and its match's length.
pub const Found = struct { at: u32, len: u32 };

/// A run of ids the record holds: its length and its first position.
pub const Run = struct { len: usize = 0, at: u32 = 0 };

pub const Record = struct {
    gpa: Allocator,
    chunks: std.ArrayList(Chunk) = .empty,
    first: u32 = 0, // the oldest held chunk's number: positions start at first * chunk_tokens
    end: u32 = 0, // one past the newest position
    heads: []u32, // each bucket's newest position plus one (0: none)
    last: [key_len]u32 = @splat(mark), // the newest ids, the next key's
    run: u32 = 0, // ids since the last mark

    /// An empty record whose key table has 2^bucket_bits buckets (fixed: growth never rebuilds the index).
    pub fn init(gpa: Allocator, bucket_bits: u5) !Record {
        const heads = try gpa.alloc(u32, @as(usize, 1) << bucket_bits);
        @memset(heads, 0);
        return .{ .gpa = gpa, .heads = heads };
    }

    pub fn deinit(r: *Record) void {
        for (r.chunks.items) |c| r.free(c);
        r.chunks.deinit(r.gpa);
        r.gpa.free(r.heads);
    }

    fn free(r: *Record, c: Chunk) void {
        r.gpa.free(c.ids);
        r.gpa.free(c.back);
    }

    /// The oldest held position.
    pub fn base(r: *const Record) u32 {
        return r.first * chunk_tokens;
    }

    /// Ids held, marks included.
    pub fn held(r: *const Record) u32 {
        return r.end - r.base();
    }

    /// Bytes the record takes in RAM: its chunks' ids and links, and the key table.
    pub fn ramBytes(r: *const Record) u64 {
        return @as(u64, r.chunks.items.len) * chunk_tokens * 8 + @as(u64, r.heads.len) * 4;
    }

    pub fn id(r: *const Record, p: u32) u32 {
        return r.chunks.items[p / chunk_tokens - r.first].ids[p % chunk_tokens];
    }

    fn backOf(r: *const Record, p: u32) u32 {
        return r.chunks.items[p / chunk_tokens - r.first].back[p % chunk_tokens];
    }

    fn bucket(r: *const Record, key: *const [key_len]u32) usize {
        return @as(usize, @truncate(std.hash.Wyhash.hash(0x5eca11, std.mem.asBytes(key)))) & (r.heads.len - 1);
    }

    /// Whether `n` more ids fit before the positions run out.
    pub fn room(r: *const Record, n: usize) bool {
        return @as(u64, r.end) + n + 1 <= limit;
    }

    /// `ids` then a mark, each new position indexed; the caller checks `room` first.
    pub fn add(r: *Record, ids: []const u32) !void {
        for (ids) |t| try r.push(t);
        try r.push(mark);
    }

    /// One id at the end; a full newest chunk takes a new one first.
    pub fn push(r: *Record, t: u32) !void {
        const p = r.end;
        if (@as(usize, p) == @as(usize, r.base()) + r.chunks.items.len * chunk_tokens) {
            const ids = try r.gpa.alloc(u32, chunk_tokens);
            errdefer r.gpa.free(ids);
            const back = try r.gpa.alloc(u32, chunk_tokens);
            errdefer r.gpa.free(back);
            try r.chunks.append(r.gpa, .{ .ids = ids, .back = back });
        }
        const c = r.chunks.items[r.chunks.items.len - 1];
        const at = p % chunk_tokens;
        c.ids[at] = t;
        c.back[at] = 0;
        r.end = p + 1;
        if (t == mark) {
            r.run = 0;
            return;
        }
        std.mem.copyForwards(u32, r.last[0 .. key_len - 1], r.last[1..]);
        r.last[key_len - 1] = t;
        r.run += 1;
        if (r.run < key_len) return;
        const b = r.bucket(&r.last);
        if (r.heads[b] != 0) c.back[at] = p - (r.heads[b] - 1);
        r.heads[b] = p + 1;
    }

    /// The oldest chunk freed (never the newest): links into it end the searches that reach them.
    pub fn dropOldest(r: *Record) bool {
        if (r.chunks.items.len < 2) return false;
        r.free(r.chunks.orderedRemove(0));
        r.first += 1;
        return true;
    }

    /// Every id gone and the key table zeroed: positions start at 0 again.
    pub fn clear(r: *Record) void {
        for (r.chunks.items) |c| r.free(c);
        r.chunks.clearRetainingCapacity();
        @memset(r.heads, 0);
        r.first = 0;
        r.end = 0;
        r.run = 0;
        r.last = @splat(mark);
    }

    /// Whether the key ending at `p` holds `key`.
    fn keyAt(r: *const Record, p: u32, key: []const u32) bool {
        const from = p + 1 - key_len;
        for (key, 0..) |t, i| if (r.id(from + @as(u32, @intCast(i))) != t) return false;
        return true;
    }

    /// Positions whose keys may hold `key`, newest first, at most max_steps, each held whole.
    const Walk = struct {
        r: *const Record,
        head: u32,
        steps: u32 = 0,

        fn next(w: *Walk) ?u32 {
            if (w.steps == max_steps or w.head < w.r.base() + key_len) return null;
            w.steps += 1;
            const p = w.head - 1;
            const d = w.r.backOf(p);
            w.head = if (d == 0) 0 else p - d + 1;
            return p;
        }
    };

    fn walk(r: *const Record, key: *const [key_len]u32) Walk {
        return .{ .r = r, .head = r.heads[r.bucket(key)] };
    }

    /// How far back from the key ending at `p` the record matches the context's tail (key_len to max_match).
    fn extend(r: *const Record, p: u32, ctx: []const u32) u32 {
        var n: u32 = key_len;
        while (n < max_match and n < ctx.len and p >= r.base() + n) : (n += 1) {
            const t = r.id(p - n);
            if (t == mark or t != ctx[ctx.len - 1 - n]) break;
        }
        return n;
    }

    /// The longest match of `ctx`'s last ids with an id after it, the newest of equal length.
    pub fn longest(r: *const Record, ctx: []const u32) Match {
        if (ctx.len < key_len) return .{};
        const key = ctx[ctx.len - key_len ..][0..key_len];
        var best: Match = .{};
        var w = r.walk(key);
        while (w.next()) |p| {
            if (!r.keyAt(p, key) or p + 1 >= r.end or r.id(p + 1) == mark) continue;
            const n = r.extend(p, ctx);
            if (n <= best.len) continue;
            best = .{ .len = n, .at = p + 1 };
            if (n == max_match) break;
        }
        return best;
    }

    /// The ids from `at` up to a mark, the end or `out.len`, copied into `out`.
    pub fn copy(r: *const Record, at: u32, out: []u32) []u32 {
        var n: usize = 0;
        while (n < out.len and @as(u64, at) + n < r.end) : (n += 1) {
            const t = r.id(at + @as(u32, @intCast(n)));
            if (t == mark) break;
            out[n] = t;
        }
        return out[0..n];
    }

    /// Up to `out.len` continuations of `ctx` matching `min` ids or more whose first `span` ids differ, newest first.
    pub fn continuations(r: *const Record, ctx: []const u32, min: u32, span: usize, scratch: []u32, out: []Found) usize {
        if (ctx.len < key_len or out.len == 0) return 0;
        const key = ctx[ctx.len - key_len ..][0..key_len];
        var n: usize = 0;
        var w = r.walk(key);
        while (w.next()) |p| {
            if (!r.keyAt(p, key) or p + 1 >= r.end or r.id(p + 1) == mark) continue;
            const len = r.extend(p, ctx);
            if (len < min) continue;
            const mine = r.copy(p + 1, scratch[0..span]);
            const seen = for (out[0..n]) |f| {
                if (r.same(f.at, mine)) break true;
            } else false;
            if (seen) continue;
            out[n] = .{ .at = p + 1, .len = len };
            n += 1;
            if (n == out.len) break;
        }
        return n;
    }

    /// Whether the ids from `at` start with `ids` and stop there or continue past them.
    fn same(r: *const Record, at: u32, ids: []const u32) bool {
        for (ids, 0..) |t, i| {
            const q = @as(u64, at) + i;
            if (q >= r.end or r.id(@intCast(q)) != t) return false;
        }
        return true;
    }

    /// The longest prefix of `s` the record holds in one run, and the position where it starts (len 0: none).
    pub fn heldRun(r: *const Record, s: []const u32) Run {
        if (s.len < key_len) return .{};
        const key = s[0..key_len];
        var best: Run = .{};
        var w = r.walk(key);
        while (w.next()) |p| {
            if (!r.keyAt(p, key)) continue;
            var n: usize = key_len;
            while (n < s.len) : (n += 1) {
                const q = @as(u64, p) + 1 + n - key_len;
                if (q >= r.end) break;
                const t = r.id(@intCast(q));
                if (t == mark or t != s[n]) break;
            }
            if (n <= best.len) continue;
            best = .{ .len = n, .at = p + 1 - key_len };
            if (n == s.len) break;
        }
        return best;
    }
};

test {
    _ = @import("recall_record_test.zig");
}
