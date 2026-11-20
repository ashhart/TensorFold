//! Python's difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() over interned tokens.
const std = @import("std");

pub const Tag = enum { equal, replace, delete, insert };

/// a[i1..i2] becomes b[j1..j2].
pub const Op = struct { tag: Tag, i1: usize, i2: usize, j1: usize, j2: usize };

const Block = struct { a: usize, b: usize, n: usize };

const Matcher = struct {
    a: []const u32,
    b: []const u32,
    /// b's indices by element, ascending: `at[starts[e] .. starts[e + 1]]` for element e.
    starts: []usize,
    at: []usize,
    len: []usize,
    next: []usize,

    /// find_longest_match: the longest block of a[alo..ahi] in b[blo..bhi]; of equals, the earliest in a, then in b.
    fn longest(m: *Matcher, alo: usize, ahi: usize, blo: usize, bhi: usize) Block {
        var best: Block = .{ .a = alo, .b = blo, .n = 0 };
        @memset(m.len, 0);
        for (alo..ahi) |i| {
            @memset(m.next, 0);
            const e = m.a[i];
            if (e + 1 < m.starts.len) for (m.at[m.starts[e]..m.starts[e + 1]]) |j| {
                if (j < blo) continue;
                if (j >= bhi) break;
                const k = (if (j > 0) m.len[j - 1] else 0) + 1;
                m.next[j] = k;
                if (k > best.n) best = .{ .a = i + 1 - k, .b = j + 1 - k, .n = k };
            };
            std.mem.swap([]usize, &m.len, &m.next);
        }
        return best;
    }
};

fn before(_: void, x: Block, y: Block) bool {
    return if (x.a != y.a) x.a < y.a else x.b < y.b;
}

/// The opcodes that turn `a` into `b` (caller frees); elements are small ids (interned tokens).
pub fn opcodes(gpa: std.mem.Allocator, a: []const u32, b: []const u32) ![]Op {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const s = arena.allocator();
    var top: usize = 0;
    for (b) |e| top = @max(top, e + 1);
    var m: Matcher = .{ .a = a, .b = b, .starts = try s.alloc(usize, top + 1), .at = try s.alloc(usize, b.len), .len = try s.alloc(usize, b.len), .next = try s.alloc(usize, b.len) };
    @memset(m.starts, 0);
    for (b) |e| m.starts[e + 1] += 1;
    for (1..top + 1) |e| m.starts[e] += m.starts[e - 1];
    const fill = try s.dupe(usize, m.starts[0..top]);
    for (b, 0..) |e, j| {
        m.at[fill[e]] = j;
        fill[e] += 1;
    }
    // get_matching_blocks: the longest block, then the same on both sides of it
    const Range = struct { alo: usize, ahi: usize, blo: usize, bhi: usize };
    var queue: std.ArrayList(Range) = .empty;
    var blocks: std.ArrayList(Block) = .empty;
    try queue.append(s, .{ .alo = 0, .ahi = a.len, .blo = 0, .bhi = b.len });
    while (queue.pop()) |r| {
        const x = m.longest(r.alo, r.ahi, r.blo, r.bhi);
        if (x.n == 0) continue;
        try blocks.append(s, x);
        if (r.alo < x.a and r.blo < x.b) try queue.append(s, .{ .alo = r.alo, .ahi = x.a, .blo = r.blo, .bhi = x.b });
        if (x.a + x.n < r.ahi and x.b + x.n < r.bhi) try queue.append(s, .{ .alo = x.a + x.n, .ahi = r.ahi, .blo = x.b + x.n, .bhi = r.bhi });
    }
    std.mem.sort(Block, blocks.items, {}, before);
    var out: std.ArrayList(Op) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    while (k <= blocks.items.len) {
        // adjacent blocks are one; the end of both sequences closes the list
        var x: Block = if (k < blocks.items.len) blocks.items[k] else .{ .a = a.len, .b = b.len, .n = 0 };
        k += 1;
        while (k < blocks.items.len and x.n != 0 and blocks.items[k].a == x.a + x.n and blocks.items[k].b == x.b + x.n) : (k += 1) x.n += blocks.items[k].n;
        const tag: ?Tag = if (i < x.a and j < x.b) .replace else if (i < x.a) .delete else if (j < x.b) .insert else null;
        if (tag) |t| try out.append(gpa, .{ .tag = t, .i1 = i, .i2 = x.a, .j1 = j, .j2 = x.b });
        i = x.a + x.n;
        j = x.b + x.n;
        if (x.n != 0) try out.append(gpa, .{ .tag = .equal, .i1 = x.a, .i2 = i, .j1 = x.b, .j2 = j });
    }
    return out.toOwnedSlice(gpa);
}

test "opcodes: difflib's own answers" {
    const t = std.testing;
    // SequenceMatcher(None, "qabxcd", "abycdf", autojunk=False).get_opcodes()
    const a = [_]u32{ 16, 0, 1, 23, 2, 3 };
    const b = [_]u32{ 0, 1, 24, 2, 3, 5 };
    const got = try opcodes(t.allocator, &a, &b);
    defer t.allocator.free(got);
    const want = [_]Op{
        .{ .tag = .delete, .i1 = 0, .i2 = 1, .j1 = 0, .j2 = 0 },
        .{ .tag = .equal, .i1 = 1, .i2 = 3, .j1 = 0, .j2 = 2 },
        .{ .tag = .replace, .i1 = 3, .i2 = 4, .j1 = 2, .j2 = 3 },
        .{ .tag = .equal, .i1 = 4, .i2 = 6, .j1 = 3, .j2 = 5 },
        .{ .tag = .insert, .i1 = 6, .i2 = 6, .j1 = 5, .j2 = 6 },
    };
    try t.expectEqualSlices(Op, &want, got);
    const same = try opcodes(t.allocator, &a, &a);
    defer t.allocator.free(same);
    try t.expectEqualSlices(Op, &.{.{ .tag = .equal, .i1 = 0, .i2 = 6, .j1 = 0, .j2 = 6 }}, same);
    const none = try opcodes(t.allocator, &.{}, &b);
    defer t.allocator.free(none);
    try t.expectEqualSlices(Op, &.{.{ .tag = .insert, .i1 = 0, .i2 = 0, .j1 = 0, .j2 = 6 }}, none);
}
