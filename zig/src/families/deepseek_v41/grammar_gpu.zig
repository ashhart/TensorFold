//! Structured output on the GPU forward: rank 0 sends each pick's masked rows (op_grammar); every rank masks its part.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const iface = @import("draft/iface.zig");
const partition = @import("partition.zig");
const Allocator = std.mem.Allocator;

/// One "w.logits" row's allowed tokens for the next picks: token t is bit t % 32 of word t / 32 (lanes/grammar.zig).
pub const Row = struct { row: u32, bits: []const u32 };

/// The most masked rows one forward stages (a row window's).
pub const max_rows = 64;

/// The rows the next picks mask, as rank 0 staged them and its followers received them.
pub const Staged = struct {
    rows: std.ArrayList(u32) = .empty,
    bits: std.ArrayList(u32) = .empty, // `words` a row, in `rows` order
    done: std.ArrayList(bool) = .empty, // the row's logits are masked already: a later pick of the forward skips it
    words: usize = 0,

    pub fn deinit(st: *Staged, gpa: Allocator) void {
        st.rows.deinit(gpa);
        st.bits.deinit(gpa);
        st.done.deinit(gpa);
    }

    pub fn count(st: *const Staged) usize {
        return st.rows.items.len;
    }

    fn clear(st: *Staged) void {
        st.rows.clearRetainingCapacity();
        st.bits.clearRetainingCapacity();
        st.done.clearRetainingCapacity();
        st.words = 0;
    }

    /// Replaces the staged rows with `rows` (one width of bits); a row that allows every token is left out.
    pub fn set(st: *Staged, gpa: Allocator, rows: []const Row) !void {
        var words: usize = 0;
        var n: usize = 0;
        for (rows) |r| {
            if (allowsAll(r.bits)) continue;
            if (words != 0 and r.bits.len != words) return error.InvalidGrammarMasks;
            words = r.bits.len;
            n += 1;
        }
        if (n > max_rows) return error.InvalidGrammarMasks;
        try st.rows.ensureTotalCapacity(gpa, n);
        try st.bits.ensureTotalCapacity(gpa, n * words);
        try st.done.ensureTotalCapacity(gpa, n);
        st.clear();
        st.words = words;
        for (rows) |r| if (!allowsAll(r.bits)) {
            st.rows.appendAssumeCapacity(r.row);
            st.bits.appendSliceAssumeCapacity(r.bits);
            st.done.appendAssumeCapacity(false);
        };
    }

    /// op_grammar's message: [op, rows, words, each row, then the bits two to a word].
    pub fn encode(st: *const Staged, gpa: Allocator, out: *std.ArrayList(i64)) !void {
        out.clearRetainingCapacity();
        try out.appendSlice(gpa, &.{ fwd.Forward.op_grammar, @intCast(st.count()), @intCast(st.words) });
        for (st.rows.items) |r| try out.append(gpa, r);
        const b = st.bits.items;
        var i: usize = 0;
        while (i < b.len) : (i += 2) {
            const hi: u64 = if (i + 1 < b.len) b[i + 1] else 0;
            try out.append(gpa, @bitCast(@as(u64, b[i]) | hi << 32));
        }
    }

    /// The rows a follower received in `encode`'s message; a malformed one is error.BadPlan.
    pub fn decode(st: *Staged, gpa: Allocator, msg: []const i64) !void {
        if (msg.len < 3 or msg[1] < 0 or msg[1] > max_rows or msg[2] < 0 or msg[2] > 1 << 20) return error.BadPlan;
        const n: usize = @intCast(msg[1]);
        const words: usize = @intCast(msg[2]);
        const total = n * words;
        if ((n > 0 and words == 0) or msg.len != 3 + n + (total + 1) / 2) return error.BadPlan;
        for (msg[3..][0..n]) |r| if (r < 0 or r > std.math.maxInt(u32)) return error.BadPlan;
        try st.rows.ensureTotalCapacity(gpa, n);
        try st.bits.ensureTotalCapacity(gpa, total);
        try st.done.ensureTotalCapacity(gpa, n);
        st.clear();
        st.words = words;
        for (msg[3..][0..n]) |r| {
            st.rows.appendAssumeCapacity(@intCast(r));
            st.done.appendAssumeCapacity(false);
        }
        for (0..total) |i| {
            const w: u64 = @bitCast(msg[3 + n + i / 2]);
            st.bits.appendAssumeCapacity(@truncate(w >> @intCast(32 * (i % 2))));
        }
    }
};

/// Whether `bits` allow every token they cover (no grammar on the row, or one not started yet).
pub fn allowsAll(bits: []const u32) bool {
    for (bits) |w| if (w != std.math.maxInt(u32)) return false;
    return true;
}

/// Turns a row's disallowed logits to -inf: column c is token lo + c; tokens past `bits` are disallowed.
pub fn maskRow(row: []f32, lo: usize, bits: []const u32) void {
    var c: usize = 0;
    while (c < row.len) {
        const t = lo + c;
        const word: u32 = if (t / 32 < bits.len) bits[t / 32] else 0;
        const end = @min(row.len, c + 32 - t % 32);
        if (word != std.math.maxInt(u32)) for (row[c..end], t..) |*v, tok| {
            if ((word >> @intCast(tok % 32)) & 1 == 0) v.* = -std.math.inf(f32);
        };
        c = end;
    }
}

/// The segments' masked rows at their places in one forward (each segment's rows after the one before).
pub fn segmentRows(out: []Row, segs: []const iface.Segment) ![]Row {
    var n: usize = 0;
    var at: u32 = 0;
    for (segs) |s| {
        defer at += @intCast(s.tokens.len);
        if (s.masks.len == 0) continue;
        const rows = s.tokens.len;
        if (rows == 0 or s.masks.len % rows != 0) return error.InvalidGrammarMasks;
        const words = s.masks.len / rows;
        for (0..rows) |r| {
            const bits = s.masks[r * words ..][0..words];
            if (allowsAll(bits)) continue;
            if (n == out.len) return error.InvalidGrammarMasks;
            out[n] = .{ .row = at + @as(u32, @intCast(r)), .bits = bits };
            n += 1;
        }
    }
    return out[0..n];
}

/// Staged rows [at, at + len): "w.logits" rows from `row` and pinned slots from `at`, one copy each way.
pub const Run = struct { at: u32, row: u32, len: u32 };

/// The staged rows a pick of [row0, row0 + n) still masks, adjacent ones (staged and in "w.logits") as one run.
pub fn runs(st: *const Staged, row0: u32, n: u32, out: *[max_rows]Run) []Run {
    var k: usize = 0;
    for (st.rows.items, st.done.items, 0..) |row, done, i| {
        if (done or row < row0 or row - row0 >= n) continue;
        if (k > 0 and out[k - 1].at + out[k - 1].len == i and out[k - 1].row + out[k - 1].len == row) {
            out[k - 1].len += 1;
            continue;
        }
        out[k] = .{ .at = @intCast(i), .row = row, .len = 1 };
        k += 1;
    }
    return out[0..k];
}

/// Every rank's grammar hook on its forward: the staged rows and the pinned copy their logits are masked in.
pub const Masks = struct {
    gpa: Allocator,
    f: *fwd.Forward,
    lo: u32, // this rank's first column's token
    width: u32, // its columns: a "w.logits" row
    st: Staged = .{},
    msg: std.ArrayList(i64) = .empty,
    host: cuda.HostBuffer, // max_rows rows: a staged row's slot is its place in the staged list

    /// Every rank, at load: the forward's grammar hook over this rank's part of the vocabulary, its pinned rows.
    pub fn init(gpa: Allocator, f: *fwd.Forward) !*Masks {
        const part = try partition.share(f.cfg.vocab, f.comm.rank(), f.comm.world());
        var host = try cuda.HostBuffer.alloc(f.runner.d, max_rows * 4 * @as(usize, part.len));
        errdefer host.free();
        const x = try gpa.create(Masks);
        x.* = .{ .gpa = gpa, .f = f, .lo = part.lo, .width = part.len, .host = host };
        f.grammar = .{ .ptr = x, .follow = follow, .apply = apply };
        return x;
    }

    pub fn deinit(x: *Masks) void {
        x.f.grammar = null;
        x.st.deinit(x.gpa);
        x.msg.deinit(x.gpa);
        x.host.free();
        x.gpa.destroy(x);
    }

    fn self(ptr: *anyopaque) *Masks {
        return @ptrCast(@alignCast(ptr));
    }

    /// Rank 0: the next picks' masked rows (none: unconstrained), sent to the followers whenever something is staged.
    pub fn stage(x: *Masks, rows: []const Row) !void {
        const had = x.st.count() > 0;
        try x.st.set(x.gpa, rows);
        if (!had and x.st.count() == 0) return;
        if (!x.f.leads()) return;
        try x.st.encode(x.gpa, &x.msg);
        try x.f.link.?.send(x.msg.items);
    }

    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x = self(ptr);
        try x.st.decode(x.gpa, msg);
    }

    /// Before a pick of rows [row0, row0 + n): the staged rows there, masked once in the pinned copy and written back.
    fn apply(ptr: *anyopaque, row0: u32, n: u32) anyerror!void {
        const x = self(ptr);
        const st = &x.st;
        var buf: [max_rows]Run = undefined;
        const rs = runs(st, row0, n, &buf);
        if (rs.len == 0) return;
        const r = x.f.runner;
        const row_bytes = 4 * @as(usize, x.width);
        const host = x.host.bytes;
        const logits: []f32 = @alignCast(std.mem.bytesAsSlice(f32, host));
        const base = r.addressOf("w.logits") orelse return error.Unbound;
        for (rs) |q| {
            const dst = host[q.at * row_bytes ..][0 .. q.len * row_bytes];
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, base + q.row * row_bytes, dst.len, r.stream.handle), "cuMemcpyDtoHAsync");
        }
        try r.stream.synchronize();
        for (rs) |q| for (q.at..q.at + q.len) |i| {
            maskRow(logits[i * x.width ..][0..x.width], x.lo, st.bits.items[i * st.words ..][0..st.words]);
            st.done.items[i] = true;
        };
        for (rs) |q| try cuda.copy_source.enqueue(r.d, base + q.row * row_bytes, host[q.at * row_bytes ..][0 .. q.len * row_bytes], r.stream);
    }
};
