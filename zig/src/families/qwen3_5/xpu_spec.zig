//! Speculative greedy decoding with a pluggable drafter; row-invariant window verification keeps output greedy.

const std = @import("std");
const model = @import("xpu_model.zig");
const qb = @import("xpu_blocks.zig");
const mtp = @import("xpu_mtp.zig");
const exl3 = @import("xpu").exl3;

pub const Stats = struct { windows: u64 = 0, plain: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: u64 = 0 };

/// A source of draft tokens: `propose` fills `out` with up to k tokens after `hist`; `observe` reports kept rows.
pub const Drafter = struct {
    ptr: *anyopaque,
    proposeFn: *const fn (ptr: *anyopaque, hist: []const u32, k: usize, out: []u32) usize,
    observeFn: *const fn (ptr: *anyopaque, m: *model.Model, hidden: qb.Buf, n: usize, next: []const i32, pos0: u32, want: bool, k: usize) anyerror!void,
    /// After a window that accepted nothing, draft again only after this many plain steps (matches come in runs).
    cooldown: u32 = 0,

    pub fn propose(self: Drafter, hist: []const u32, k: usize, out: []u32) usize {
        return self.proposeFn(self.ptr, hist, k, out);
    }

    pub fn observe(self: Drafter, m: *model.Model, hidden: qb.Buf, n: usize, next: []const i32, pos0: u32, want: bool, k: usize) !void {
        try self.observeFn(self.ptr, m, hidden, n, next, pos0, want, k);
    }
};

pub const Match = struct { cnt: usize, n: usize };

/// Copies up to k tokens after the latest earlier match of ctx's longest suffix (max_n..min_n); returns count, length.
pub fn draftN(ctx: []const u32, k: usize, max_n: usize, min_n: usize, out: []u32) Match {
    const len = ctx.len;
    if (len < 2 or k == 0) return .{ .cnt = 0, .n = 0 };
    var n = @min(max_n, len - 1);
    while (n >= min_n and n > 0) : (n -= 1) {
        const suffix = ctx[len - n ..];
        var p: usize = len - n; // a match must leave at least one following token
        while (p > 0) {
            p -= 1;
            if (!std.mem.eql(u32, ctx[p .. p + n], suffix)) continue;
            const start = p + n;
            const cnt = @min(k, len - start);
            @memcpy(out[0..cnt], ctx[start .. start + cnt]);
            return .{ .cnt = cnt, .n = n };
        }
    }
    return .{ .cnt = 0, .n = 0 };
}

/// Index of the context g-grams by last position: a match costs a chain of candidates, not the history.
pub const GramIndex = struct {
    const none = std.math.maxInt(u32);
    const bits = 18;
    const max_chain = 512;
    g: usize = 0,
    head: []u32 = &.{},
    prev: std.ArrayList(u32) = .empty, // prev[e]: the earlier end position with the same gram hash
    next: usize = 0, // grams ending before this position are indexed

    fn hash(t: []const u32) usize {
        var h: u32 = 0x9e3779b9;
        for (t) |x| h = (h ^ x) *% 0x85ebca6b;
        return (h ^ (h >> 15)) >> (32 - bits);
    }

    pub fn deinit(self: *GramIndex) void {
        std.heap.page_allocator.free(self.head);
        self.prev.deinit(std.heap.page_allocator);
        self.* = .{};
    }

    /// Indexes the grams ending before the last position (the suffix is the query); false when out of memory.
    fn update(self: *GramIndex, ctx: []const u32, g: usize) bool {
        const a = std.heap.page_allocator;
        if (self.g != g or ctx.len < self.next) self.deinit();
        if (self.head.len == 0) {
            self.head = a.alloc(u32, 1 << bits) catch return false;
            @memset(self.head, none);
            self.g = g;
            self.next = g - 1;
        }
        const end = ctx.len - 1; // exclusive
        if (self.prev.items.len < end) self.prev.resize(a, end) catch return false;
        while (self.next < end) : (self.next += 1) {
            const e = self.next;
            const h = hash(ctx[e + 1 - g .. e + 1]);
            self.prev.items[e] = self.head[h];
            self.head[h] = @intCast(e);
        }
        return true;
    }

    pub fn draft(self: *GramIndex, ctx: []const u32, k: usize, max_n: usize, min_n: usize, out: []u32) Match {
        const len = ctx.len;
        if (len < 2 or k == 0 or min_n == 0 or min_n > len - 1) return .{ .cnt = 0, .n = 0 };
        if (!self.update(ctx, min_n)) return draftN(ctx, k, max_n, min_n, out);
        var best_e: usize = 0;
        var best_n: usize = 0;
        var e: u32 = self.head[hash(ctx[len - min_n ..])];
        var steps: usize = 0;
        while (e != none and steps < max_chain) : (steps += 1) {
            var l: usize = 0; // common length of the text ending at e and the suffix
            const cap = @min(@min(max_n, len - 1), @as(usize, e) + 1);
            while (l < cap and ctx[e - l] == ctx[len - 1 - l]) l += 1;
            if (l >= min_n and l > best_n) {
                best_n = l;
                best_e = e;
                if (l == @min(max_n, len - 1)) break;
            }
            e = self.prev.items[e];
        }
        if (best_n == 0) return .{ .cnt = 0, .n = 0 };
        const start = best_e + 1;
        const cnt = @min(k, len - start);
        @memcpy(out[0..cnt], ctx[start .. start + cnt]);
        return .{ .cnt = cnt, .n = best_n };
    }
};

pub const CopyDrafter = struct {
    min_n: usize = 3,
    max_n: usize = 6,
    idx: GramIndex = .{},

    fn propose(p: *anyopaque, hist: []const u32, k: usize, out: []u32) usize {
        const self: *CopyDrafter = @ptrCast(@alignCast(p));
        return self.idx.draft(hist, k, self.max_n, self.min_n, out).cnt;
    }

    fn observe(_: *anyopaque, _: *model.Model, _: qb.Buf, _: usize, _: []const i32, _: u32, _: bool, _: usize) anyerror!void {}

    pub fn drafter(self: *CopyDrafter) Drafter {
        return .{ .ptr = self, .proposeFn = propose, .observeFn = observe, .cooldown = 4 };
    }
};

/// The MTP head as a drafter: observe absorbs the kept rows into its cache and chains the next drafts on the device.
pub const MtpDrafter = struct {
    mt: *mtp.Mtp,
    st: mtp.Stats = .{},
    drafts: [16]u32 = undefined,
    nd: usize = 0,

    /// `first` drafts come from mtp.prefill.
    pub fn init(mt: *mtp.Mtp, first: []const u32) MtpDrafter {
        var d: MtpDrafter = .{ .mt = mt };
        @memcpy(d.drafts[0..first.len], first);
        d.nd = first.len;
        return d;
    }

    fn propose(p: *anyopaque, _: []const u32, k: usize, out: []u32) usize {
        const self: *MtpDrafter = @ptrCast(@alignCast(p));
        const n = @min(self.nd, k);
        @memcpy(out[0..n], self.drafts[0..n]);
        return n;
    }

    fn observe(p: *anyopaque, m: *model.Model, hidden: qb.Buf, n: usize, next: []const i32, pos0: u32, want: bool, k: usize) anyerror!void {
        const self: *MtpDrafter = @ptrCast(@alignCast(p));
        const hs: usize = qb.hidden * 2;
        var stg: [16]u32 = undefined;
        for (0..n) |i| stg[i] = @intCast(next[i]);
        try self.mt.stage(stg[0..n]);
        for (0..n) |i| {
            const is_last = i == n - 1 and want;
            try self.mt.step(m, exl3.at(self.mt.tokdev, i * 4), exl3.at(hidden, i * hs), pos0 + @as(u32, @intCast(i)), if (is_last) self.mt.drafts else null);
            if (is_last) self.st.steps += 1 else self.st.absorbs += 1;
        }
        self.nd = 0;
        if (want) {
            self.nd = @min(k, 15);
            try self.mt.chain(m, 1, self.nd, pos0 + @as(u32, @intCast(n)), self.drafts[0..], &self.st);
        }
    }

    pub fn drafter(self: *MtpDrafter) Drafter {
        return .{ .ptr = self, .proposeFn = propose, .observeFn = observe };
    }
};

/// Context copy on a match of at least `long_n` tokens, else the MTP head (which observes every step).
pub const AutoDrafter = struct {
    mtp_d: *MtpDrafter,
    long_n: usize = 6,
    copies: u64 = 0,
    idx: GramIndex = .{},

    fn propose(p: *anyopaque, hist: []const u32, k: usize, out: []u32) usize {
        const self: *AutoDrafter = @ptrCast(@alignCast(p));
        const c = self.idx.draft(hist, k, 12, self.long_n, out);
        if (c.cnt > 0) {
            self.copies += 1;
            return c.cnt;
        }
        return MtpDrafter.propose(self.mtp_d, hist, k, out);
    }

    fn observe(p: *anyopaque, m: *model.Model, hidden: qb.Buf, n: usize, next: []const i32, pos0: u32, want: bool, k: usize) anyerror!void {
        const self: *AutoDrafter = @ptrCast(@alignCast(p));
        try MtpDrafter.observe(self.mtp_d, m, hidden, n, next, pos0, want, k);
    }

    pub fn drafter(self: *AutoDrafter) Drafter {
        return .{ .ptr = self, .proposeFn = propose, .observeFn = observe };
    }
};

fn isEos(t: u32, eos: []const u32) bool {
    for (eos) |e| if (t == e) return true;
    return false;
}

/// Appends up to n_new tokens to ctx; needs m.pos == ctx.len - 1 and a drafter that has seen the same context.
pub fn run(m: *model.Model, gpa: std.mem.Allocator, ctx: *std.ArrayList(u32), n_new: usize, max_k: usize, d: Drafter, eos: []const u32, ignore_eos: bool, st: *Stats) !void {
    var produced: usize = 0;
    var win: [16]u32 = undefined;
    var drafts: [16]u32 = undefined;
    var g: [16]i32 = undefined;
    var cool: u32 = 0; // plain steps left before drafting again after a window that accepted nothing
    while (produced < n_new) {
        if (m.pos + 20 > m.ops.cap) return error.ContextFull; // window + chained draft positions must fit the caches
        const last = ctx.items[ctx.items.len - 1];
        const room = @min(@min(max_k, 15), n_new - produced - 1);
        const kd = if (room == 0 or cool > 0) 0 else d.propose(ctx.items, room, &drafts);
        if (cool > 0) cool -= 1;
        const pos0 = m.pos;
        var a: usize = 0;
        var hidden: qb.Buf = m.ops.s.xn;
        if (kd == 0) {
            try m.forward(last, true);
            g[0] = @intCast(try m.argmax());
            st.plain += 1;
        } else {
            win[0] = last;
            @memcpy(win[1 .. 1 + kd], drafts[0..kd]);
            try m.forwardRows(win[0 .. kd + 1], @intCast(kd + 1), false);
            try m.rowArgmax(&g);
            while (a < kd and win[1 + a] == @as(u32, @intCast(g[a]))) a += 1;
            try m.commitRows(@intCast(a + 1));
            hidden = m.win.xn;
            st.windows += 1;
            st.drafted += kd;
            st.accepted += a;
            if (a == 0) cool = d.cooldown;
        }
        const want = produced + a + 1 < n_new; // another step follows
        try d.observe(m, hidden, a + 1, g[0 .. a + 1], pos0, want, max_k);
        for (0..a + 1) |i| {
            const t: u32 = @intCast(g[i]);
            try ctx.append(gpa, t);
            produced += 1;
            st.tokens += 1;
            if (!ignore_eos and isEos(t, eos)) return;
        }
    }
}
