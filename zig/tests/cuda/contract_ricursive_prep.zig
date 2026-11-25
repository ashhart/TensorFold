//! Ripple's CUDA attention prep under the contract: tf_rc_prep with its controls, tf_rc_prep_rows bit-equal to it.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rp = @import("ricursive_cuda").prep;
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const D = 128;
const heads = 80;
const kv_heads = 8;
const kvw = kv_heads * D;
const qk_stride = (heads + kv_heads) * D; // a fused q | k projection's row; v is its own projection
const width = (heads + 2 * kv_heads) * D; // a row's outputs: its q, k and v heads
const eps: f32 = 1e-6;
const theta: f64 = 10000;
const sliding_ring = 3072; // state.zig: the window less 1 plus a 2,048-row chunk, to 128s
const page = 512; // positions a global layer's page holds: tree_attention's span, as Kolibri-1's pages

/// ricursive_prep.cuh's rounding points.
const norm_row: i32 = 1;
const norm_out: i32 = 2;
const cos_pt: i32 = 4;
const sin_pt: i32 = 8;
const out_pt: i32 = 16;
const all: i32 = 31;

/// Layer kinds: sliding (RoPE, the 3,072-slot ring), global (NoPE, a ring that never wraps), global in 512-row pages.
pub const Form = enum { sliding, global, paged };

/// A registration: the served kernel (keep null), or a control keeping every rounding point but one.
pub const Entry = struct { name: []const u8, form: Form, keep: ?i32 = null };

/// NoPE stores the normed value as it is, so the norm's last rounding and the output's show only on sliding layers.
pub const entries = [_]Entry{
    .{ .name = "cuda ricursive tf_rc_prep sliding (RoPE, the 3,072-slot ring)", .form = .sliding },
    .{ .name = "cuda ricursive tf_rc_prep global (NoPE, a slot a position)", .form = .global },
    .{ .name = "cuda ricursive tf_rc_prep global over pages of 512 (NoPE, a shuffled page table)", .form = .paged },
    .{ .name = "control: ricursive tf_rc_prep drops bf16(x r)", .form = .sliding, .keep = all & ~norm_row },
    .{ .name = "control: ricursive tf_rc_prep drops bf16(w bf16(x r))", .form = .sliding, .keep = all & ~norm_out },
    .{ .name = "control: ricursive tf_rc_prep drops bf16(cos)", .form = .sliding, .keep = all & ~cos_pt },
    .{ .name = "control: ricursive tf_rc_prep drops bf16(sin)", .form = .sliding, .keep = all & ~sin_pt },
    .{ .name = "control: ricursive tf_rc_prep truncates its output, not rounding it", .form = .sliding, .keep = all & ~out_pt },
};

/// Rows x a row's outputs x the call's end position (pos + rows): windows across the wrap, deep and last; chunks.
const sliding_served = [_]c.Shape{
    .{ .rows = 16, .n = width, .k = 3 * sliding_ring + 9 }, // a decode window across the ring's wrap
    .{ .rows = 1, .n = width, .k = sliding_ring }, // the ring's last slot
    .{ .rows = 1, .n = width, .k = sliding_ring + 1 }, // slot 0 again
    .{ .rows = 7, .n = width, .k = 100_007 },
    .{ .rows = 16, .n = width, .k = 262_144 }, // the longest context checked
    .{ .rows = 2048, .n = width, .k = 2048 }, // the first prompt chunk
    .{ .rows = 2048, .n = width, .k = 4096 }, // the second, across the wrap
    .{ .rows = 1000, .n = width, .k = 2 * sliding_ring + 500 }, // half before the wrap, half after
    .{ .rows = 333, .n = width, .k = 333 },
    .{ .rows = 333, .n = width, .k = sliding_ring + 100 },
};

/// The ring is slots(capacity) for a capacity of the end position: k a multiple of 64 ends on its last slot.
const global_served = [_]c.Shape{
    .{ .rows = 16, .n = width, .k = 100_016 },
    .{ .rows = 1, .n = width, .k = 262_144 }, // the last position of the longest context
    .{ .rows = 5, .n = width, .k = 704 },
    .{ .rows = 2048, .n = width, .k = 2048 },
    .{ .rows = 1000, .n = width, .k = 3048 },
    .{ .rows = 333, .n = width, .k = 65_869 },
};

/// The global shapes over pages, and a decode window across a page's edge (positions 1,016 to 1,031).
const paged_served = global_served ++ [_]c.Shape{.{ .rows = 16, .n = width, .k = 1032 }};

/// A lane's 4 fma steps, the warp's 5 shuffles.
fn depth(_: c.Shape) usize {
    return 4 + 5;
}

pub fn prepSpec(e: Entry) c.Spec {
    const served: []const c.Shape = switch (e.form) {
        .sliding => &sliding_served,
        .global => &global_served,
        .paged => &paged_served,
    };
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .served = served, .sums = .{} };
}

/// q [rows][80][128], then the k and v rings' held slots, or (paged) every row of the call's pages and two spares.
pub const Prep = struct {
    kern: *const rp.Kernels,
    entry: Entry,
    ctl: ?cuda.Function = null,
    shape: c.Shape = undefined,
    pos: usize = 0,
    ring: usize = 0,
    first: usize = 0, // the lowest slot the outputs hold
    held: usize = 0, // the slots they hold
    qk: []u16 = &.{},
    v: []u16 = &.{},
    qw: []u16 = &.{},
    kw: []u16 = &.{},
    inv: []f32 = &.{},
    table: []i32 = &.{}, // paged: the stream's page ids by position / 512
    dev: [6]u64 = @splat(0), // qk, v, inv, qw, kw, table
    outs: [3]u64 = @splat(0), // q, and the k and v rings' bases (slot 0)

    pub const sums = {};

    pub fn spec(self: *const Prep) c.Spec {
        return prepSpec(self.entry);
    }

    pub fn control(self: *const Prep) ?g.Gate {
        return if (self.entry.keep != null) .shapes else null;
    }

    pub fn free(self: *Prep, a: std.mem.Allocator) void {
        for ([_]*[]u16{ &self.qk, &self.v, &self.qw, &self.kw }) |x| {
            a.free(x.*);
            x.* = &.{};
        }
        a.free(self.inv);
        self.inv = &.{};
        a.free(self.table);
        self.table = &.{};
    }

    /// The call at `shape`: its position, ring or pages, inputs and outputs.
    pub fn prepare(self: *Prep, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != width or shape.rows == 0 or shape.k < shape.rows) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        self.pos = shape.k - rows;
        const sliding = self.entry.form == .sliding;
        const paged = self.entry.form == .paged;
        self.ring = if (sliding) sliding_ring else if (paged) 0 else std.mem.alignForward(usize, shape.k, 64);
        if (!paged and rows > self.ring) return false;
        self.first = if (sliding or paged) 0 else self.pos;
        self.held = if (sliding) self.ring else rows;
        if (paged) {
            // the call's pages drawn from a pool two larger, in shuffled order; pages below them are never read (-1)
            const lo = self.pos / page;
            const hi = (shape.k - 1) / page;
            const pool = hi - lo + 1 + 2;
            const ids = try a.alloc(i32, pool);
            defer a.free(ids);
            for (ids, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, ids);
            self.table = try a.alloc(i32, hi + 1);
            @memset(self.table, -1);
            @memcpy(self.table[lo..], ids[0 .. hi - lo + 1]);
            self.held = pool * page;
        }
        try self.inputs(a, rig, rows, fill, rng);
        const off = self.first * kvw * 2; // the rings' base, so slot `first` lands on the buffer's first row
        self.outs[0] = try rig.output(rows * heads * D * 2); // allocated in the order the outputs gather: q, k, v
        self.outs[1] = try rig.output(self.held * kvw * 2) - off;
        self.outs[2] = try rig.output(self.held * kvw * 2) - off;
        return true;
    }

    /// Heads in [-3, 3) (one in 9 a tenth the size), weights in [0.5, 1.5); stress: integers in [-64, 64]; uploaded.
    fn inputs(self: *Prep, a: std.mem.Allocator, rig: *rig_.Rig, rows: usize, fill: g.Fill, rng: std.Random) !void {
        self.qk = try a.alloc(u16, rows * qk_stride);
        self.v = try a.alloc(u16, rows * kvw);
        for ([_][]u16{ self.qk, self.v }, [_]usize{ qk_stride, kvw }) |x, stride| for (x, 0..) |*e, i| {
            e.* = c.toBf16(if (fill == .stress) ref.int(rng, -64, 64) else (rng.float(f32) * 6 - 3) * (if (i % stride / D % 9 == 4) @as(f32, 0.1) else 1));
        };
        if (fill == .nonfinite and rows >= 4) {
            self.qk[3] = c.toBf16(ref.nan); // row 0's first q head: its every output NaN
            self.qk[qk_stride + D + 9] = c.toBf16(ref.inf); // row 1's second q head: r = 0, so dim 9 (and 73 rotated) NaN
            self.qk[2 * qk_stride + heads * D + 70] = c.toBf16(-ref.inf); // row 2's first k head, in the second half
            self.v[3 * kvw + 2 * D + 5] = c.toBf16(ref.inf); // row 3's third v head: copied as it is
        }
        self.qw = try ref.bfs(a, rng, D, 0.5, 1.5);
        self.kw = try ref.bfs(a, rng, D, 0.5, 1.5);
        self.inv = try a.alloc(f32, D / 2);
        for (self.inv, 0..) |*f, i| f.* = 1.0 / @as(f32, @floatCast(std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / D))); // config.zig's inverseFrequencies
        self.dev = .{ try ref.input(rig, self.qk), try ref.input(rig, self.v), try ref.input(rig, self.inv), try ref.input(rig, self.qw), try ref.input(rig, self.kw), if (self.table.len > 0) try ref.input(rig, self.table) else 0 };
    }

    pub fn run(self: *Prep, rig: *rig_.Rig) !void {
        const off = self.first * kvw * 2;
        if (self.held > self.shape.rows) for (self.outs[1..]) |o| try rig.zero(o + off, self.held * kvw * 2); // slots no row reaches: zero, as in the reference
        const d = self.dev;
        const p: rp.Prep = .{ .xq = rp.Rows.of(d[0], qk_stride), .xk = rp.Rows.of(d[0] + heads * D * 2, qk_stride), .xv = rp.Rows.of(d[1], kvw), .inv = d[2], .qw = d[3], .kw = d[4], .q = self.outs[0], .kc = self.outs[1], .vc = self.outs[2], .rows = self.shape.rows, .heads = heads, .kv_heads = kv_heads, .pos = self.pos, .ring = self.ring, .rope = self.entry.form == .sliding, .eps = eps, .pg = if (self.entry.form == .paged) .{ .table = d[5], .page_elems = page * kvw, .tokens = page } else null };
        if (self.ctl) |f| try rp.prepVia(f, self.entry.keep.?, rig.s, p) else try rp.prep(self.kern, rig.s, p);
    }

    pub fn count(self: *const Prep) usize {
        return self.shape.rows * heads * D + 2 * self.held * kvw;
    }

    /// One head's 128 outputs: bf16(w bf16(x r)), then on sliding layers bf16(x c + other s), c, s the bf16 cos, sin.
    fn head(self: *const Prep, s: Sums, x: []const u16, w: []const u16, cs: []const Iv, sn: []const Iv, out: []Iv, scratch: []f64) void {
        var sq: [D]Iv = undefined;
        var n: [D]Iv = undefined;
        for (&sq, x) |*q, v| q.* = Iv.of(ref.bf(v) * ref.bf(v));
        const r = s.sum(&sq, depth(self.shape), scratch).div(Iv.of(D)).add(Iv.of(eps)).rsqrt();
        for (&n, x, w) |*o, v, wv| o.* = Iv.of(ref.bf(wv)).mul(Iv.of(ref.bf(v)).mul(r).bf()).bf();
        for (out, 0..) |*o, d| {
            if (self.entry.form != .sliding) {
                o.* = n[d];
                continue;
            }
            const other = if (d < D / 2) n[d + D / 2].neg() else n[d - D / 2];
            o.* = n[d].mul(cs[d % (D / 2)]).add(other.mul(sn[d % (D / 2)])).bf();
        }
    }

    /// Position p's row in the outputs' k and v: its ring slot, or its page's row in the pool.
    fn slot(self: *const Prep, p: usize) usize {
        if (self.entry.form == .paged) return @as(usize, @intCast(self.table[p / page])) * page + p % page;
        return p % self.ring - self.first;
    }

    /// Row r's position and its row in the outputs' k and v: the call's pos + r.
    fn rowPos(self: *const Prep, r: usize) usize {
        return self.pos + r;
    }

    fn rowSlot(self: *const Prep, r: usize) usize {
        return self.slot(self.pos + r);
    }

    /// Every rounding point kept, whatever the entry: a control's kernel drops one and must leave these ranges.
    pub fn solve(self: *const Prep, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        self.solveAt(s, out, self);
    }

    /// The outputs with row r at by.rowPos(r), its k and v at row by.rowSlot(r) of the held ones (the rest zero).
    fn solveAt(self: *const Prep, s: Sums, out: []Iv, by: anytype) void {
        const rows = self.shape.rows;
        const qn = rows * heads * D;
        const ks = out[qn..][0 .. self.held * kvw];
        const vs = out[qn + self.held * kvw ..];
        @memset(ks, Iv.of(0));
        @memset(vs, Iv.of(0));
        var scratch: [D]f64 = undefined;
        var cs: [D / 2]Iv = undefined;
        var sn: [D / 2]Iv = undefined;
        for (0..rows) |r| {
            const position: f32 = @floatFromInt(by.rowPos(r));
            for (&cs, &sn, self.inv) |*co, *si, f| {
                const angle = Iv.of(position * f); // the kernel's fp32 angle, exactly
                co.* = angle.trig(false).bf();
                si.* = angle.trig(true).bf();
            }
            const t = by.rowSlot(r); // the row's slot, as the outputs hold it
            const xr = self.qk[r * qk_stride ..][0..qk_stride];
            for (0..heads) |h| self.head(s, xr[h * D ..][0..D], self.qw, &cs, &sn, out[(r * heads + h) * D ..][0..D], &scratch);
            for (0..kv_heads) |h| self.head(s, xr[(heads + h) * D ..][0..D], self.kw, &cs, &sn, ks[t * kvw + h * D ..][0..D], &scratch);
            for (vs[t * kvw ..][0..kvw], self.v[r * kvw ..][0..kvw]) |*o, x| o.* = Iv.of(ref.bf(x));
        }
    }
};

/// tf_rc_prep_rows' calls: 4 streams' rows round-robin in 5 shuffled slots (one unused), last rows 3,073 apart.
const streams = 4;
const slots = streams + 1;
const apart = sliding_ring + 1;

/// A registration of tf_rc_prep_rows (form sliding or paged): the served kernel, or the control naming no row's stream.
pub const RowsEntry = struct { name: []const u8, form: Form, keep: ?i32 = null };

pub const rows_entries = [_]RowsEntry{
    .{ .name = "cuda ricursive tf_rc_prep_rows sliding (4 streams' rows in their 3,072-slot rings)", .form = .sliding },
    .{ .name = "cuda ricursive tf_rc_prep_rows global over pages of 512 (4 streams, each its row of the page table)", .form = .paged },
    .{ .name = "control: ricursive tf_rc_prep_rows drops the slot (every row into slot 0's ring)", .form = .sliding, .keep = all },
};

/// Rows x a row's outputs x the first stream's last position + 1 (the others end 3,073, 6,146 and 9,219 below it).
const rows_served = [_]c.Shape{
    .{ .rows = 16, .n = width, .k = 4 * sliding_ring + 2 }, // verify windows of 4 ending at ring slots 1, 0, 3,071, 3,070
    .{ .rows = 4, .n = width, .k = 4 * sliding_ring + 2 }, // a decode round: a row a stream, at those slots
    .{ .rows = 7, .n = width, .k = 3 * apart + 3 }, // windows of 2, 2, 2 and 1, the last stream's row at position 2
    .{ .rows = 64, .n = width, .k = 100_007 },
    .{ .rows = 16, .n = width, .k = 262_144 }, // the longest context checked
};

/// Gate 3 holds tf_rc_prep_rows to tf_rc_prep's bits; gates 1 and 4 to the float64 reference, as tf_rc_prep.
pub fn rowsSpec(e: RowsEntry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .replaces = "tf_rc_prep (a launch a row, at its position on its stream's cache)", .served = &rows_served, .sums = .{} };
}

/// Rows stream i holds in a call of `rows`: rows i, i + 4, ...
fn streamRows(rows: usize, i: usize) usize {
    return (rows + streams - 1 - i) / streams;
}

/// Stream i's last position.
fn lastPos(shape: c.Shape, i: usize) usize {
    return shape.k - 1 - i * apart;
}

/// Stream i's pages: its first and one past its last (none without rows).
fn pageSpan(shape: c.Shape, i: usize) [2]usize {
    const n = streamRows(shape.rows, i);
    if (n == 0) return .{ 0, 0 };
    return .{ (lastPos(shape, i) + 1 - n) / page, lastPos(shape, i) / page + 1 };
}

/// tf_rc_prep_rows on Prep's inputs, against tf_rc_prep a launch a row into scratch alike (every slot's ring or pages).
pub const PrepRows = struct {
    kern: *const rp.Kernels,
    entry: RowsEntry,
    ctl: ?cuda.Function = null,
    p: Prep, // inputs, outputs and arithmetic: the form's one-stream registration's
    at: []i32 = &.{}, // row r's position
    of: []i32 = &.{}, // row r's slot
    stride: usize = 0, // paged: a slot's entries in the table
    dev: [2]u64 = @splat(0), // at, of
    base: [3]u64 = @splat(0), // tf_rc_prep's q and k and v caches
    host: std.ArrayList(u8) = .empty, // their bytes, in the outputs' order

    pub const sums = {};

    pub fn spec(self: *const PrepRows) c.Spec {
        return rowsSpec(self.entry);
    }

    pub fn control(self: *const PrepRows) ?g.Gate {
        return if (self.entry.keep != null) .shapes else null;
    }

    pub fn free(self: *PrepRows, a: std.mem.Allocator) void {
        self.p.free(a);
        a.free(self.at);
        self.at = &.{};
        a.free(self.of);
        self.of = &.{};
        self.host.deinit(a);
        self.host = .empty;
    }

    pub fn prepare(self: *PrepRows, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const rows = shape.rows;
        const most = streamRows(rows, 0);
        if (shape.n != width or rows == 0 or most > sliding_ring or shape.k < (streams - 1) * apart + most) return false;
        self.free(a);
        const p = &self.p;
        p.shape = shape;
        const paged = self.entry.form == .paged;
        var ids: [slots]i32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, &ids); // stream i in slot ids[i]
        self.at = try a.alloc(i32, rows);
        self.of = try a.alloc(i32, rows);
        for (self.at, self.of, 0..) |*x, *t, r| {
            const i = r % streams;
            x.* = @intCast(lastPos(shape, i) + 1 - streamRows(rows, i) + r / streams);
            t.* = ids[i];
        }
        p.ring = if (paged) 0 else sliding_ring;
        p.first = 0;
        p.held = slots * sliding_ring;
        if (paged) {
            // each stream's pages from a pool two larger, in shuffled order; a slot's other entries are never read (-1)
            self.stride = (shape.k - 1) / page + 1;
            var need: usize = 0;
            for (0..streams) |i| need += pageSpan(shape, i)[1] - pageSpan(shape, i)[0];
            const pool = try a.alloc(i32, need + 2);
            defer a.free(pool);
            for (pool, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, pool);
            p.table = try a.alloc(i32, slots * self.stride);
            @memset(p.table, -1);
            var next: usize = 0;
            for (0..streams) |i| {
                const span = pageSpan(shape, i);
                const row = p.table[@as(usize, @intCast(ids[i])) * self.stride ..];
                for (span[0]..span[1]) |pg| {
                    row[pg] = pool[next];
                    next += 1;
                }
            }
            p.held = pool.len * page;
        }
        try p.inputs(a, rig, rows, fill, rng);
        self.dev = .{ try ref.input(rig, self.at), try ref.input(rig, self.of) };
        const qb = rows * heads * D * 2;
        const kb = p.held * kvw * 2;
        p.outs = .{ try rig.output(qb), try rig.output(kb), try rig.output(kb) }; // gathered in this order: q, k, v
        self.base = .{ try rig.scratch(qb), try rig.scratch(kb), try rig.scratch(kb) };
        return true;
    }

    fn call(self: *const PrepRows) rp.PrepRows {
        const p = &self.p;
        const d = p.dev;
        return .{ .xq = rp.Rows.of(d[0], qk_stride), .xk = rp.Rows.of(d[0] + heads * D * 2, qk_stride), .xv = rp.Rows.of(d[1], kvw), .inv = d[2], .qw = d[3], .kw = d[4], .q = p.outs[0], .kc = p.outs[1], .vc = p.outs[2], .rows = p.shape.rows, .heads = heads, .kv_heads = kv_heads, .pos = self.dev[0], .slot = self.dev[1], .ring = p.ring, .rope = self.entry.form == .sliding, .eps = eps, .pg = if (self.entry.form == .paged) .{ .table = d[5], .page_elems = page * kvw, .tokens = page, .stride = self.stride } else null };
    }

    pub fn run(self: *PrepRows, rig: *rig_.Rig) !void {
        for (self.p.outs[1..]) |o| try rig.zero(o, self.p.held * kvw * 2); // slots no row reaches: zero, as in the reference
        if (self.ctl) |f| try rp.prepRowsVia(f, self.entry.keep.?, rig.s, self.call()) else try rp.prepRows(self.kern, rig.s, self.call());
    }

    /// Gate 3's baseline: tf_rc_prep, a launch a row at the row's position, on its slot's ring or its slot's table row.
    pub fn baseline(self: *PrepRows, rig: *rig_.Rig) ![]const u8 {
        const p = &self.p;
        const d = p.dev;
        const b = self.base;
        const kb = p.held * kvw * 2;
        for (b[1..]) |o| try rig.zero(o, kb);
        const paged = self.entry.form == .paged;
        for (self.at, self.of, 0..) |x, t, r| {
            const ts: usize = @intCast(t);
            const ring_at = ts * p.ring * kvw * 2; // the slot's ring (pages: ring 0, the pools themselves)
            try rp.prep(self.kern, rig.s, .{
                .xq = rp.Rows.of(d[0] + r * qk_stride * 2, qk_stride),
                .xk = rp.Rows.of(d[0] + (r * qk_stride + heads * D) * 2, qk_stride),
                .xv = rp.Rows.of(d[1] + r * kvw * 2, kvw),
                .inv = d[2],
                .qw = d[3],
                .kw = d[4],
                .q = b[0] + r * heads * D * 2,
                .kc = b[1] + ring_at,
                .vc = b[2] + ring_at,
                .rows = 1,
                .heads = heads,
                .kv_heads = kv_heads,
                .pos = @intCast(x),
                .ring = p.ring,
                .rope = !paged,
                .eps = eps,
                .pg = if (paged) .{ .table = d[5] + ts * self.stride * 4, .page_elems = page * kvw, .tokens = page } else null,
            });
        }
        const a = rig.gpu.gpa;
        self.host.clearRetainingCapacity();
        for (b, [_]usize{ p.shape.rows * heads * D * 2, kb, kb }) |o, n| try self.host.appendSlice(a, try rig.bytes(o, n));
        return self.host.items;
    }

    pub fn count(self: *const PrepRows) usize {
        return self.p.count();
    }

    /// Row r's position, and its row in the outputs' k and v: its slot's ring slot, or its page's row in the pool.
    fn rowPos(self: *const PrepRows, r: usize) usize {
        return @intCast(self.at[r]);
    }

    fn rowSlot(self: *const PrepRows, r: usize) usize {
        const x: usize = @intCast(self.at[r]);
        const t: usize = @intCast(self.of[r]);
        if (self.entry.form == .paged) return @as(usize, @intCast(self.p.table[t * self.stride + x / page])) * page + x % page;
        return t * sliding_ring + x % sliding_ring;
    }

    /// Every rounding point kept: Prep's reference at each row's own position and slot.
    pub fn solve(self: *const PrepRows, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        self.p.solveAt(s, out, self);
    }
};

/// ref.Gated(PrepRows) and the spec's baseline: tf_rc_prep's bytes on the same inputs (ref.Gated has no `replaced`).
pub const RowsGated = struct {
    in: ref.Gated(PrepRows),
    rig: *rig_.Rig, // in.rig: the hooks read the outputs there

    const vt = g.hooks(RowsGated);

    pub fn kernel(self: *RowsGated) g.Kernel {
        return .{ .spec = self.in.k.spec(), .ctx = self, .vt = &vt, .control = self.in.k.control() };
    }

    pub fn prepare(self: *RowsGated, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        return self.in.prepare(shape, fill, seed);
    }

    pub fn run(self: *RowsGated) !void {
        return self.in.run();
    }

    pub fn count(self: *RowsGated) usize {
        return self.in.count();
    }

    pub fn reference(self: *RowsGated, want: []f64, mag: []f64) !void {
        return self.in.reference(want, mag);
    }

    pub fn bounds(self: *RowsGated, want: []const f64, mag: []const f64, allow: []f64) !void {
        return self.in.bounds(want, mag, allow);
    }

    pub fn intervals(self: *RowsGated, low: []f64, high: []f64) !void {
        return self.in.intervals(low, high);
    }

    pub fn simulate(self: *RowsGated, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.in.simulate(acc, order, out);
    }

    pub fn replaced(self: *RowsGated) ![]const u8 {
        return self.in.k.baseline(self.rig);
    }
};

/// The registrations, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: rp.Kernels,
    ctl: cuda.Module,
    preps: [entries.len]ref.Gated(Prep) = undefined,
    rows: [rows_entries.len]RowsGated = undefined,

    /// The set, its kernels appended to `list` (the controls after the kernel they break).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try rp.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        const f = try ctl.function("tf_ctl_rc_prep");
        const fr = try ctl.function("tf_ctl_rc_prep_rows");
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        for (&set.preps, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = .{ .kern = &set.k, .entry = e, .ctl = if (e.keep != null) f else null } };
        for (&set.rows, rows_entries) |*x, e| {
            x.* = .{ .in = .{ .rig = .{ .gpu = gpu, .s = s }, .k = .{ .kern = &set.k, .entry = e, .ctl = if (e.keep != null) fr else null, .p = .{ .kern = &set.k, .entry = .{ .name = e.name, .form = e.form } } } }, .rig = undefined };
            x.rig = &x.in.rig;
        }
        for (&set.preps) |*x| try list.append(a, x.kernel());
        for (&set.rows) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.rows) |*x| x.in.deinit();
        for (&set.preps) |*x| x.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

test "each control keeps all of tf_rc_prep's rounding points but one, and each point has its control" {
    var dropped: i32 = 0;
    for (entries) |e| if (e.keep) |k| {
        const gone = all & ~k;
        try std.testing.expectEqual(@as(u32, 1), @popCount(gone));
        try std.testing.expect((dropped & gone) == 0);
        try std.testing.expect(e.form == .sliding);
        dropped |= gone;
    };
    try std.testing.expectEqual(all, dropped);
}
