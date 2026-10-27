//! tf_q36m_attn_rows under the contract: several streams' rows over their own rings, each row tf_q36m_attn's bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const heads = 16;
const kv_heads = 2;
const hd = 256;
const width = kv_heads * hd; // a ring slot's or a block row's keys (or values)
const gstride = heads * 2 * hd; // a row's q and gate halves
const nan_bits: u16 = 0x7fc0;
const rule = "as tf_q36m_attn gives them, launched once a segment over the same rows";

/// The packed row's ring, as the kernel's RingRow (40 bytes).
const RingRow = extern struct { rk: u64, rv: u64, begin: i64, end: i64, seg: i32, w: i32 };

/// One stream's part of a packing: its ring's capacity, first slot and slots, and its rows' packed range.
const Seg = struct { cap: usize, begin: usize, slots: usize, first: usize, rows: usize };

/// Each row's terms run over its ring's slots, then its segment's rows up to its own: two roundings a term.
fn depth(s: c.Shape) usize {
    return 2 * (s.k + s.rows) + 16;
}

pub fn spec() c.Spec {
    return .{
        .name = "cuda qwen3.6 mtp attention over rings tf_q36m_attn_rows",
        .accum = .f32,
        .depth = depth,
        .ops = 2, // o / l, then the gate's product
        .mids = 1, // o / l rounded to bf16 before the gate
        .out = .bf16,
        .replaces = "tf_q36m_attn",
        .nan = .{ .stated = rule },
        // the first: three rings, so the single-shape gates (repeats, guards, NaN) run across streams
        .served = &.{ .{ .rows = 16, .n = heads * hd, .k = 6 }, .{ .rows = 1, .n = heads * hd, .k = 6 }, .{ .rows = 4, .n = heads * hd, .k = 6 }, .{ .rows = 4, .n = heads * hd, .k = 4200 }, .{ .rows = 16, .n = heads * hd, .k = 4200 } },
    };
}

/// Three streams at most: rows 1, up to 3, then the rest, each over its own ring (one of them wrapping).
fn pack(rows: usize, k: usize, out: *[3]Seg) []Seg {
    const sizes = [3]usize{ 1, @min(3, rows -| 1), rows -| 4 };
    const caps = [3]usize{ k + 1, 2 * k + 64, 32768 };
    const begins = [3]usize{ 0, 5, 32768 - 3 };
    const slots = [3]usize{ k, k / 2 + 1, k -| 1 };
    var n: usize = 0;
    var first: usize = 0;
    for (sizes, caps, begins, slots) |size, cap, begin, slot| if (size > 0) {
        out[n] = .{ .cap = cap, .begin = begin, .slots = slot, .first = first, .rows = size };
        first += size;
        n += 1;
    };
    return out[0..n];
}

pub const Rows = struct {
    rig: rig_.Rig,
    attn: cuda.Function,
    rows_fn: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    segs_buf: [3]Seg = undefined,
    segs: []Seg = &.{},
    q: []u16 = &.{},
    gate: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    rk: [3][]u16 = .{ &.{}, &.{}, &.{} },
    rv: [3][]u16 = .{ &.{}, &.{}, &.{} },
    dq: u64 = 0,
    dg: u64 = 0,
    dkb: u64 = 0,
    dvb: u64 = 0,
    drk: [3]u64 = @splat(0),
    drv: [3]u64 = @splat(0),
    table: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn init(gpu: tc.Gpu, s: cuda.Stream, mod: cuda.Module) !Rows {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .attn = try mod.function("tf_q36m_attn"), .rows_fn = try mod.function("tf_q36m_attn_rows"), .arena = .init(gpu.gpa) };
    }

    pub fn deinit(self: *Rows) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Rows) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Rows);

    fn segOf(self: *const Rows, row: usize) usize {
        for (self.segs, 0..) |s, i| if (row < s.first + s.rows) return i;
        unreachable;
    }

    /// Row `m`'s terms in the kernel's order: its ring's slots, then its segment's rows up to its own.
    fn terms(self: *const Rows, m: usize) usize {
        const s = self.segs[self.segOf(m)];
        return s.slots + (m - s.first) + 1;
    }

    /// Key or value `j` of row `m`'s terms for kv head `kvh`.
    fn term(self: *const Rows, m: usize, j: usize, kvh: usize, values: bool) []const u16 {
        const si = self.segOf(m);
        const s = self.segs[si];
        if (j < s.slots) {
            const at = ((s.begin + j) % s.cap) * width + kvh * hd;
            return (if (values) self.rv[si] else self.rk[si])[at..][0..hd];
        }
        const r = s.first + (j - s.slots);
        return (if (values) self.vb else self.kb)[r * width + kvh * hd ..][0..hd];
    }

    fn value(self: *const Rows, rng: std.Random, j: usize) u16 {
        return switch (self.fill) {
            .stress => c.toBf16(([3]f32{ 4096, 1, -4096 })[j % 3]),
            else => c.toBf16((rng.float(f32) - 0.5) * 0.5),
        };
    }

    pub fn prepare(self: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n != heads * hd or shape.k < 2) return false;
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        const a = self.arena.allocator();
        self.shape = shape;
        self.fill = fill;
        self.segs = pack(shape.rows, shape.k, &self.segs_buf);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.q = try a.alloc(u16, rows * heads * hd);
        self.gate = try a.alloc(u16, rows * gstride);
        self.kb = try a.alloc(u16, rows * width);
        self.vb = try a.alloc(u16, rows * width);
        for (self.q) |*x| x.* = if (fill == .stress) 0 else c.toBf16((rng.float(f32) - 0.5) * 0.5);
        for (self.gate) |*x| x.* = if (fill == .stress) 0 else c.toBf16((rng.float(f32) - 0.5) * 4);
        for (self.segs, 0..) |s, si| {
            // every physical slot outside [begin, begin + slots) of the ring holds NaN: a slot it should not read
            self.rk[si] = try a.alloc(u16, s.cap * width);
            self.rv[si] = try a.alloc(u16, s.cap * width);
            @memset(self.rk[si], nan_bits);
            @memset(self.rv[si], nan_bits);
            for (0..s.slots) |j| {
                const at = ((s.begin + j) % s.cap) * width;
                for (0..width) |i| {
                    self.rk[si][at + i] = if (fill == .stress) 0 else c.toBf16((rng.float(f32) - 0.5) * 0.5);
                    self.rv[si][at + i] = self.value(rng, j);
                }
            }
            for (0..s.rows) |r| for (0..width) |i| {
                self.kb[(s.first + r) * width + i] = if (fill == .stress) 0 else c.toBf16((rng.float(f32) - 0.5) * 0.5);
                self.vb[(s.first + r) * width + i] = self.value(rng, s.slots + r);
            };
        }
        if (fill == .nonfinite) {
            self.q[1 * hd + 3] = nan_bits; // row 0, head 1
            const last = self.segs[self.segs.len - 1];
            const at = (last.begin % last.cap) * width + 7;
            self.rk[self.segs.len - 1][at] = c.toBf16(std.math.inf(f32)); // the last stream's first slot
            self.vb[(rows - 1) * width + 9] = c.toBf16(-std.math.inf(f32));
        }
        self.dq = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dg = try self.rig.input(std.mem.sliceAsBytes(self.gate));
        self.dkb = try self.rig.input(std.mem.sliceAsBytes(self.kb));
        self.dvb = try self.rig.input(std.mem.sliceAsBytes(self.vb));
        const tab = try a.alloc(RingRow, rows);
        for (self.segs, 0..) |s, si| {
            self.drk[si] = try self.rig.input(std.mem.sliceAsBytes(self.rk[si]));
            self.drv[si] = try self.rig.input(std.mem.sliceAsBytes(self.rv[si]));
            for (0..s.rows) |r| tab[s.first + r] = .{ .rk = self.drk[si], .rv = self.drv[si], .begin = @intCast(s.begin), .end = @intCast(s.begin + s.slots), .seg = @intCast(s.first), .w = @intCast(s.cap) };
        }
        self.table = try self.rig.input(std.mem.sliceAsBytes(tab));
        self.out = try self.rig.output(rows * heads * hd * 2);
        self.base = try self.rig.scratch(rows * heads * hd * 2);
        return true;
    }

    pub fn run(self: *Rows) !void {
        var a: cuda.Args = .{};
        inline for (.{ self.dq, self.dg, @as(c_int, gstride), self.dkb, self.dvb, self.table, @as(c_int, heads), @as(c_int, kv_heads), self.out }) |v| a.add(v);
        try cuda.launch.launch(self.rows_fn, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = heads }, .block = .{ .x = 32 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// tf_q36m_attn once a segment, each on its own ring and its own rows' offsets.
    pub fn replaced(self: *Rows) ![]const u8 {
        for (self.segs, 0..) |s, si| {
            const off = s.first;
            var a: cuda.Args = .{};
            inline for (.{ self.dq + off * heads * hd * 2, self.dg + off * gstride * 2, @as(c_int, gstride), self.dkb + off * width * 2, self.dvb + off * width * 2, self.drk[si], self.drv[si], @as(c_int, heads), @as(c_int, kv_heads) }) |v| a.add(v);
            a.add(@as(c_int, @intCast(s.cap)));
            a.add(@as(c_longlong, @intCast(s.begin)));
            a.add(@as(c_longlong, @intCast(s.begin + s.slots)));
            a.add(self.base + off * heads * hd * 2);
            try cuda.launch.launch(self.attn, .{ .grid = .{ .x = @intCast(s.rows), .y = heads }, .block = .{ .x = 32 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.shape.rows * heads * hd * 2);
    }

    /// The stated NaN rule: the outputs equal tf_q36m_attn's bytes on the same planted inputs.
    pub fn check(self: *Rows, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const got = try self.arena.allocator().dupe(u8, try self.rig.outputBytes());
        const want = try self.replaced();
        return if (std.mem.eql(u8, got, want)) null else "its bytes differ from tf_q36m_attn's on the planted inputs";
    }

    pub fn count(self: *const Rows) usize {
        return self.shape.rows * heads * hd;
    }

    pub fn reference(self: *Rows, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        for (0..self.shape.rows) |m| for (0..heads) |h| {
            const kvh = h / (heads / kv_heads);
            const n = self.terms(m);
            const p = try a.alloc(f64, n);
            defer a.free(p);
            for (p, 0..) |*x, j| {
                var dot: f64 = 0;
                for (self.q[(m * heads + h) * hd ..][0..hd], self.term(m, j, kvh, false)) |qv, kv| dot += c.fromBf16(qv) * c.fromBf16(kv);
                x.* = dot / 16;
            }
            const top = std.mem.max(f64, p);
            var total: f64 = 0;
            for (p) |*x| {
                x.* = @exp(x.* - top);
                total += x.*;
            }
            for (0..hd) |d| {
                var sum: f64 = 0;
                var most: f64 = 0;
                for (p, 0..) |w, j| {
                    const v = c.fromBf16(self.term(m, j, kvh, true)[d]);
                    sum += w * v;
                    most = @max(most, @abs(v));
                }
                const gate = 1 / (1 + @exp(-c.fromBf16(self.gate[m * gstride + h * 2 * hd + hd + d])));
                const o = (m * heads + h) * hd + d;
                want[o] = sum / total * gate;
                mag[o] = most * gate;
            }
        };
    }

    /// Stress inputs: every weight is 1, so each output is its values' sum in `acc` and `order`, over their count.
    pub fn simulate(self: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        for (0..self.shape.rows) |m| {
            const n = self.terms(m);
            const t = try a.alloc(f64, n);
            defer a.free(t);
            for (0..heads) |h| for (0..hd) |d| {
                for (t, 0..) |*x, j| x.* = c.fromBf16(self.term(m, j, h / (heads / kv_heads), true)[d]);
                const s = contract.rows.sumIn(acc, order, t);
                const mean: f32 = @floatCast(s / @as(f64, @floatFromInt(n)));
                out[(m * heads + h) * hd + d] = c.fromBf16(c.toBf16(c.fromBf16(c.toBf16(mean)) * 0.5));
            };
        }
    }
};

/// The kernel and its module, registered in the contract's list; destroyed after the gates have run.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    rows: Rows,

    pub fn create(a: std.mem.Allocator, gpu: tc.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen36_mtp);
        errdefer self.mod.unload();
        self.rows = try Rows.init(gpu, s, self.mod);
        errdefer self.rows.deinit();
        try kernels.append(a, self.rows.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        self.rows.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};

/// Gates 1-4 on this kernel alone, then its FP64 check: the kernel-contract run's share, for a quick look.
pub fn alone(gpu: tc.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const group = try Group.create(gpu.gpa, gpu, s, &kernels);
    defer group.destroy();
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x6b63, &t);
    try fp64(gpu);
    try tc.expect(t.problems() == 0, "tf_q36m_attn_rows: {d} gate failures", .{t.failed});
}

/// tf_q36m_attn_rows against float64 at the served packings (gate 3 compares a replacement by its bytes only).
pub fn fp64(gpu: tc.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen36_mtp);
    defer mod.unload();
    var x = try Rows.init(gpu, s, mod);
    defer x.deinit();
    const sp = spec();
    for (sp.served, 0..) |shape, i| {
        try tc.expect(try x.prepare(shape, .random, 0x5eed + i), "tf_q36m_attn_rows refused a served shape", .{});
        try x.run();
        const n = x.count();
        const want = try gpu.gpa.alloc(f64, 2 * n);
        defer gpu.gpa.free(want);
        try x.reference(want[0..n], want[n..]);
        const out = try gpu.gpa.dupe(u8, try x.rig.outputBytes());
        defer gpu.gpa.free(out);
        for (0..n) |o| {
            const t: f64 = @floatFromInt(x.terms(o / (heads * hd)));
            const bound = (2 * c.unit(.bf16) + c.gamma(256 + 8 * t, c.unit(.f32))) * want[n + o] + 0x1p-133;
            const got = c.read(.bf16, out, o);
            try tc.expect(std.math.isFinite(got) and @abs(got - want[o]) <= bound, "tf_q36m_attn_rows FP64 mismatch at {d}x{d}, output {d}: {d} vs {d}", .{ shape.rows, shape.k, o, got, want[o] });
        }
        @memset(out[0..2], 0x40); // a planted finite corruption must leave the bound
        const bad = c.read(.bf16, out, 0);
        try tc.expect(@abs(bad - want[0]) > (2 * c.unit(.bf16) + c.gamma(256 + 8 * @as(f64, @floatFromInt(x.terms(0))), c.unit(.f32))) * want[n], "tf_q36m_attn_rows FP64 check missed a planted corruption", .{});
    }
    std.debug.print("PASS tf_q36m_attn_rows: heterogeneous rings (one wrapping), NaN outside each ring's slots, FP64 at {d} served shapes; corruption RED\n", .{sp.served.len});
}
