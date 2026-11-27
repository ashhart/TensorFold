//! ERNIE 5.5's HCA attention under the contract: hca_attn.cu's chunk tiles, then the mla image's sink merge (hca.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_hca_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const hh = ernie.hca;

const hd = hh.head_dim;
/// The vendor's bmm1_scale, 512^-0.5, as fp32.
const scale: f32 = @floatCast(1 / @sqrt(@as(f64, hd)));
const u = 0x1p-24;

/// A planted fault (the launch's arguments): the sinks left out or times the scale, or the compressed rows not counted.
pub const Fault = enum { none, no_sink, scaled_sink, no_compressed };

pub const Entry = struct { name: []const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 hca_attn tf_ernie55_hca_chunks and mla tf_mla_merge_sink (one 512-wide key a row, window and compressed rows, fp32 sinks)" },
    .{ .name = "control: ernie5_5 hca attention with every head's sink left out", .fault = .no_sink },
    .{ .name = "control: ernie5_5 hca attention with each sink times the softmax scale", .fault = .scaled_sink },
    .{ .name = "control: ernie5_5 hca attention over the window's keys alone", .fault = .no_compressed },
};

/// rows, first position, heads: the first compressed row, 1-4 keys, a block's end, 513 keys, the 1M end, a prompt.
pub const served = &[_]c.Shape{ .{ .rows = 3, .n = 126, .k = 64 }, .{ .rows = 4, .n = 0, .k = 64 }, .{ .rows = 2, .n = 128 * 5 + 127, .k = 64 }, .{ .rows = 2, .n = 49278, .k = 32 }, .{ .rows = 2, .n = (1 << 20) - 2, .k = 64 }, .{ .rows = 64, .n = 1000, .k = 64 } };
/// The controls' shapes, in gate 3's order: 1-4 keys, where a sink weighs most, then 513 keys, 385 compressed rows.
const control_shapes = &[_]c.Shape{ served[1], served[3] };

/// A key's 512 products, a chunk's 512 keys, its 16 tiles' rescales, the merge's chunks and the sink.
fn depth(s: c.Shape) usize {
    return hd + hh.chunk + 16 + nchOf(s) + 1;
}

fn nchOf(s: c.Shape) usize {
    return (hh.keys(s.n + s.rows - 1) + hh.chunk - 1) / hh.chunk;
}

pub fn spec(e: Entry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .served = if (e.fault == .none) served else control_shapes };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const Attn = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    mla: *const cuda.mla.Mla,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    heads: usize = 0,
    w: usize = 0, // a row's list width: the last row's keys
    nch: usize = 0,
    comp: usize = 0, // compressed rows in the store, before the window's
    lo: usize = 0, // the first window position in the store
    q: []u16 = &.{}, // [rows][heads][512]
    store: []u16 = &.{}, // compressed rows 0 .. comp - 1, then positions lo .. the last row's
    tok: []i32 = &.{}, // [rows][w]: the window's rows by position, then the compressed rows
    sinks: []f32 = &.{},
    tol: []f64 = &.{},
    dev: [5]u64 = @splat(0), // q, store, tok, cnt, sinks
    parts: cuda.mla.Partials = .{ .po = 0, .pm = 0, .pl = 0 },
    out: u64 = 0,

    const vt = g.hooks(Attn);

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Attn) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.store);
        a.free(self.tok);
        a.free(self.sinks);
        a.free(self.tol);
        self.q = &.{};
        self.store = &.{};
        self.tok = &.{};
        self.sinks = &.{};
        self.tol = &.{};
    }

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
    }

    /// q, window keys [-1, 1), compressed keys 1 higher, sinks [-1, 3); stress: q 0, keys 1/16 but 2^15, 3/16, -2^15.
    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k > 64 or shape.n + shape.rows > 1 << 20) return false;
        const last = shape.n + shape.rows - 1;
        if (fill == .stress and shape.n < hh.windowStart(last) + 2) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.fill = fill;
        self.heads = shape.k;
        self.w = @intCast(hh.keys(last));
        self.nch = (self.w + hh.chunk - 1) / hh.chunk;
        self.comp = @intCast(hh.compressedRows(last));
        self.lo = @intCast(hh.windowStart(shape.n));
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.q = try a.alloc(u16, shape.rows * self.heads * hd);
        self.store = try a.alloc(u16, (self.comp + last + 1 - self.lo) * hd);
        self.tok = try a.alloc(i32, shape.rows * self.w);
        self.sinks = try a.alloc(f32, self.heads);
        if (fill == .stress) {
            @memset(self.q, 0);
            @memset(self.sinks, -std.math.inf(f32));
            @memset(self.store, c.toBf16(1.0 / 16.0));
            const first = hh.windowStart(last); // positions first .. shape.n are in every row's window
            for ([_]struct { usize, f32 }{ .{ first, 32768 }, .{ first + 1, 3.0 / 16.0 }, .{ shape.n, -32768 } }) |x|
                @memset(self.store[(self.comp + x[0] - self.lo) * hd ..][0..hd], c.toBf16(x[1]));
        } else {
            for (self.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            for (self.store, 0..) |*v, i| v.* = c.toBf16(rng.float(f32) * 2 - 1 + @as(f32, if (i / hd < self.comp) 1 else 0));
            for (self.sinks) |*v| v.* = rng.float(f32) * 4 - 1;
        }
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0
            const at = (self.comp + last - self.lo) * hd; // the last position's row: the last query row's own key
            self.store[at + 9] = c.toBf16(std.math.inf(f32));
            if (self.comp > 0) self.store[300] = c.toBf16(-std.math.inf(f32));
        }
        const cnt = try a.alloc(i32, shape.rows);
        defer a.free(cnt);
        @memset(self.tok, 0);
        for (cnt, 0..) |*n, r| {
            const p = shape.n + r;
            const list = self.tok[r * self.w ..][0..hh.keys(p)];
            const win = p + 1 - hh.windowStart(p);
            for (list[0..win], 0..) |*t, i| t.* = @intCast(self.comp + hh.windowStart(p) + i - self.lo);
            for (list[win..], 0..) |*t, j| t.* = @intCast(j);
            n.* = @intCast(if (self.entry.fault == .no_compressed) win else list.len);
        }
        const planted = try a.dupe(f32, self.sinks);
        defer a.free(planted);
        for (planted) |*v| v.* = switch (self.entry.fault) {
            .no_sink => -std.math.inf(f32),
            .scaled_sink => v.* * scale,
            else => v.*,
        };
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.q)), try self.rig.input(std.mem.sliceAsBytes(self.store)), try self.rig.input(std.mem.sliceAsBytes(self.tok)), try self.rig.input(std.mem.sliceAsBytes(cnt)), try self.rig.input(std.mem.sliceAsBytes(planted)) };
        const n = self.nch * shape.rows * self.heads;
        self.parts = .{ .po = try self.rig.scratch(n * hd * 4), .pm = try self.rig.scratch(n * 4), .pl = try self.rig.scratch(n * 4) };
        self.out = try self.rig.output(shape.rows * self.heads * hd * 2);
        return true;
    }

    pub fn run(self: *Attn) !void {
        try ernie.attendWith(self.f, self.mla, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.parts, self.out, .{ .rows = self.shape.rows, .heads = self.heads, .w = self.w, .nch = self.nch }, scale);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * self.heads * hd;
    }

    /// Row r's keys as f64, in the list's order.
    fn gather(self: *Attn, r: usize, kv: []f64) []f64 {
        const p = self.shape.n + r;
        const n = hh.keys(p);
        for (self.tok[r * self.w ..][0..n], 0..) |t, j| for (0..hd) |d| {
            kv[j * hd + d] = bf(self.store[@as(usize, @intCast(t)) * hd + d]);
        };
        return kv[0 .. n * hd];
    }

    fn query(self: *Attn, r: usize, h: usize) [hd]f64 {
        var q: [hd]f64 = undefined;
        for (&q, self.q[(r * self.heads + h) * hd ..][0..hd]) |*x, v| x.* = bf(v);
        return q;
    }

    /// hca.attend in float64; tol: bf16 p and out, the scores' (512 + 8) 2^-24 scale sum |q k|, the sums' chains.
    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        const kv = try a.alloc(f64, self.w * hd);
        defer a.free(kv);
        const t = try a.alloc(f64, 4 * self.w + 1 + hd);
        defer a.free(t);
        const chain: f64 = @floatFromInt(hh.chunk + 16 + self.nch + 4);
        const sc: f64 = scale;
        for (0..self.shape.rows) |r| {
            const keys = self.gather(r, kv);
            const n = keys.len / hd;
            for (0..self.heads) |h| {
                const at = (r * self.heads + h) * hd;
                const q = self.query(r, h);
                const sink: f64 = self.sinks[h];
                hh.attend(&q, keys, sink, scale, hh.exact, t, want[at..][0..hd]);
                const e = t[n..][0 .. n + 1];
                const ds = t[2 * n + 1 ..][0..n];
                for (ds, 0..) |*x, j| {
                    var abs: f64 = 0;
                    for (q, keys[j * hd ..][0..hd]) |qv, kv_| abs += @abs(qv * kv_);
                    x.* = abs * sc * (hd + 8) * u + chain * u;
                }
                var den = e[0];
                var dl = e[0] * 4 * u;
                for (e[1..], ds) |w, dv| {
                    den += w;
                    dl += w * dv;
                }
                for (want[at..][0..hd], self.tol[at..][0..hd], 0..) |o, *tl, d| {
                    var spread: f64 = 0;
                    for (e[1..], ds, 0..) |w, dv, j| spread += w * @abs(keys[j * hd + d]) * (0x1p-8 + dv);
                    tl.* = @abs(o) * (0x1p-8 + dl / den + 4 * u) + spread / den + 1e-30;
                }
            }
        }
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    /// Stress: the spec's exact-sum bound; otherwise the derived one.
    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.fill == .stress) return;
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    /// hca.attend with the scores', the weights' and the weighted keys' sums in `acc`, in `order`.
    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const sim = struct {
            acc: c.Accum,
            order: g.Order,
            pub fn of(s: @This(), t: []const f64) f64 {
                return rows_.sumIn(s.acc, s.order, t);
            }
        }{ .acc = acc, .order = order };
        const kv = try a.alloc(f64, self.w * hd);
        defer a.free(kv);
        const t = try a.alloc(f64, 2 * self.w + 1 + @max(self.w, hd));
        defer a.free(t);
        for (0..self.shape.rows) |r| {
            const keys = self.gather(r, kv);
            for (0..self.heads) |h| {
                const at = (r * self.heads + h) * hd;
                const q = self.query(r, h);
                hh.attend(&q, keys, self.sinks[h], scale, sim, t, out[at..][0..hd]);
            }
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    attn: ernie.Attn,
    mla: cuda.mla.Mla,
    items: [entries.len]Attn = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var x = try ernie.Attn.load(gpu.d);
        errdefer x.unload();
        var m = try cuda.mla.Mla.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .attn = x, .mla = m };
        for (&set.items, entries) |*it, e| it.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = set.attn.chunks_fn, .mla = &set.mla };
        for (&set.items) |*it| try list.append(a, it.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*it| it.deinit();
        set.mla.unload();
        set.attn.unload();
        set.gpa.destroy(set);
    }
};

/// The HCA attention's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x61747464, &t);
    std.debug.print("SUMMARY ernie5_5 hca_attn: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-hca-attn contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the launch: 512 then 513 keys, 8,320 in 17 chunks at the 1M end, few keys at the start" {
    for (served) |x| try hh.attnCheck(.{ .rows = x.rows, .heads = x.k, .w = hh.keys(x.n + x.rows - 1), .nch = nchOf(x) });
    try std.testing.expectEqual(@as(u64, 512), hh.keys(served[3].n));
    try std.testing.expectEqual(@as(u64, 513), hh.keys(served[3].n + 1));
    try std.testing.expectEqual(@as(usize, 17), nchOf(served[4]));
    try std.testing.expectEqual(@as(u64, 1), hh.keys(served[1].n));
    // the controls' shapes: rows of a few keys (the sinks'), then rows mostly of compressed keys (the third control's)
    const few = control_shapes[0];
    const comp = control_shapes[1];
    try std.testing.expect(hh.keys(few.n + few.rows - 1) == 4 and hh.compressedRows(few.n + few.rows - 1) == 0);
    try std.testing.expect(hh.compressedRows(comp.n) > 2 * hh.window);
}

test "the attention's tiles take wide_smem<16>() (83,328 bytes, the record's), within the 99 KiB a GB10 block may take" {
    try std.testing.expectEqual(@as(u32, 83328), ernie.attn_smem);
    try std.testing.expect(ernie.attn_smem <= 99 * 1024);
}

test "gate 4's shape (the first with the most heads, 3 rows) has three positions every row's window holds" {
    const x = served[0];
    try std.testing.expect(x.n >= hh.windowStart(x.n + @min(x.rows, 4) - 1) + 2);
}
