//! ERNIE 5.5's VHA (ernie5_5/hca.cu) under the contract: a token's 64 heads mixed, o + V (U^T o), against float64.
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

const nh = hh.vha_heads;
const nr = hh.vha_rank;

/// A planted fault (control entries): o left out of o + V (U^T o), or U and V read rank-major.
pub const Fault = enum { none, no_skip, transposed };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 hca tf_ernie55_vha (a token's 64 heads mixed at rank 16, o + V (U^T o) in fp32, one rounding)", .symbol = "tf_ernie55_vha" },
    .{ .name = "control: ernie5_5 vha without o, the identity of I + V U^T", .symbol = "tf_ctl_ernie55_vha_no_skip", .fault = .no_skip },
    .{ .name = "control: ernie5_5 vha reading U and V rank-major", .symbol = "tf_ctl_ernie55_vha_transposed", .fault = .transposed },
};

/// tokens, the width, the heads: a decode token, a few, a prompt chunk at HCA's 512, and DSA's 256.
pub const served = &[_]c.Shape{ .{ .rows = 1, .n = hh.head_dim, .k = nh }, .{ .rows = 4, .n = hh.head_dim, .k = nh }, .{ .rows = 64, .n = hh.head_dim, .k = nh }, .{ .rows = 4, .n = 256, .k = nh } };

/// z's 64 terms, then V z's 16 and o.
fn depth(_: c.Shape) usize {
    return nh + nr + 1;
}

pub fn spec(e: Entry) c.Spec {
    const tiles = [3]c.Tile{ .{}, .{ .size = ernie.threads }, .{} };
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .tiles = tiles, .served = if (e.fault == .none) served else served[0..1] };
}

pub const Vha = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    o: []u16 = &.{}, // [rows][64][d], bf16
    u: [nh * nr]f64 = undefined, // bf16 values, [64][16]
    v: [nh * nr]f64 = undefined,
    dev: [3]u64 = @splat(0), // o, U, V
    out: u64 = 0,

    const vt = g.hooks(Vha);

    pub fn kernel(self: *Vha) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    pub fn deinit(self: *Vha) void {
        self.rig.gpu.gpa.free(self.o);
        self.o = &.{};
        self.rig.deinit();
    }

    /// o [-1, 1), U and V [-1/4, 1/4); stress: U 1 and V 1/16, a dim's heads 2^20, 24, -2^20, then 1/4 (z = 39.25).
    pub fn prepare(self: *Vha, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != nh) return false;
        hh.vhaCheck(shape.rows, shape.n) catch return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.o);
        self.shape = shape;
        const d = shape.n;
        self.o = try a.alloc(u16, shape.rows * nh * d);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var u16s: [2][nh * nr]u16 = undefined;
        for (&u16s, [2]*[nh * nr]f64{ &self.u, &self.v }, [2]f32{ 1, 1.0 / 16.0 }) |*bits, vals, stress| for (bits, vals) |*b, *x| {
            b.* = c.toBf16(if (fill == .stress) stress else (rng.float(f32) * 2 - 1) / 4);
            x.* = c.fromBf16(b.*);
        };
        if (fill == .stress) {
            for (0..shape.rows) |r| for (0..nh) |i| {
                const x: f32 = switch (i) {
                    0 => 0x1p20,
                    1 => 24,
                    2 => -0x1p20,
                    else => 0.25,
                };
                @memset(self.o[(r * nh + i) * d ..][0..d], c.toBf16(x));
            };
        } else for (self.o) |*x| {
            x.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, f32 }{ .{ 3, std.math.nan(f32) }, .{ nh * d + 7, std.math.inf(f32) }, .{ self.o.len - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < self.o.len) {
                self.o[p[0]] = c.toBf16(p[1]);
            };
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.o)), try self.rig.input(std.mem.sliceAsBytes(&u16s[0])), try self.rig.input(std.mem.sliceAsBytes(&u16s[1])) };
        self.out = try self.rig.output(self.o.len * 2);
        return true;
    }

    pub fn run(self: *Vha) !void {
        try ernie.vhaWith(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.shape.rows, self.shape.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Vha) usize {
        return self.o.len;
    }

    /// Token r's heads at dim col, as f64.
    fn column(self: *Vha, r: usize, col: usize, x: *[nh]f64) void {
        const d = self.shape.n;
        for (x, 0..) |*v, i| v.* = c.fromBf16(self.o[(r * nh + i) * d + col]);
    }

    /// hca.vha in float64; mag: |o_h| plus V's terms' |V_hr| times z_r's terms' sum of |U_kr o_k|.
    pub fn reference(self: *Vha, want: []f64, mag: []f64) !void {
        const d = self.shape.n;
        var x: [nh]f64 = undefined;
        var t: [nh]f64 = undefined;
        var y: [nh]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..d) |col| {
            self.column(r, col, &x);
            hh.vha(&x, &self.u, &self.v, hh.exact, &t, &y);
            var zm: [nr]f64 = @splat(0);
            for (&zm, 0..) |*m, k| {
                for (x, 0..) |a, i| m.* += @abs(self.u[i * nr + k] * a);
            }
            for (y, x, 0..) |w, a, i| {
                var m: f64 = @abs(a);
                for (zm, 0..) |z, k| m += @abs(self.v[i * nr + k]) * z;
                want[(r * nh + i) * d + col] = w;
                mag[(r * nh + i) * d + col] = m;
            }
        };
    }

    /// hca.vha with z's and V z's sums in `acc`, in `order`.
    pub fn simulate(self: *Vha, acc: c.Accum, order: g.Order, out: []f64) !void {
        const sim = struct {
            acc: c.Accum,
            order: g.Order,
            pub fn of(s: @This(), t: []const f64) f64 {
                return rows_.sumIn(s.acc, s.order, t);
            }
        }{ .acc = acc, .order = order };
        const d = self.shape.n;
        var x: [nh]f64 = undefined;
        var t: [nh]f64 = undefined;
        var y: [nh]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..d) |col| {
            self.column(r, col, &x);
            hh.vha(&x, &self.u, &self.v, sim, &t, &y);
            for (y, 0..) |w, i| out[(r * nh + i) * d + col] = w;
        };
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    hca: ernie.Hca,
    items: [entries.len]Vha = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try ernie.Hca.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .hca = m };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.hca.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.hca.unload();
        set.gpa.destroy(set);
    }
};

/// VHA's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x766861, &t);
    std.debug.print("SUMMARY ernie5_5 vha: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-vha contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the launch, at HCA's 512 and DSA's 256 dims" {
    for (served) |x| try hh.vhaCheck(x.rows, x.n);
    try std.testing.expectEqual(@as(usize, hh.head_dim), served[0].n);
    try std.testing.expectEqual(@as(usize, 256), served[served.len - 1].n);
}

test "gate 4's column: z exact, fp32 within the output's rounding in every order, bf16's lose the 24 (over 100x), fp16's overflow" {
    var x: [nh]f64 = undefined;
    for (&x, 0..) |*v, i| v.* = switch (i) {
        0 => 0x1p20,
        1 => 24,
        2 => -0x1p20,
        else => 0.25,
    };
    const u: [nh * nr]f64 = @splat(1);
    const v: [nh * nr]f64 = @splat(1.0 / 16.0);
    var t: [nh]f64 = undefined;
    var want: [nh]f64 = undefined;
    var got: [nh]f64 = undefined;
    hh.vha(&x, &u, &v, hh.exact, &t, &want);
    try std.testing.expectEqual(@as(f64, 39.5), want[3]);
    const s = spec(entries[0]);
    inline for (.{ c.Accum.f32, c.Accum.bf16, c.Accum.f16 }) |acc| for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        const sim = struct {
            order: g.Order,
            pub fn of(q: @This(), terms: []const f64) f64 {
                return rows_.sumIn(acc, q.order, terms);
            }
        }{ .order = order };
        hh.vha(&x, &u, &v, sim, &t, &got);
        for (got, want) |a, w| {
            const miss = @abs(a - w) / c.boundExact(s, w);
            switch (acc) {
                .f32 => try std.testing.expect(miss <= 1),
                else => {},
            }
        }
        const miss3 = if (std.math.isFinite(got[3])) @abs(got[3] - want[3]) / c.boundExact(s, want[3]) else std.math.inf(f64);
        if (acc != .f32) try std.testing.expect(miss3 >= 100);
    };
}
