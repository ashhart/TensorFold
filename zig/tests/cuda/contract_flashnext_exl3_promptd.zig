//! Flash Next EXL3's prompt tile in decode's bytes (tf_exl3_promptd_0, _1, _2) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const exl3 = cuda.exl3;
const fmt = core.exl3_format;

/// A weight's launch: its own `plan()`, or the (K splits, warps) a family fixes (`Weight.sk` and `.wk`).
pub const Plan = enum { own, fixed_1_4, fixed_2_8 };
/// A planted fault: another K split and warp count than the weight's, or x with its non-finite values zeroed.
pub const Fault = enum { none, split, sanitized };

pub const Reg = struct { book: exl3.Codebook = .mul1, bits: u32 = 4, plan: Plan = .own, fault: Fault = .none };

const max_rows = 2048; // the prompt chunk (256 rows) eight times over
const max_k = 4096;

const why = "the K sums of decoded fp16 weights: no input makes them exact in fp32, so gate 4 proves fp16's miss (an overflow) alone";
const rule = "linear_body's bytes on the same inputs: a NaN and both infinities in x reach the outputs as linear_body's do";

/// A K sum in any order (gate 4 simulates a sequential one), then the chains', splits' and output rotation's adds.
fn depth(s: c.Shape) usize {
    return s.k + 64;
}

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

// Rows past the 64-row block and the 256-row chunk; plan()'s forms: no split (K 512), a split, 8 warps (n 8192)
const own_served = [_]c.Shape{ sh(65, 1024, 512), sh(257, 1024, 4096), sh(256, 8192, 1024), sh(130, 8192, 512), sh(2048, 1024, 4096) };
// A family's fixed plan on shapes whose own plan splits K differently.
const fixed_served = [_]c.Shape{ sh(65, 1024, 4096), sh(257, 2048, 2048), sh(129, 1024, 1024) };

fn served(plan: Plan) []const c.Shape {
    const own: []const c.Shape = &own_served;
    const fixed: []const c.Shape = &fixed_served;
    return if (plan == .own) own else fixed;
}

/// The weight a registration launches on a shape (device pointers filled by the case).
fn weightOf(r: Reg, shape: c.Shape) exl3.Weight {
    const halves: u32 = 2 * r.bits;
    const st = exl3.strides(shape.k, shape.n, halves, false);
    var w: exl3.Weight = .{ .trellis = 0, .suh = 0, .svh = 0, .k = @intCast(shape.k), .n = @intCast(shape.n), .halves = halves, .codebook = r.book, .stride_k = st.k, .stride_nb = st.nb };
    switch (r.plan) {
        .own => {},
        .fixed_1_4 => {
            w.sk = 1;
            w.wk = 4;
        },
        .fixed_2_8 => {
            w.sk = 2;
            w.wk = 8;
        },
    }
    return w;
}

/// Another valid (K splits, warps) of the same shape: the planted fault.
fn other(w: exl3.Weight) exl3.Weight {
    const p = exl3.planOf(w);
    var x = w;
    x.wk = if (p.wk == 4) 8 else 4;
    x.sk = if (p.wk == 4) @max(1, p.sk / 2) else p.sk * 2;
    return x;
}

fn divides(w: exl3.Weight) bool {
    const p = exl3.planOf(w);
    return (w.k / 16) % (p.sk * p.wk) == 0;
}

/// Whether the launch takes the shape: whole 128-blocks, rows and K within the scratch, a plan K's tiles divide by.
fn takes(r: Reg, shape: c.Shape) bool {
    if (shape.rows == 0 or shape.rows > max_rows or shape.k == 0 or shape.n == 0) return false;
    if (shape.k % 128 != 0 or shape.n % 128 != 0 or shape.k > max_k) return false;
    const w = weightOf(r, shape);
    return divides(w) and (r.fault != .split or divides(other(w)));
}

fn kernelName(buf: []u8, r: Reg) []const u8 {
    const plan = switch (r.plan) {
        .own => "",
        .fixed_1_4 => ", fixed plan 1 split of 4 warps",
        .fixed_2_8 => ", fixed plan 2 splits of 8 warps",
    };
    const lead = switch (r.fault) {
        .none => "cuda Flash Next EXL3 prompt tile",
        .split => "control: Flash Next EXL3 prompt tile at another K split posing as",
        .sanitized => "control: Flash Next EXL3 prompt tile with x's non-finite values zeroed posing as",
    };
    return std.fmt.bufPrint(buf, "{s} tf_exl3_promptd_{d} ({s}, {d}-bit{s}, linear_body's bytes)", .{ lead, @backingInt(r.book), @tagName(r.book), r.bits, plan }) catch unreachable;
}

/// The served shapes: every row edge of a 64-row block, the 128-column and 128-input blocks, and the largest shapes.
pub fn spec(r: Reg, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = .bf16,
        .replaces = "tf_exl3_linear (linear_body)",
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 64 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = served(r.plan),
        .sums = .{ .f16 = 4, .bf16 = 0, .exact = false, .why = why },
    };
}

/// core/exl3_format.zig's name for a codebook of the device's numbering.
fn formatBook(b: exl3.Codebook) fmt.Codebook {
    return switch (b) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    };
}

fn random16(rng: std.Random, out: []u16) void {
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
}

pub const Case = struct {
    rig: rig_.Rig,
    de: *exl3.Dense,
    reg: Reg,
    name_buf: [200]u8 = undefined,
    shape: c.Shape = undefined,
    w: exl3.Weight = undefined,
    trellis: []i16 = &.{},
    suh: []u16 = &.{},
    svh: []u16 = &.{},
    x: []u16 = &.{},
    xd: u64 = 0,
    xz: u64 = 0, // x with its non-finite values zeroed: what the sanitized control launches on
    out: u64 = 0,
    base: u64 = 0,
    xh: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Case) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Case) void {
        const a = self.rig.gpu.gpa;
        a.free(self.trellis);
        a.free(self.suh);
        a.free(self.svh);
        a.free(self.x);
        self.trellis = &.{};
        self.suh = &.{};
        self.svh = &.{};
        self.x = &.{};
    }

    pub fn kernel(self: *Case) g.Kernel {
        const s = spec(self.reg, kernelName(&self.name_buf, self.reg));
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = switch (self.reg.fault) {
            .none => null,
            .split => .shapes,
            .sanitized => .nonfinite,
        } };
    }

    const vt = g.hooks(Case);

    /// Random trellis words, scales in [-1, 1) and bf16 x; non-finite: x[1][3] NaN, x[2][5] +Inf, x[2][9] -Inf.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!takes(self.reg, shape)) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x70726d64);
        const rng = prng.random();
        const width = fmt.Width{ .halves = @intCast(2 * self.reg.bits) };
        self.trellis = try a.alloc(i16, (shape.k / 16) * (shape.n / 16) * width.tileWords());
        self.suh = try a.alloc(u16, shape.k);
        self.svh = try a.alloc(u16, shape.n);
        if (fill == .stress) { // one weight at every width: all-ones states, scales 1
            @memset(self.trellis, -1);
            @memset(self.suh, 0x3c00);
            @memset(self.svh, 0x3c00);
        } else {
            for (self.trellis) |*v| v.* = rng.int(i16);
            random16(rng, self.suh);
            random16(rng, self.svh);
        }
        self.x = try a.alloc(u16, shape.rows * shape.k);
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        switch (fill) {
            .random => {},
            .nonfinite => if (shape.rows > 2) {
                self.x[shape.k + 3] = 0x7fc0;
                self.x[2 * shape.k + 5] = 0x7f80;
                self.x[2 * shape.k + 9] = 0xff80;
            },
            .stress => { // one input a block, rotated to 4096: K equal terms, past fp16's range
                @memset(self.x, 0);
                for (0..shape.rows) |r| for (0..shape.k / 128) |b| {
                    self.x[r * shape.k + b * 128] = c.toBf16(46336);
                };
            },
        }
        self.w = weightOf(self.reg, shape);
        self.w.trellis = try self.rig.input(std.mem.sliceAsBytes(self.trellis));
        self.w.suh = try self.rig.input(std.mem.sliceAsBytes(self.suh));
        self.w.svh = try self.rig.input(std.mem.sliceAsBytes(self.svh));
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.xz = self.xd;
        if (self.reg.fault == .sanitized) {
            const z = try a.dupe(u16, self.x);
            defer a.free(z);
            for (z) |*v| if (v.* & 0x7f80 == 0x7f80) {
                v.* = 0;
            };
            self.xz = try self.rig.input(std.mem.sliceAsBytes(z));
        }
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        self.base = try self.rig.scratch(shape.rows * shape.n * 2);
        self.xh = try self.rig.scratch(shape.rows * shape.k * 2);
        return true;
    }

    pub fn run(self: *Case) !void {
        const w = if (self.reg.fault == .split) other(self.w) else self.w;
        try self.de.promptExact(self.rig.s, w, self.xz, self.shape.rows, self.out, .bf16);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n;
    }

    /// linear_body on the same inputs: the decode kernel's launch in 128-row blocks, never the mid-M kernels.
    pub fn replaced(self: *Case) ![]const u8 {
        self.de.lin.midm = false;
        defer self.de.lin.midm = true;
        try self.de.decode(self.rig.s, self.w, self.xd, .bf16, self.shape.rows, self.base, .bf16);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    /// Gate 1: linear_body's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from linear_body's", .{ at, got.len });
        for (0..self.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    /// x * suh rotated per 128 by linear_body's own rot_in: the fp16 rows both kernels take, as float64.
    fn rotated(self: *Case, xs: []f64) !void {
        var args: cuda.Args = .{};
        args.add(self.xd);
        args.add(self.w.suh);
        args.add(self.xh);
        args.add(@as(i32, @backingInt(exl3.DType.bf16)));
        args.add(@as(i32, @intCast(self.shape.rows)));
        args.add(@as(i32, @intCast(self.shape.k)));
        try cuda.launch.launch(self.de.lin.rot_in, .{ .grid = .{ .x = @intCast((self.shape.k / 128 + 3) / 4), .y = @intCast(self.shape.rows), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, self.rig.s, &args);
        const raw = try self.rig.bytes(self.xh, xs.len * 2);
        for (xs, 0..) |*v, q| v.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[2 * q ..][0..2], .little))));
    }

    /// Float64 outputs and their terms' magnitude; with `acc`, the K sums simulated in that accumulator and order.
    fn outputs(self: *Case, want: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.gpu.gpa;
        const k = self.shape.k;
        const n = self.shape.n;
        const rows = self.shape.rows;
        const xs = try a.alloc(f64, rows * k);
        defer a.free(xs);
        try self.rotated(xs);
        const wq = try a.alloc(f16, k * n);
        defer a.free(wq);
        fmt.unpack(.{ .halves = @intCast(self.w.halves) }, formatBook(self.reg.book), self.trellis, k / 16, n / 16, wq);
        const s1 = try a.alloc(f64, rows * n);
        defer a.free(s1);
        const t = try a.alloc(f64, k);
        defer a.free(t);
        for (0..rows) |r| for (0..n) |j| {
            var abs: f64 = 0;
            for (0..k) |q| {
                t[q] = xs[r * k + q] * @as(f64, @floatCast(wq[q * n + j]));
                abs += @abs(t[q]);
            }
            var sum: f64 = 0;
            if (acc) |ac| {
                sum = contract.rows.sumIn(ac, order, t);
            } else for (t) |v| {
                sum += v;
            }
            want[r * n + j] = sum;
            s1[r * n + j] = abs;
        };
        fmt.rotate(want, rows, n, 1);
        for (0..rows) |r| for (0..n) |j| {
            const sc: f64 = @floatCast(@as(f16, @bitCast(self.svh[j])));
            want[r * n + j] *= sc;
            if (mag) |mg| {
                var s: f64 = 0;
                const b0 = j / 128 * 128;
                for (b0..b0 + 128) |q| s += s1[r * n + q];
                mg[r * n + j] = @abs(sc) * s / @sqrt(128.0);
            }
        };
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(out, null, acc, order);
    }
};

/// Every registration and control: the three codebooks at 1-8 bits, the fixed plans, and the planted faults.
pub const registrations = blk: {
    const books = [_]exl3.Codebook{ .inst3, .mcg, .mul1 };
    var list: [books.len * 8 + 2 + books.len + 1]Reg = undefined;
    var i = 0;
    for (books) |b| for (1..9) |bits| {
        list[i] = .{ .book = b, .bits = bits };
        i += 1;
    };
    list[i] = .{ .plan = .fixed_1_4 };
    list[i + 1] = .{ .plan = .fixed_2_8 };
    i += 2;
    for (books) |b| {
        list[i] = .{ .book = b, .fault = .split };
        i += 1;
    }
    list[i] = .{ .fault = .sanitized };
    break :blk list;
};

/// The registrations over one Dense (its decode kernel and its candidate module), for the runner's list.
pub const Set = struct {
    de: ?exl3.Dense = null,
    cases: [registrations.len]Case = undefined,
    n: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, list: *std.ArrayList(g.Kernel)) !void {
        self.de = try exl3.Dense.load(gpu.d);
        try self.de.?.candidates(max_k);
        for (registrations) |r| {
            self.cases[self.n] = .{ .rig = .{ .gpu = gpu, .s = s }, .de = &self.de.?, .reg = r };
            self.n += 1;
        }
        for (self.cases[0..self.n]) |*cs| try list.append(gpu.gpa, cs.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.cases[0..self.n]) |*cs| cs.deinit();
        if (self.de) |*d| d.unload();
    }
};

/// tf-cuda-test exl3-promptd-contract: these registrations' gates 1-4 and controls alone (kernel-contract has them).
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    const set = try gpu.gpa.create(Set);
    defer gpu.gpa.destroy(set);
    set.* = .{};
    defer set.deinit();
    try set.add(gpu, s, &list);
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x7064, &t);
    std.debug.print("SUMMARY exl3 promptd {d} kernels: {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "exl3-promptd-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "promptd contract: every served shape and edge is one the kernel takes, and a control's other plan too" {
    const a = std.testing.allocator;
    for (registrations) |r| {
        const sp = spec(r, "t");
        const list = try c.shapes(a, sp);
        defer a.free(list);
        for (sp.served) |s| try std.testing.expect(takes(r, s));
        // an edge the plan cannot divide is refused by the kernel and by prepare alike, never run
        for (list) |s| if (takes(r, s)) {
            const w = weightOf(r, s);
            try std.testing.expect(divides(w));
            try std.testing.expect(@as(?usize, exl3.variantIndex(w.halves, w.codebook, exl3.planOf(w).wk)) != null);
        };
    }
}

const Form = struct { split: bool, wk: u32 };

fn formOf(r: Reg, s: c.Shape) Form {
    const p = exl3.planOf(weightOf(r, s));
    return .{ .split = p.sk > 1, .wk = p.wk };
}

test "promptd contract: every plan() form at K and N to 16384 is a served shape's of the own-plan registrations" {
    const a = std.testing.allocator;
    const sp = spec(.{}, "t");
    const list = try c.shapes(a, sp);
    defer a.free(list);
    var have: std.ArrayList(Form) = .empty;
    defer have.deinit(a);
    for (list) |s| if (takes(.{}, s)) {
        const f = formOf(.{}, s);
        for (have.items) |x| {
            if (std.meta.eql(x, f)) break;
        } else try have.append(a, f);
    };
    var k: usize = 128;
    while (k <= 16384) : (k += 128) {
        var n: usize = 128;
        while (n <= 16384) : (n += 128) {
            const f = formOf(.{}, .{ .rows = 1, .n = n, .k = k });
            for (have.items) |x| {
                if (std.meta.eql(x, f)) break;
            } else {
                std.debug.print("unregistered promptd form: split {} {d} warps (n {d}, k {d})\n", .{ f.split, f.wk, n, k });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "promptd contract: gate 4's all-ones states decode to one weight that overflows an fp16 sum at every width" {
    for (registrations) |r| {
        if (r.plan != .own or r.fault != .none) continue;
        const width = fmt.Width{ .halves = @intCast(2 * r.bits) };
        var words: [128]i16 = @splat(-1);
        var tile: [256]f16 = undefined;
        fmt.unpack(width, formatBook(r.book), words[0..width.tileWords()], 1, 1, &tile);
        const w: f64 = @floatCast(tile[0]);
        for (tile) |v| try std.testing.expectEqual(tile[0], v);
        // K equal terms of 4,096 w at the stress shape's K: past fp16's range by the margin the spec asks
        try std.testing.expect(@abs(w) * 4096 * max_k >= 4 * 65504);
    }
}
