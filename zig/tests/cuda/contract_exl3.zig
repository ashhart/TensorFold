//! EXL3's replacements of linear_body under the contract on CUDA: the mid-M kernels, linear_wc, the groups, rot_group.
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

/// What a registration launches; a control poses as its kind with one planted fault and must go red at gate 3.
pub const Kind = enum { midm, wc, group, rot_group };

const why = "the K sums of decoded fp16 weights: no input makes them exact in fp32, so gate 4 proves fp16's miss (an overflow) alone";
const rule = "linear_body's bytes on the same inputs: a NaN and both infinities in x reach the outputs as linear_body's do";

/// A K sum in any order (gate 4 simulates a sequential one), then the warps', splits' and output rotation's adds.
fn depth(s: c.Shape) usize {
    return s.k + 64;
}

fn rotDepth(_: c.Shape) usize {
    return 7 + 8;
}

/// The widths with mid-M kernels (exl3.mpg_variants); linear_wc takes 4- and 6-bit.
const mid_widths = [_]u32{ 6, 8, 10, 12 };

fn name(kind: Kind, halves: u32, control: bool) []const u8 {
    inline for (mid_widths) |h| if (h == halves) return switch (kind) {
        .midm => if (control) std.fmt.comptimePrint("control: exl3 {d}-bit mid-M kernels at another K split posing as linear_body", .{h / 2}) else std.fmt.comptimePrint("cuda exl3 mid-M kernels tf_exl3_mpg ({d}-bit, 1-128 rows)", .{h / 2}),
        else => if (control) std.fmt.comptimePrint("control: exl3 {d}-bit linear_wc at another K split posing as linear_body", .{h / 2}) else std.fmt.comptimePrint("cuda exl3 linear_wc tf_exl3_wc ({d}-bit, 1-128 rows)", .{h / 2}),
    };
    unreachable;
}

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

// The 27B's shapes (K 5120 / 6144), then shapes reaching the rest of plan()'s forms (the host test below sweeps them)
const qwen4 = [_]c.Shape{ sh(128, 1024, 5120), sh(100, 17408, 5120), sh(49, 6144, 5120), sh(33, 5120, 6144), sh(24, 12288, 5120), sh(64, 5120, 6144) };
const qwen6 = [_]c.Shape{ sh(128, 2048, 5120), sh(97, 4096, 5120), sh(40, 2048, 5120) };
const small_k = [_]c.Shape{ sh(17, 1024, 512), sh(49, 1024, 512), sh(17, 8192, 1024), sh(49, 8192, 1024) };
const wc_k = [_]c.Shape{ sh(17, 1024, 640), sh(33, 1024, 512), sh(17, 1024, 1024), sh(17, 1024, 1408), sh(49, 1024, 640), sh(49, 1024, 1024), sh(49, 1024, 1408), sh(17, 4096, 1024), sh(17, 4096, 1280), sh(17, 8192, 1152), sh(33, 4096, 1280), sh(49, 8192, 1152) };
const wc6_k = [_]c.Shape{ sh(17, 1024, 512), sh(17, 1024, 640), sh(33, 1024, 512), sh(17, 1024, 1024), sh(33, 1024, 640), sh(17, 1024, 1408), sh(33, 1024, 1024), sh(49, 1024, 1408), sh(17, 4096, 1024), sh(17, 4096, 1280), sh(33, 4096, 1024), sh(17, 8192, 1152), sh(49, 8192, 1152) };
// n names the members (members())
const groups = [_]c.Shape{ sh(16, 6144, 5120), sh(16, 17408, 5120), sh(7, 12288, 5120), sh(16, 9216, 512) };

fn served(kind: Kind, halves: u32) []const c.Shape {
    return switch (kind) {
        .midm => switch (halves) {
            8 => &(qwen4 ++ small_k[0..1].*),
            12 => &(qwen6 ++ small_k[2..4].*),
            else => &(qwen4[0..1].* ++ small_k),
        },
        .wc => if (halves == 8) &(qwen4 ++ wc_k) else &(qwen6 ++ wc6_k),
        .group, .rot_group => &groups,
    };
}

/// The served shapes: every form plan() gives the replacement at each width, every row edge at the first.
pub fn spec(kind: Kind, halves: u32, control: bool) c.Spec {
    const rows = [3]c.Tile{ .{ .size = 16 }, .{}, .{} };
    const sums: c.Sums = .{ .f16 = 4, .bf16 = 0, .exact = false, .why = why };
    return switch (kind) {
        .midm, .wc => .{
            .name = name(kind, halves, control),
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = .bf16,
            .replaces = "tf_exl3_linear (linear_body)",
            .nan = .{ .stated = rule },
            .tiles = rows,
            .served = served(kind, halves),
            .sums = sums,
        },
        .group => .{
            .name = if (control) "control: exl3 group with a member's scales swapped posing as its members' linear_body" else "cuda exl3 projection groups tf_exl3_group_8_2_4 / _8 (1-16 rows)",
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = .bf16,
            .replaces = "each member's tf_exl3_linear (linear_body)",
            .nan = .{ .stated = rule },
            .tiles = rows,
            .served = served(kind, halves),
            .sums = sums,
        },
        .rot_group => .{
            .name = if (control) "control: exl3 rot_group with a member's suh swapped posing as rot_in" else "cuda exl3 tf_exl3_rot_group (a group's rotated inputs, 1-16 rows)",
            .accum = .f32,
            .depth = rotDepth,
            .ops = 2,
            .out = .f16,
            .replaces = "each member's tf_exl3_rot_in",
            .tiles = rows,
            .served = served(kind, halves),
            .sums = .{ .f16 = 4, .bf16 = 0, .exact = false, .why = "128-term rotations of bf16 inputs: fp32 is not exact on them, so gate 4 proves fp16's miss alone" },
        },
    };
}

/// A weight's host copy (trellis words, scales) and its device tensors.
const Mat = struct {
    k: usize,
    n: usize,
    trellis: []i16 = &.{},
    suh: []u16 = &.{},
    svh: []u16 = &.{},
    w: exl3.Weight = undefined,
};

/// The members a group shape names (n): their output widths.
fn members(n: usize) []const usize {
    return switch (n) {
        17408 => &.{ 17408, 17408 }, // gate + up, 8 warps
        12288 => &.{ 12288, 17408, 10240 }, // q + gate + qkv, 8 warps
        9216 => &.{ 8192, 1024 }, // two layers at K 512: 4 warps, no split
        else => &.{ 6144, 1024 }, // z + k_v, 4 warps, K split
    };
}

/// Another valid (K splits, warps) of the same shape: the controls' planted fault.
fn other(w: exl3.Weight) exl3.Weight {
    const p = exl3.planOf(w);
    var x = w;
    x.wk = if (p.wk == 4) 8 else 4;
    x.sk = if (p.wk == 4) @max(1, p.sk / 2) else p.sk * 2;
    return x;
}

pub const Case = struct {
    rig: rig_.Rig,
    lin: *exl3.Linear,
    kind: Kind,
    halves: u32 = 8,
    control: bool = false,
    shape: c.Shape = undefined,
    mats: [3]Mat = .{ .{ .k = 0, .n = 0 }, .{ .k = 0, .n = 0 }, .{ .k = 0, .n = 0 } },
    nm: usize = 1,
    x: []u16 = &.{},
    xd: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    xh: u64 = 0,
    xh_base: u64 = 0,
    z: u64 = 0,
    counters: u64 = 0,
    wbuf: [3]exl3.Weight = undefined,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Case) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Case) void {
        const a = self.rig.gpu.gpa;
        for (&self.mats) |*m| {
            a.free(m.trellis);
            a.free(m.suh);
            a.free(m.svh);
            m.* = .{ .k = 0, .n = 0 };
        }
        a.free(self.x);
        self.x = &.{};
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.kind, self.halves, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Case);

    fn random16(rng: std.Random, out: []u16) void {
        for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
    }

    /// Random trellis words and scales, x bf16 in [-1, 1); non-finite: x[1][3] NaN, x[2][5] +Inf, x[2][9] -Inf.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const grouped = self.kind == .group or self.kind == .rot_group;
        if (shape.rows == 0 or shape.rows > (if (grouped) @as(usize, 16) else 128) or shape.k % 128 != 0 or shape.n % 128 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const widths: []const usize = if (grouped) members(shape.n) else &.{shape.n};
        self.nm = widths.len;
        const width = fmt.Width{ .halves = @intCast(self.halves) };
        var zb: usize = 0;
        var cb: usize = 0;
        for (widths, self.mats[0..widths.len]) |n, *m| {
            m.k = shape.k;
            m.n = n;
            m.trellis = try a.alloc(i16, (shape.k / 16) * (n / 16) * width.tileWords());
            m.suh = try a.alloc(u16, shape.k);
            m.svh = try a.alloc(u16, n);
            if (fill == .stress and self.kind != .rot_group) { // one weight at every width: all-ones states, scales 1
                @memset(m.trellis, @bitCast(@as(u16, 0xffff)));
                @memset(m.suh, 0x3c00);
                @memset(m.svh, 0x3c00);
            } else {
                for (m.trellis) |*v| v.* = rng.int(i16);
                random16(rng, m.suh);
                random16(rng, m.svh);
            }
            const st = exl3.strides(shape.k, n, self.halves, false);
            m.w = .{ .trellis = try self.rig.input(std.mem.sliceAsBytes(m.trellis)), .suh = try self.rig.input(std.mem.sliceAsBytes(m.suh)), .svh = try self.rig.input(std.mem.sliceAsBytes(m.svh)), .k = @intCast(shape.k), .n = @intCast(n), .halves = self.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
            zb += @max(exl3.Linear.zBytes(m.w, 128), exl3.Linear.zBytes(other(m.w), 128));
            cb += exl3.Linear.counterBytes(m.w);
        }
        if (grouped and !exl3.groupable(self.weights(), shape.rows)) return false;
        self.x = try a.alloc(u16, shape.rows * shape.k);
        for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        switch (fill) {
            .random => {},
            .nonfinite => if (shape.rows > 2) {
                self.x[shape.k + 3] = 0x7fc0;
                self.x[2 * shape.k + 5] = 0x7f80;
                self.x[2 * shape.k + 9] = 0xff80;
            },
            .stress => if (self.kind == .rot_group) {
                for (self.x) |*v| v.* = c.toBf16(c.fromBf16(v.*) * 64);
            } else { // one input a block, rotated to 4096: K equal terms, past fp16's range
                @memset(self.x, 0);
                for (0..shape.rows) |r| for (0..shape.k / 128) |b| {
                    self.x[r * shape.k + b * 128] = c.toBf16(46336);
                };
            },
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.out = try self.rig.output(self.outElems() * 2);
        self.base = try self.rig.scratch(self.ysElems() * 2); // (linear_body's y; rot_group's group's y too)
        self.xh = try self.rig.scratch(self.nm * shape.rows * shape.k * 2);
        self.xh_base = try self.rig.scratch(self.nm * shape.rows * shape.k * 2);
        self.z = try self.rig.scratch(@max(zb, 16));
        self.counters = try self.rig.scratch(@max(cb, 16));
        return true;
    }

    fn weights(self: *Case) []const exl3.Weight {
        for (self.mats[0..self.nm], 0..) |m, i| self.wbuf[i] = m.w;
        return self.wbuf[0..self.nm];
    }

    /// Output elements: the members' rows side by side (rot_group: their rotated inputs, fp16).
    fn outElems(self: *const Case) usize {
        return if (self.kind == .rot_group) self.nm * self.shape.rows * self.shape.k else self.ysElems();
    }
    fn ysElems(self: *const Case) usize {
        var n: usize = 0;
        for (self.mats[0..self.nm]) |m| n += m.n;
        return self.shape.rows * n;
    }

    fn ys(self: *const Case, at: u64) [3]u64 {
        var y: [3]u64 = .{ 0, 0, 0 };
        var off: u64 = 0;
        for (self.mats[0..self.nm], 0..) |m, i| {
            y[i] = at + off;
            off += self.shape.rows * m.n * 2;
        }
        return y;
    }

    pub fn run(self: *Case) !void {
        const m = self.shape.rows;
        switch (self.kind) {
            .midm, .wc => {
                self.lin.midm = true;
                self.lin.use_wc = self.kind == .wc;
                self.lin.min_rows = 1;
                defer self.restore();
                const w = if (self.control) other(self.mats[0].w) else self.mats[0].w;
                try self.lin.matmul(self.rig.s, w, self.xd, .bf16, m, self.out, .bf16, self.xh, self.z, self.counters);
            },
            .group, .rot_group => {
                var ws = self.weights();
                var bad: [3]exl3.Weight = undefined;
                if (self.control) {
                    @memcpy(bad[0..self.nm], ws);
                    if (self.kind == .group) bad[1].svh = bad[0].svh else bad[1].suh = bad[0].suh;
                    ws = bad[0..self.nm];
                }
                const y = self.ys(if (self.kind == .group) self.out else self.base);
                try self.lin.matmulGroup(self.rig.s, ws, self.xd, .bf16, m, y[0..self.nm], .bf16, if (self.kind == .rot_group) self.out else self.xh, self.z, self.counters);
            },
        }
        try self.rig.s.synchronize();
    }

    fn restore(self: *Case) void {
        self.lin.midm = true;
        self.lin.use_wc = true;
        self.lin.min_rows = 17;
    }

    /// linear_body on the same inputs, member by member (rot_group: each member's rot_in, the xh it leaves).
    pub fn replaced(self: *Case) ![]const u8 {
        self.lin.midm = false;
        defer self.restore();
        const m = self.shape.rows;
        const y = self.ys(self.base);
        for (self.mats[0..self.nm], 0..) |mat, i| {
            const xh = self.xh_base + i * m * self.shape.k * 2;
            try self.lin.matmul(self.rig.s, mat.w, self.xd, .bf16, m, y[i], .bf16, xh, self.z, self.counters);
        }
        try self.rig.s.synchronize();
        if (self.kind == .rot_group) return self.rig.bytes(self.xh_base, self.outElems() * 2);
        return self.rig.bytes(self.base, self.outElems() * 2);
    }

    /// Gate 1: linear_body's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        const s = spec(self.kind, self.halves, self.control);
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, s.replaces.? });
        for (0..self.count()) |i| if (c.classOf(c.read(s.out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Case) usize {
        return self.outElems();
    }

    /// Member i's x * suh rotated per 128 (rot_in's sums), each output's 128 terms summed in `acc` (null: float64).
    fn rotated(self: *Case, i: usize, xh: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.gpu.gpa;
        const k = self.shape.k;
        const h = try a.alloc(f64, 128 * 128); // the rotation's matrix: row i the image of basis vector i
        defer a.free(h);
        @memset(h, 0);
        for (0..128) |q| h[q * 128 + q] = 1;
        fmt.rotate(h, 128, 128, 1);
        var v: [128]f64 = undefined;
        var t: [128]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..k / 128) |b| {
            for (&v, 0..) |*e, q| e.* = c.fromBf16(self.x[r * k + b * 128 + q]) * @as(f64, @floatCast(@as(f16, @bitCast(self.mats[i].suh[b * 128 + q]))));
            for (0..128) |j| {
                var abs: f64 = 0;
                for (&t, 0..) |*e, q| {
                    e.* = h[q * 128 + j] * v[q];
                    abs += @abs(e.*);
                }
                var sum: f64 = 0;
                if (acc) |ac| {
                    sum = contract.rows.sumIn(ac, order, &t);
                } else {
                    for (t) |e| sum += e;
                }
                xh[r * k + b * 128 + j] = sum;
                if (mag) |m| m[r * k + b * 128 + j] = abs;
            }
        };
    }

    /// The rotated inputs the kernels take (linear_body's own rot_in's, member after member), fp16 values as float64.
    fn kernelXh(self: *Case, i: usize, xs: []f64) !void {
        _ = try self.replaced();
        const n = self.shape.rows * self.shape.k;
        const raw = try self.rig.bytes(self.xh_base + i * n * 2, n * 2);
        for (xs, 0..) |*v, q| v.* = @floatCast(@as(f16, @bitCast(std.mem.bytesToValue(u16, raw[2 * q ..][0..2]))));
    }

    /// Float64 outputs and their terms' magnitude; with `acc`, the K sums simulated in that accumulator and order.
    fn outputs(self: *Case, want: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.gpu.gpa;
        const k = self.shape.k;
        const rows = self.shape.rows;
        const xs = try a.alloc(f64, rows * k);
        defer a.free(xs);
        var at: usize = 0;
        for (0..self.nm) |i| {
            if (self.kind == .rot_group) {
                try self.rotated(i, want[at..][0 .. rows * k], if (mag) |mg| mg[at..][0 .. rows * k] else null, acc, order);
                at += rows * k;
                continue;
            }
            try self.kernelXh(i, xs); // (the K sums are the linear's: its inputs are rot_in's fp16 rows, as they ran)
            const n = self.mats[i].n;
            const wq = try a.alloc(f16, k * n);
            defer a.free(wq);
            fmt.unpack(.{ .halves = @intCast(self.halves) }, .mul1, self.mats[i].trellis, k / 16, n / 16, wq);
            const ypre = try a.alloc(f64, rows * n);
            defer a.free(ypre);
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
                } else {
                    for (t) |v| sum += v;
                }
                ypre[r * n + j] = sum;
                s1[r * n + j] = abs;
            };
            fmt.rotate(ypre, rows, n, 1);
            for (0..rows) |r| for (0..n) |j| {
                const sc: f64 = @floatCast(@as(f16, @bitCast(self.mats[i].svh[j])));
                want[at + r * n + j] = ypre[r * n + j] * sc;
                if (mag) |mg| {
                    var s: f64 = 0;
                    const b0 = j / 128 * 128;
                    for (b0..b0 + 128) |q| s += s1[r * n + q];
                    mg[at + r * n + j] = @abs(sc) * s / @sqrt(128.0);
                }
            };
            at += rows * n;
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(out, null, acc, order);
    }
};

/// Every registration and control, on one Linear the cases share; `add` appends them to the runner's list.
const Reg = struct { k: Kind, h: u32 = 8, ctl: bool = false };

/// The mid-M kernels at every width they have, linear_wc at 4 and 6 bits, the groups, rot_group, and the controls.
pub const registrations = [_]Reg{
    .{ .k = .midm, .h = 6 },              .{ .k = .midm },                   .{ .k = .midm, .h = 10 },              .{ .k = .midm, .h = 12 },
    .{ .k = .wc },                        .{ .k = .wc, .h = 12 },            .{ .k = .group },                      .{ .k = .rot_group },
    .{ .k = .midm, .h = 6, .ctl = true }, .{ .k = .midm, .ctl = true },      .{ .k = .midm, .h = 10, .ctl = true }, .{ .k = .wc, .ctl = true },
    .{ .k = .group, .ctl = true },        .{ .k = .rot_group, .ctl = true },
};

pub const Set = struct {
    lin: ?exl3.Linear = null,
    cases: [registrations.len]Case = undefined,
    n: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, list: *std.ArrayList(g.Kernel)) !void {
        self.lin = try exl3.Linear.load(gpu.d);
        for (registrations) |x| {
            self.cases[self.n] = .{ .rig = .{ .gpu = gpu, .s = s }, .lin = &self.lin.?, .kind = x.k, .halves = x.h, .control = x.ctl };
            self.n += 1;
        }
        for (self.cases[0..self.n]) |*cs| try list.append(gpu.gpa, cs.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.cases[0..self.n]) |*cs| cs.deinit();
        if (self.lin) |*l| l.unload();
    }
};

/// `tf-cuda-test exl3-contract`: these registrations' gates 1-4 and controls alone (kernel-contract runs them all).
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    var set: Set = .{};
    defer set.deinit();
    try set.add(gpu, s, &list);
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x5eed, &t);
    std.debug.print("SUMMARY exl3 {d} kernels: {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    if (t.problems() > 0) return error.TestFailed;
}

/// A launch the dispatch can reach: its kernel (a group's by warps), and whether its plan splits K.
const Form = struct { kind: enum { wc, mpg, group }, index: usize, split: bool };

fn weightOf(k: usize, n: usize, halves: u32) exl3.Weight {
    return .{ .trellis = 0, .suh = 0, .svh = 0, .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
}

fn formOf(w: exl3.Weight, m: usize, use_wc: bool, min_rows: usize) ?Form {
    const split = exl3.planOf(w).sk > 1;
    return switch (exl3.pick(w, m, true, use_wc, min_rows) orelse return null) {
        .wc => |i| .{ .kind = .wc, .index = i, .split = split },
        .mpg => |i| .{ .kind = .mpg, .index = i, .split = split },
        .linear => null,
    };
}

fn put(a: std.mem.Allocator, set: *std.ArrayList(Form), f: Form) !void {
    for (set.items) |x| if (std.meta.eql(x, f)) return;
    try set.append(a, f);
}

/// Every form a registration's gate-3 shapes run (the mid-M and linear_wc cases at 1 row up, as Case.run sets them).
fn covered(a: std.mem.Allocator) !std.ArrayList(Form) {
    var set: std.ArrayList(Form) = .empty;
    errdefer set.deinit(a);
    for (registrations) |r| if (!r.ctl and r.k != .rot_group) {
        const shapes = try c.shapes(a, spec(r.k, r.h, false));
        defer a.free(shapes);
        for (shapes) |s| if (r.k == .group) {
            for (members(s.n)) |n| {
                const w = weightOf(s.k, n, 8);
                try put(a, &set, .{ .kind = .group, .index = exl3.planOf(w).wk, .split = exl3.planOf(w).sk > 1 });
            }
        } else if (formOf(weightOf(s.k, s.n, r.h), s.rows, r.k == .wc, 1)) |f| try put(a, &set, f);
    };
    return set;
}

// K and N in steps of 128 to 16384: past every threshold plan() and the dispatch read (64 blocks, 8 k tiles a warp)
test "contract_exl3: every mid-M and linear_wc form plan() reaches at 17-128 rows is a registration's gate-3 shape" {
    const a = std.testing.allocator;
    var have = try covered(a);
    defer have.deinit(a);
    var widths: [8]u32 = undefined; // every width with a mid-M kernel or a linear_wc
    var nw: usize = 0;
    inline for (.{ exl3.mpg_variants, exl3.wc_variants }) |vs| for (vs) |v| if (std.mem.indexOfScalar(u32, widths[0..nw], v.halves) == null) {
        widths[nw] = v.halves;
        nw += 1;
    };
    var k: usize = 128;
    while (k <= 16384) : (k += 128) {
        var n: usize = 128;
        while (n <= 16384) : (n += 128) for (widths[0..nw]) |h| for (17..129) |m| {
            const f = formOf(weightOf(k, n, h), m, true, 17) orelse continue;
            for (have.items) |x| {
                if (std.meta.eql(x, f)) break;
            } else {
                std.debug.print("unregistered: {t} {d} split {} ({d}-bit, {d} rows, n {d}, k {d})\n", .{ f.kind, f.index, f.split, h / 2, m, n, k });
                return error.TestUnexpectedResult;
            }
        };
    }
}

test "contract_exl3: every group form groupable() admits is a registration's gate-3 shape" {
    const a = std.testing.allocator;
    var have = try covered(a);
    defer have.deinit(a);
    var k: usize = 128;
    while (k <= 16384) : (k += 128) {
        var n: usize = 128;
        while (n <= 16384) : (n += 128) {
            const w = weightOf(k, n, 8);
            if (!exl3.groupable(&.{ w, w }, 16)) continue;
            const f: Form = .{ .kind = .group, .index = exl3.planOf(w).wk, .split = exl3.planOf(w).sk > 1 };
            for (have.items) |x| {
                if (std.meta.eql(x, f)) break;
            } else {
                std.debug.print("unregistered group form: {d} warps, split {} (n {d}, k {d})\n", .{ f.index, f.split, n, k });
                return error.TestUnexpectedResult;
            }
        }
    }
}
