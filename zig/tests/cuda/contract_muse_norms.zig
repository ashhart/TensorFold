//! Muse Spark's norm glue under the contract against float64; the 8-block cluster forms bit-equal to the block forms.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_muse_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

/// A thread's ceil(D / 1024) squares, then 32 lanes and 32 warps (or 8 blocks' 4 warps) by halves.
fn rmsDepth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 12;
}

const tiles: [3]c.Tile = .{ .{}, .{}, .{ .size = 1024 } };
const half_why = "an fp16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const point = "bf16(fp32(y + y2) * scale) is exllamav3's rounding point, replayed on the host from the kernel's two IEEE fp32 " ++
    "steps (an add, then a multiply: nothing to contract); float64 from there";

pub const Kind = enum { embed, gate_res, gate_x, gate_xf, esum };

pub fn spec(kind: Kind, cluster: bool) c.Spec {
    const hidden = &[_]c.Shape{ .{ .rows = 8, .n = 8192, .k = 8192 }, .{ .rows = 1, .n = 8192, .k = 8192 } };
    const latent = &[_]c.Shape{ .{ .rows = 8, .n = 2048, .k = 4096 }, .{ .rows = 1, .n = 2048, .k = 4096 } };
    const gate_sums: c.Sums = .{ .exact = false, .why = point };
    var s: c.Spec = switch (kind) {
        .embed => .{ .name = "cuda muse muse_embed_norm (bf16 rows, fp32 out)", .accum = .f32, .depth = rmsDepth, .ops = 8, .out = .f32, .tiles = tiles, .served = hidden },
        .gate_res => .{ .name = "cuda muse muse_gate_norm (the residual, in place)", .accum = .f32, .depth = rmsDepth, .out = .f32, .in_place = true, .tiles = tiles, .served = hidden, .sums = gate_sums },
        .gate_x => .{ .name = "cuda muse muse_gate_norm (the next norm's fp16 x)", .accum = .f32, .depth = rmsDepth, .out = .f16, .tiles = tiles, .served = hidden, .sums = .{ .bf16 = 0, .exact = false, .why = half_why } },
        .gate_xf => .{ .name = "cuda muse muse_gate_norm (the next norm's fp32 xf)", .accum = .f32, .depth = rmsDepth, .out = .f32, .tiles = tiles, .served = hidden, .sums = gate_sums },
        .esum => .{ .name = "cuda muse muse_expert_sum (8 slots, a rank's fp16 columns)", .accum = .f32, .depth = rmsDepth, .out = .f16, .tiles = tiles, .served = latent, .sums = .{ .bf16 = 0, .exact = false, .why = half_why } },
    };
    if (cluster) {
        s.name = switch (kind) {
            .embed => unreachable,
            .gate_res => "cuda muse muse_gate_norm_c (8 blocks a row: the residual)",
            .gate_x => "cuda muse muse_gate_norm_c (8 blocks a row: fp16 x)",
            .gate_xf => "cuda muse muse_gate_norm_c (8 blocks a row: fp32 xf)",
            .esum => "cuda muse muse_expert_sum_c (8 blocks a row)",
        };
        s.replaces = if (kind == .esum) "muse_expert_sum" else "muse_gate_norm";
        s.nan = .{ .stated = "the bytes of the 1,024-thread form, non-finite inputs included" };
    }
    return s;
}

const slots = 8; // Muse Spark's top 8
const post_eps: f32 = 1e-8;
const next_eps: f32 = 1e-5;

pub const Norm = struct {
    rig: rig_.Rig,
    kind: Kind,
    cluster: bool = false,
    f: cuda.Function,
    base: ?cuda.Function = null,
    arena: std.heap.ArenaAllocator,
    r: ?rows.Rows = null,
    shape: c.Shape = undefined,
    // host values as the kernel reads them: gate (res, y, y2, scale, ab, w); esum (y, y2, wts, scale)
    v: [6][]f64 = undefined,
    t: []f64 = &.{}, // the rounding point bf16(fp32(y + y2) * scale)
    dev: [8]u64 = @splat(0),
    lo: usize = 0,
    host: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.host.deinit(self.rig.gpu.gpa);
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Norm);
        v.check = null;
        v.replaced = null;
        break :blk v;
    };
    const vt_rep = g.hooks(Norm);

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(self.kind, self.cluster), .ctx = self, .vt = if (self.cluster) &vt_rep else &vt_ref };
    }

    fn D(self: *const Norm) usize {
        return self.shape.k;
    }

    fn outW(self: *const Norm) usize {
        return if (self.kind == .esum) @min(self.shape.n, self.shape.k) else self.shape.k;
    }

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.k > 16 * 1024 or shape.rows == 0) return false;
        if (self.kind == .esum and shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (self.r) |r| r.deinit();
        self.r = null;
        self.shape = shape;
        const ga = self.rig.gpu.gpa;
        if (self.kind == .embed) return self.prepareEmbed(fill, seed);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = self.D();
        const n = if (self.kind == .esum) shape.rows * slots else shape.rows; // rows of y
        const lens = if (self.kind == .esum) [6]usize{ n * d, n * d, n, d, 0, 0 } else [6]usize{ n * d, n * d, n * d, d, 2 * d, d };
        for (&self.v, lens) |*x, l| x.* = try a.alloc(f64, l);
        const v = &self.v;
        const ys = if (self.kind == .esum) [2]usize{ 0, 1 } else [2]usize{ 1, 2 };
        for (ys) |i| for (v[i]) |*x| {
            x.* = if (fill == .stress) mu.int(rng, -16, 16) else mu.uni(rng, f32, -2, 2);
        };
        if (self.kind == .esum) {
            for (v[2]) |*x| x.* = if (fill == .stress) 1 else mu.uni(rng, f32, 0, 0.5);
            for (v[3]) |*x| x.* = if (fill == .stress) 1 else mu.uni(rng, u16, 0.5, 1.5);
            self.lo = if (self.outW() < d and seed % 2 == 1) d - self.outW() else 0; // either rank's columns
        } else {
            for (v[0]) |*x| x.* = if (fill == .stress) mu.int(rng, -64, 64) else mu.uni(rng, f32, -4, 4);
            for (v[3]) |*x| x.* = if (fill == .stress) 1 else mu.uni(rng, f32, 0.25, 2);
            for (v[4]) |*x| x.* = if (fill == .stress) 1 else mu.uni(rng, f32, 0.5, 1.5);
            for (v[5]) |*x| x.* = mu.uni(rng, f32, 0.5, 1.5);
        }
        if (fill == .nonfinite) {
            const y = v[ys[0]];
            y[0] = std.math.nan(f64); // row (or slot) 0: every output NaN
            const second: usize = if (self.kind == .esum) slots + 2 else 2; // row 1's third slot, or row 2
            if (n > second) v[ys[1]][second * d + 5] = std.math.inf(f64); // inf * 0 at one column, the rest 0 * finite
            if (self.kind == .esum) {
                if (n > slots + 1) v[2][slots + 1] = -std.math.inf(f64); // row 1's second slot weight
            } else if (shape.rows > 3) v[0][3 * d + d - 1] = -std.math.inf(f64); // res: the next norm's row 3
        }
        self.t = try a.alloc(f64, n * d);
        for (self.t, 0..) |*o, i| {
            const sum: f32 = @as(f32, @floatCast(v[ys[0]][i])) + @as(f32, @floatCast(v[ys[1]][i]));
            o.* = mu.bf16r(sum * @as(f32, @floatCast(v[3][i % d])));
        }
        for (0..6) |i| {
            if (lens[i] == 0) continue;
            self.dev[i] = try mu.put(&self.rig, ga, f32, v[i]);
        }
        const ob = shape.rows * self.outW();
        switch (self.kind) {
            .gate_res => {
                self.dev[6] = try self.rig.output(ob * 4);
                self.dev[7] = try self.rig.scratch(ob * 2 + ob * 4);
            },
            .gate_x => {
                self.dev[6] = try self.rig.output(ob * 2);
                self.dev[7] = try self.rig.scratch(ob * 8);
            },
            .gate_xf => {
                self.dev[6] = try self.rig.output(ob * 4);
                self.dev[7] = try self.rig.scratch(ob * 4 + ob * 2);
            },
            .esum => self.dev[6] = try self.rig.output(ob * 2),
            .embed => unreachable,
        }
        return true;
    }

    fn prepareEmbed(self: *Norm, fill: g.Fill, seed: u64) !bool {
        const ga = self.rig.gpu.gpa;
        var r = try rows.Rows.init(ga, .rms, self.shape);
        r.eps = next_eps;
        self.r = r;
        r.fill(fill, seed);
        @memset(r.w, c.toBf16(1)); // the weightless embed_norm
        const n = self.shape.rows;
        const k = self.shape.k;
        const table = try self.arena.allocator().alloc(u16, r.x.len); // row i of x at table row n - 1 - i
        for (0..n) |i| @memcpy(table[(n - 1 - i) * k ..][0..k], r.x[i * k ..][0..k]);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(table));
        const ids = try self.arena.allocator().alloc(i64, n);
        for (ids, 0..) |*o, i| o.* = @intCast(n - 1 - i);
        self.dev[1] = try mu.ints(&self.rig, i64, ids);
        self.dev[2] = try self.rig.output(r.outputs() * 4);
        return true;
    }

    /// Where the view's buffers sit in a call: res, x, xf (gate) from the output and scratch slots.
    fn gateOuts(self: *const Norm, out: u64, extra: u64) [3]u64 {
        const ob = self.shape.rows * self.D();
        return switch (self.kind) {
            .gate_res => .{ out, extra, extra + ob * 2 },
            .gate_x => .{ extra, out, extra + ob * 4 },
            else => .{ extra, extra + ob * 4, out },
        };
    }

    fn launch(self: *Norm, f: cuda.Function, cluster: bool, out: u64, extra: u64) !void {
        const s = self.rig.s;
        const n = self.shape.rows;
        var a: cuda.Args = .{};
        switch (self.kind) {
            .embed => {
                inline for (.{ self.dev[0], self.dev[1], self.dev[2] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(self.D())));
                a.add(next_eps);
            },
            .esum => {
                inline for (.{ self.dev[0], self.dev[1], self.dev[2], self.dev[3] }) |p| a.add(p);
                a.add(post_eps);
                a.add(out);
                inline for (.{ self.D(), slots, self.lo, self.outW() }) |v| a.add(@as(c_int, @intCast(v)));
            },
            else => {
                const o = self.gateOuts(out, extra);
                try mu.copy(&self.rig, o[0], self.dev[0], n * self.D() * 4);
                inline for (.{ o[0], self.dev[1], self.dev[2], self.dev[3] }) |p| a.add(p);
                a.add(@as(c_int, 1));
                a.add(post_eps);
                inline for (.{ self.dev[4], self.dev[5] }) |p| a.add(p);
                a.add(next_eps);
                inline for (.{ o[1], o[2] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(self.D())));
            },
        }
        const grid: u32 = @intCast(if (cluster) n * 8 else n);
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid, .y = 1, .z = 1 }, .block = .{ .x = if (cluster) 128 else 1024 } }, s, &a);
        try s.synchronize();
    }

    /// The family's launch: ops.gateNormAt / moe's expert sum (rowLaunch), forward's embed.
    pub fn run(self: *Norm) !void {
        return self.launch(self.f, self.cluster, self.dev[6], self.dev[7]);
    }

    pub fn count(self: *Norm) usize {
        return self.shape.rows * self.outW();
    }

    /// Float64 outputs from the rounding point on; mag unused (bounds derives each allowance).
    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        if (self.kind == .embed) return self.r.?.reference(want, mag);
        @memset(mag, 0);
        return self.eval(.none, .sequential, want, null);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .embed) return self.r.?.simulate(acc, order, out);
        return self.eval(acc, order, out, null);
    }

    /// Each output's allowance from first-order propagation of its fp32 roundings (doubled for the second order).
    pub fn bounds(self: *Norm, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.kind == .embed) return; // the rows family's bound, as registered
        const tmp = try self.arena.allocator().alloc(f64, want.len);
        try self.eval(.none, .sequential, tmp, allow);
    }

    fn sq(acc: c.Accum, order: g.Order, a: std.mem.Allocator, x: []const f64) !f64 {
        const t = try a.alloc(f64, x.len);
        defer a.free(t);
        for (t, x) |*o, v| o.* = v * v;
        return if (acc == .none) blk: {
            var s: f64 = 0;
            for (t) |v| s += v;
            break :blk s;
        } else rows.sumIn(acc, order, t);
    }

    /// Outputs with the sums in `acc` (none: float64), or (err non-null) each output's allowance.
    fn eval(self: *Norm, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const d = self.D();
        const df: f64 = @floatFromInt(d);
        const g1 = mu.gamma(@floatFromInt(rmsDepth(self.shape)));
        const e_rs = 0.5 * g1 + 3 * mu.u + mu.fast; // ss / D, + eps, rsqrtf
        const v = self.v;
        if (self.kind == .esum) {
            const accs = try a.alloc(f64, 2 * d);
            defer a.free(accs);
            const acc_v = accs[0..d];
            const acc_e = accs[d..];
            for (0..self.shape.rows) |r| {
                @memset(accs, 0);
                for (0..slots) |k| {
                    const p = r * slots + k;
                    const t = self.t[p * d ..][0..d];
                    const rs = 1 / @sqrt(try sq(acc, order, a, t) / df + post_eps);
                    for (acc_v, acc_e, t) |*o, *e, x| {
                        o.* += x * rs * v[2][p];
                        e.* += @abs(x * rs * v[2][p]) * (e_rs + 2 * mu.u + mu.gamma(slots));
                    }
                }
                const w = self.outW();
                for (0..w) |j| {
                    out[r * w + j] = acc_v[self.lo + j];
                    if (err) |e| e[r * w + j] = mu.allow(.f16, acc_v[self.lo + j], 2 * acc_e[self.lo + j]);
                }
            }
            return;
        }
        const buf = try a.alloc(f64, 2 * d);
        defer a.free(buf);
        const hv = buf[0..d];
        const he = buf[d..];
        for (0..self.shape.rows) |r| {
            const t = self.t[r * d ..][0..d];
            const rs1 = 1 / @sqrt(try sq(acc, order, a, t) / df + post_eps);
            var sh: f64 = 0;
            for (hv, he, t, 0..) |*o, *e, x, i| {
                const ra = v[0][r * d + i] * v[4][i];
                const yb = x * rs1 * v[4][d + i];
                o.* = ra + yb;
                e.* = @abs(yb) * (e_rs + mu.u) + 3 * mu.u * (@abs(ra) + @abs(yb));
                sh += 2 * @abs(o.*) * e.*;
            }
            const ss2 = try sq(acc, order, a, hv);
            const rs2 = 1 / @sqrt(ss2 / df + next_eps);
            const e_rs2 = 0.5 * (g1 + if (ss2 > 0) sh / ss2 else 0) + 3 * mu.u + mu.fast;
            for (hv, he, 0..) |x, e, i| {
                const o = x * rs2 * v[5][i];
                const eo = @abs(rs2 * v[5][i]) * e + @abs(o) * (e_rs2 + 2 * mu.u);
                const at = r * d + i;
                switch (self.kind) {
                    .gate_res => {
                        out[at] = x;
                        if (err) |el| el[at] = mu.allow(.f32, x, 2 * e);
                    },
                    .gate_x, .gate_xf => {
                        out[at] = o;
                        if (err) |el| el[at] = mu.allow(if (self.kind == .gate_x) .f16 else .f32, o, 2 * eo);
                    },
                    else => unreachable,
                }
            }
        }
    }

    /// The 1,024-thread form on the same inputs, into fresh buffers poisoned as the rig's outputs are.
    pub fn replaced(self: *Norm) ![]const u8 {
        const d = self.rig.gpu.d;
        const ob = self.count() * (if (spec(self.kind, false).out == .f16) @as(usize, 2) else 4);
        var out = try cuda.DeviceBuffer.alloc(d, ob);
        defer out.free();
        try out.fill8(0xff, self.rig.s.handle);
        var extra = try cuda.DeviceBuffer.alloc(d, self.shape.rows * self.D() * 8 + 4);
        defer extra.free();
        try self.launch(self.base.?, false, out.ptr, extra.ptr);
        try self.host.resize(self.rig.gpu.gpa, ob);
        try out.download(0, self.host.items);
        return self.host.items;
    }

    pub fn check(self: *Norm, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        return mu.sameBytes(&self.msg, got, try self.replaced(), self.baseName(), spec(self.kind, false).out, at);
    }

    fn baseName(self: *const Norm) []const u8 {
        return if (self.kind == .esum) "muse_expert_sum" else "muse_gate_norm";
    }
};
