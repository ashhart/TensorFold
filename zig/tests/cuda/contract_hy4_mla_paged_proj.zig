//! Hy4-flash paged MLA absorb, value, and post-norm contract across worlds and head counts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_mla_paged.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const lat = hy4.latent;
const nope = hy4.nope;
const vw = hy4.v_width;
const kvh = hy4.kv_head;
const qh = hy4.q_head;
const hidden = 3072;

pub const Kind = enum { absorb, value, post };

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

fn depthAbsorb(_: c.Shape) usize {
    return nope;
}

fn depthValue(_: c.Shape) usize {
    return lat;
}

/// post: a thread's 12 squares, the warp's five steps, the block's three.
fn depthPost(_: c.Shape) usize {
    return hidden / 256 + 8;
}

const absorb_shapes = [_]c.Shape{ .{ .rows = 16, .n = 8 * lat, .k = nope }, .{ .rows = 1, .n = 64 * lat, .k = nope }, .{ .rows = 17, .n = 32 * lat, .k = nope }, .{ .rows = 40, .n = 16 * lat, .k = nope } };
const value_shapes = [_]c.Shape{ .{ .rows = 16, .n = 8 * vw, .k = lat }, .{ .rows = 1, .n = 64 * vw, .k = lat }, .{ .rows = 17, .n = 32 * vw, .k = lat }, .{ .rows = 33, .n = 16 * vw, .k = lat } };
const post_shapes = [_]c.Shape{ .{ .rows = 7, .n = hidden, .k = hidden }, .{ .rows = 1, .n = hidden, .k = hidden }, .{ .rows = 64, .n = hidden, .k = hidden } };
const post_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps (fp16 overflows)";

pub fn spec(kind: Kind, world: usize, fault: set.Fault) c.Spec {
    const rows: c.Tile = .{ .size = 16 };
    var s: c.Spec = switch (kind) {
        .absorb => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_absorb (q_nope . kv_b's key rows into the fp32 latent)", .accum = .f32, .depth = depthAbsorb, .out = .f32, .tiles = .{ rows, .{}, .{} }, .served = &absorb_shapes },
        .value => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_value (latent . kv_b's value rows, times sigmoid(gate))", .accum = .f32, .depth = depthValue, .ops = 4, .out = .bf16, .tiles = .{ rows, .{}, .{} }, .served = &value_shapes },
        .post => .{ .name = switch (world) {
            1 => "cuda hy4_mla_paged tf_hy4_mla_post (one rank's o_proj rows, post_proj_norm)",
            2 => "cuda hy4_mla_paged tf_hy4_mla_post (two ranks' partials summed, post_proj_norm)",
            else => "cuda hy4_mla_paged tf_hy4_mla_post (four ranks' partials summed, post_proj_norm)",
        }, .accum = .f32, .depth = depthPost, .ops = 6, .out = .bf16, .served = &post_shapes, .sums = .{ .bf16 = 0, .why = post_why } },
    };
    if (fault != .none) s.name = set.controlName(switch (kind) {
        .absorb => "absorb",
        .value => "value",
        .post => "post",
    }, fault);
    return s;
}

/// One call of a projection-like kernel: its inputs on the host and the device, its outputs on the device.
pub const Proj = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    fault: set.Fault = .none,
    world: usize = 1,
    shape: c.Shape = undefined,
    heads: usize = 0,
    x: []f32 = &.{}, // absorb: q (bf16 values) [rows][heads][256]; value: ol [rows][heads][512]; post: parts
    w: []u16 = &.{}, // kv_b [heads][448][512], or post_proj_norm [hidden]
    gate: []u16 = &.{}, // value: [rows][heads x 256]
    in: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Proj) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Proj) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        a.free(self.gate);
        self.x = &.{};
        self.w = &.{};
        self.gate = &.{};
    }

    pub fn kernel(self: *Proj) g.Kernel {
        return .{ .spec = spec(self.kind, self.world, self.fault), .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(Proj);

    fn xWidth(self: *const Proj) usize {
        return switch (self.kind) {
            .absorb => qh,
            .value => lat,
            .post => hidden,
        };
    }

    fn outWidth(self: *const Proj) usize {
        return switch (self.kind) {
            .absorb => lat,
            .value => vw,
            .post => hidden,
        };
    }

    pub fn count(self: *const Proj) usize {
        return self.shape.rows * self.heads * self.outWidth();
    }

    pub fn prepare(self: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.heads = if (self.kind == .post) 1 else shape.n / self.outWidth();
        const planes: usize = if (self.kind == .post) self.world else 1;
        const xn = planes * shape.rows * self.heads * self.xWidth();
        self.x = try a.alloc(f32, xn);
        self.w = try a.alloc(u16, if (self.kind == .post) hidden else self.heads * kvh * lat);
        self.gate = try a.alloc(u16, if (self.kind == .value) shape.rows * self.heads * vw else 0);
        const amp: f32 = if (self.kind == .absorb) 2 else 1;
        for (self.x) |*v| v.* = (rng.float(f32) * 2 - 1) * amp;
        if (self.kind == .absorb) for (self.x) |*v| {
            v.* = c.fromBf16(c.toBf16(v.*));
        };
        for (self.w) |*v| v.* = c.toBf16(if (self.kind == .post) 0.5 + rng.float(f32) else (rng.float(f32) * 2 - 1) * 0.125);
        for (self.gate) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => self.stress(rng),
        }
        if (self.kind == .absorb) {
            const q = try a.alloc(u16, xn);
            defer a.free(q);
            for (q, self.x) |*h, v| h.* = c.toBf16(v);
            self.in[0] = try self.rig.input(std.mem.sliceAsBytes(q));
        } else self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        if (self.kind == .value) self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.gate));
        self.out = try self.rig.output(self.count() * c.size(spec(self.kind, self.world, .none).out));
        return true;
    }

    /// Row 0 a NaN, row 1 +Inf, row 2 both infinities, in head 0's reduced inputs (rank 0); value: row 3's gate NaN.
    fn plant(self: *Proj) void {
        const k = if (self.kind == .absorb) nope else self.xWidth();
        const xw = self.xWidth() * self.heads;
        const at = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 0, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
        for (at) |p| if (p[0] < self.shape.rows) {
            self.x[p[0] * xw + p[1]] = p[2];
        };
        if (self.kind == .value and self.shape.rows > 3) self.gate[3 * self.heads * vw + 5] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: absorb/value mirror weights and inputs across k/2 (gate 0); post sums integer squares past fp16 range.
    fn stress(self: *Proj, rng: std.Random) void {
        const rows = self.shape.rows;
        if (self.kind == .post) {
            for (self.x) |*v| v.* = 0;
            for (0..rows) |r| for (0..hidden) |i| {
                const v: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 91))) - 45);
                const share: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 9))) - 4);
                self.x[r * hidden + i] = if (self.world > 1) v - share else v;
                if (self.world > 1) self.x[(self.world - 1) * rows * hidden + r * hidden + i] = share;
            };
            return;
        }
        const k: usize = if (self.kind == .absorb) nope else lat;
        const h2 = k / 2;
        const outs = self.outWidth();
        for (0..self.heads) |hd| for (0..h2) |i| for (0..outs) |o| {
            const v = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 2)));
            self.w[self.wAt(hd, i, o)] = v;
            self.w[self.wAt(hd, h2 + i, o)] = v;
        };
        const odd: u32 = if (self.kind == .absorb) 64 else 2048;
        const base: u32 = if (self.kind == .absorb) 1 else 4097;
        const unit: f32 = if (self.kind == .absorb) 512 else 1;
        for (0..rows) |r| for (0..self.heads) |hd| for (0..h2) |i| {
            const v = unit * @as(f32, @floatFromInt(base + 2 * rng.uintLessThan(u32, odd)));
            const at = (r * self.heads + hd) * self.xWidth();
            self.x[at + i] = v;
            self.x[at + h2 + i] = if (i < 8) unit - v else -v;
        };
        for (self.gate) |*v| v.* = 0;
    }

    /// kv_b's weight for reduced index i and output o of head hd: absorb's key rows [d][c], value's [v][c].
    fn wAt(self: *const Proj, hd: usize, i: usize, o: usize) usize {
        return if (self.kind == .absorb) (hd * kvh + i) * lat + o else (hd * kvh + nope + o) * lat + i;
    }

    pub fn run(self: *Proj) !void {
        const s = self.rig.s;
        const rows = self.shape.rows;
        switch (self.kind) {
            .absorb => try hy4.absorb(self.f, s, self.in[0], self.heads * qh, self.in[1], self.out, rows, self.heads),
            .value => try hy4.value(self.f, s, self.in[0], self.in[1], self.in[2], self.heads * vw, self.out, self.heads * vw, rows, self.heads),
            .post => try hy4.post(self.f, s, self.in[0], self.world, self.in[1], self.out, rows, hidden, set.eps),
        }
        try s.synchronize();
    }

    /// The products output o sums, in order; its post-sum factor (sigmoid) in `factor`.
    fn terms(self: *const Proj, o: usize, t: []f64, factor: *f64) []f64 {
        const outs = self.outWidth();
        const col = o % outs;
        const rh = o / outs;
        const hd = rh % self.heads;
        const xa = rh * self.xWidth();
        const k: usize = if (self.kind == .absorb) nope else lat;
        for (t[0..k], 0..) |*v, i| v.* = @as(f64, self.x[xa + i]) * bf(self.w[self.wAt(hd, i, col)]);
        factor.* = 1;
        if (self.kind == .value) factor.* = 1 / (1 + @exp(-@as(f64, bf(self.gate[rh * vw + col]))));
        return t[0..k];
    }

    /// post's row r: x = bf16 of the ranks' fp32 sum in rank order (the stored o_proj output), into `x`.
    fn postRow(self: *const Proj, r: usize, x: []f64) void {
        const plane = self.shape.rows * hidden;
        for (x, 0..) |*v, i| {
            var s: f32 = self.x[r * hidden + i];
            for (1..self.world) |q| s += self.x[q * plane + r * hidden + i];
            v.* = c.fromBf16(c.toBf16(s));
        }
    }

    fn postOut(self: *const Proj, r: usize, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        var x: [hidden]f64 = undefined;
        var sq: [hidden]f64 = undefined;
        self.postRow(r, &x);
        for (&sq, x) |*q, v| q.* = v * v;
        const ss = if (acc) |k| sumIn(k, order, &sq) else sumIn(.none, .sequential, &sq);
        const inv = 1 / @sqrt(ss / hidden + @as(f64, set.eps));
        for (out[r * hidden ..][0..hidden], x, self.w, 0..) |*y, v, w, i| {
            y.* = v * inv * bf(w);
            if (mag) |m| m[r * hidden + i] = @abs(y.*);
        }
    }

    pub fn reference(self: *Proj, want: []f64, mag: []f64) !void {
        if (self.kind == .post) {
            for (0..self.shape.rows) |r| self.postOut(r, null, .sequential, want, mag);
            return;
        }
        var t: [lat]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            var factor: f64 = 1;
            const ts = self.terms(o, &t, &factor);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (ts) |v| {
                sum += v;
                abs += @abs(v);
            }
            w.* = sum * factor;
            m.* = abs * @abs(factor);
        }
    }

    pub fn simulate(self: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .post) {
            for (0..self.shape.rows) |r| self.postOut(r, acc, order, out, null);
            return;
        }
        var t: [lat]f64 = undefined;
        for (out, 0..) |*v, o| {
            var factor: f64 = 1;
            const ts = self.terms(o, &t, &factor);
            v.* = sumIn(acc, order, ts) * factor;
        }
    }
};
