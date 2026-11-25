//! Contract case for `hy4_ihc_fused`: fp64 accuracy, fp32 sums, and 0-bit row invariance across launch arms.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ihc_mod = @import("hy4_ihc_fused_cuda");
const hc_mod = ihc_mod.hc;
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const Ihc = ihc_mod.Ihc;
const Mode = ihc_mod.Mode;
const hidden: usize = ihc_mod.hidden;

fn chainDepth(_: c.Shape) usize {
    return (4 * hidden + 255) / 256 + 5 + 8 + 20;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = hidden, .k = 4 * hidden },
    .{ .rows = 2, .n = hidden, .k = 4 * hidden },
    .{ .rows = 5, .n = hidden, .k = 4 * hidden },
    .{ .rows = 16, .n = hidden, .k = 4 * hidden },
    .{ .rows = 17, .n = hidden, .k = 4 * hidden },
    .{ .rows = 65, .n = hidden, .k = 4 * hidden },
    .{ .rows = 100, .n = hidden, .k = 4 * hidden },
};

pub const Fault = enum { none, bad_nan, bad_sums };

pub const Case = struct {
    rig: rig_.Rig,
    ihc: *const Ihc,
    norms: *const cuda.norms.Norms,
    ctl_pre_fn: ?cuda.Function = null,
    mode: Mode,
    fault: Fault = .none,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    arena: std.heap.ArenaAllocator,
    g_in: []f32 = &.{},
    x_in: []u16 = &.{},
    post_prev: []f32 = &.{},
    post_nw: [hidden]u16 = @splat(0),
    next_nw: [hidden]u16 = @splat(0),
    fn_w: []f32 = &.{},
    base_w: [8]f32 = @splat(0),
    scale_w: [2]f32 = .{ 0.85, 1.15 },
    eps: f32 = 1e-5,
    hc_eps: f32 = 1e-4,
    mag: f32 = 2.0,
    dev_g: u64 = 0,
    dev_x: u64 = 0,
    dev_pp: u64 = 0,
    dev_pnw: u64 = 0,
    dev_nnw: u64 = 0,
    dev_m: hc_mod.Mix = undefined,
    dev_x_work: u64 = 0,
    dev_post_out: u64 = 0,
    dev_out: u64 = 0, // bf16 [rows, hidden] when mixesM > 0, else bf16 [rows, 4, hidden]

    pub fn deinit(self: *Case) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    fn hasPost(self: *const Case) bool {
        return self.mode == .boundary or self.mode == .head or self.mode == .post_only;
    }

    fn mixesM(self: *const Case) usize {
        return switch (self.mode) {
            .boundary, .entry => 8,
            .head, .head_only => 4,
            .post_only => 0,
        };
    }

    pub fn kernel(self: *Case) g.Kernel {
        const ctl: ?g.Gate = switch (self.fault) {
            .none => null,
            .bad_nan => .nonfinite,
            .bad_sums => .sums,
        };
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = chainDepth,
                .ops = 24,
                .out = .bf16,
                .tiles = .{ .{}, .{ .size = hidden, .step = hidden, .min = hidden }, .{ .size = 4 * hidden, .step = 4 * hidden, .min = 4 * hidden } },
                .served = &served,
            },
            .ctx = self,
            .vt = &vt,
            .control = ctl,
        };
    }

    const vt = g.hooks(Case);

    pub fn count(self: *const Case) usize {
        return self.shape.rows * (if (self.mixesM() > 0) hidden else 4 * hidden);
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n != hidden or shape.k != 4 * hidden) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.fill = fill;
        const a = self.arena.allocator();
        const r = shape.rows;
        const m = self.mixesM();
        const width = 4 * hidden;
        self.g_in = try a.alloc(f32, r * hidden);
        self.x_in = try a.alloc(u16, r * width);
        self.post_prev = try a.alloc(f32, r * 4);
        self.fn_w = try a.alloc(f32, @max(m, 1) * width);

        var prng = std.Random.DefaultPrng.init(seed ^ 0x4948);
        const rng = prng.random();
        if (fill == .stress and m > 0) {
            self.scale_w = .{ 100.0, 100.0 };
            self.base_w = @splat(0);
            for (self.g_in) |*v| v.* = 0.0;
            for (self.post_prev) |*v| v.* = 0.0;
            for (&self.post_nw) |*v| v.* = c.toBf16(1.0);
            for (&self.next_nw) |*v| v.* = c.toBf16(1.0);
            for (0..r) |ri| for (0..hidden) |h| {
                self.x_in[ri * width + h] = c.toBf16(8.0);
                self.x_in[ri * width + hidden + h] = c.toBf16(-8.0);
                self.x_in[ri * width + 2 * hidden + h] = if (h % 2 == 0) c.toBf16(0.5) else c.toBf16(-0.5);
                self.x_in[ri * width + 3 * hidden + h] = 0;
            };
            for (0..m) |j| {
                const sgn: f32 = if (j % 4 == 0) 1.0 else -1.0;
                for (0..width) |k| self.fn_w[j * width + k] = if (k < 2 * hidden) sgn * 8.0 else 0.0;
                self.fn_w[j * width + hidden] = sgn * 7.875;
            }
        } else if (fill == .stress) {
            const pp: f32 = 4096.0;
            var sy64: f64 = 0;
            for (0..hidden) |h| {
                const gv: f32 = if ((h * 17) % 64 == 0) 256.0 else 32.0;
                sy64 += @as(f64, gv) * @as(f64, gv);
                self.post_nw[h] = c.toBf16(1.0 + @as(f32, @floatFromInt(h % 64)) * 0.015625);
            }
            const rinv64 = 1.0 / @sqrt(sy64 / @as(f64, @floatFromInt(hidden)) + @as(f64, self.eps));
            for (0..r) |ri| for (0..hidden) |h| {
                const gv: f32 = if ((h * 17) % 64 == 0) 256.0 else 32.0;
                self.g_in[ri * hidden + h] = gv;
                const zh = c.fromBf16(c.toBf16(@as(f32, @floatCast(c.fromBf16(self.post_nw[h]) * (@as(f64, gv) * rinv64)))));
                const xneg = c.toBf16(@as(f32, @floatCast(-@as(f64, pp) * zh)));
                for (0..4) |st| self.x_in[ri * width + st * hidden + h] = xneg;
            };
            for (self.post_prev) |*v| v.* = pp;
            for (&self.next_nw) |*v| v.* = c.toBf16(1.0);
            for (self.fn_w) |*v| v.* = 0.0;
        } else {
            self.scale_w = .{ 0.85, 1.15 };
            for (&self.base_w, 0..) |*b, i| b.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.1;
            for (self.g_in) |*v| v.* = rng.float(f32) * 3.0 - 1.5;
            for (self.x_in) |*v| v.* = c.toBf16(rng.float(f32) * 3.0 - 1.5);
            for (self.post_prev) |*v| v.* = rng.float(f32) * 1.5 + 0.1;
            for (&self.post_nw) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 + 0.75);
            for (&self.next_nw) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 + 0.75);
            for (self.fn_w) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.2;
            if (fill == .nonfinite) {
                if (m == 0) self.g_in[0] = std.math.nan(f32) else self.x_in[0] = c.toBf16(std.math.nan(f32));
            }
        }

        self.dev_g = try self.rig.input(std.mem.sliceAsBytes(self.g_in));
        self.dev_pp = try self.rig.input(std.mem.sliceAsBytes(self.post_prev));
        self.dev_pnw = try self.rig.input(std.mem.sliceAsBytes(&self.post_nw));
        self.dev_nnw = try self.rig.input(std.mem.sliceAsBytes(&self.next_nw));
        self.dev_m = .{
            .fn_ = try self.rig.input(std.mem.sliceAsBytes(self.fn_w)),
            .base = try self.rig.input(std.mem.sliceAsBytes(self.base_w[0..@max(m, 1)])),
            .scale = try self.rig.input(std.mem.sliceAsBytes(self.scale_w[0..if (m == 4) 1 else 2])),
            .eps = self.eps,
            .hc_eps = self.hc_eps,
            .magnitude = self.mag,
        };
        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(self.x_in));
        self.dev_x_work = try self.rig.scratch(r * width * 2);
        self.dev_post_out = try self.rig.scratch(r * 4 * 4);
        self.dev_out = try self.rig.output(self.count() * 2);
        return true;
    }

    fn launchIhc(self: *Case, r: usize, g_ptr: u64, x_src: u64, x_dst: u64, pp_ptr: u64, a_ptr: u64, pn_ptr: u64) !void {
        const x_bytes = r * 4 * hidden * 2;
        try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(x_dst, x_src, x_bytes, self.rig.s.handle), "cuMemcpyDtoDAsync");
        try self.ihc.run(self.rig.s, .{
            .mode = self.mode,
            .rows = r,
            .g = g_ptr,
            .x = x_dst,
            .post_prev = pp_ptr,
            .post_nw = self.dev_pnw,
            .mix = if (self.mixesM() > 0) self.dev_m else null,
            .next_nw = self.dev_nnw,
            .a = a_ptr,
            .post_next = pn_ptr,
            .eps = self.eps,
            .hc_eps = self.hc_eps,
            .mag = self.mag,
        });
    }

    pub fn run(self: *Case) !void {
        const r = self.shape.rows;
        const m = self.mixesM();
        if (self.fault != .none) {
            const x_dst = self.dev_x_work;
            try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(x_dst, self.dev_x, r * 4 * hidden * 2, self.rig.s.handle), "cuMemcpyDtoDAsync");
            if (self.fault == .bad_nan) {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = x_dst, .len = r * 4 * hidden * 2 }).fill8(0, self.rig.s.handle);
            }
            var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, r * hidden * 2);
            defer tmp.free();
            const pre_fn = self.ctl_pre_fn orelse return error.MissingControlFunction;
            try hc_mod.launchMix(pre_fn, self.rig.s, self.dev_m, x_dst, tmp.ptr, self.dev_post_out, r, hidden, 0);
            try self.norms.addRmsTail(self.rig.s, tmp.ptr, null, self.dev_nnw, 0, self.dev_out, null, r, hidden, self.eps, .{ .round = true });
            try self.rig.s.synchronize();
            return;
        }
        const x_dst = if (m == 0) self.dev_out else self.dev_x_work;
        try self.launchIhc(r, self.dev_g, self.dev_x, x_dst, self.dev_pp, if (m > 0) self.dev_out else 0, if (m == 8) self.dev_post_out else 0);
        try self.rig.s.synchronize();
    }

    fn evalRow(self: *const Case, r: usize, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const width = 4 * hidden;
        const m = self.mixesM();
        var xp: [width]f64 = undefined;
        for (0..width) |k| xp[k] = c.fromBf16(self.x_in[r * width + k]);

        if (self.hasPost()) {
            const terms_y = try a.alloc(f64, hidden);
            defer a.free(terms_y);
            var y_bf: [hidden]f64 = undefined;
            for (0..hidden) |h| {
                const yb: f64 = c.fromBf16(c.toBf16(self.g_in[r * hidden + h]));
                y_bf[h] = yb;
                terms_y[h] = yb * yb;
            }
            const sy = if (acc) |ac| rows_.sumIn(ac, order, terms_y) else blk: {
                var s: f64 = 0;
                for (terms_y) |t| s += t;
                break :blk s;
            };
            const rinv_y = 1.0 / @sqrt(sy / @as(f64, @floatFromInt(hidden)) + @as(f64, self.eps));
            for (0..hidden) |h| {
                const z = c.fromBf16(c.toBf16(@as(f32, @floatCast(c.fromBf16(self.post_nw[h]) * (y_bf[h] * rinv_y)))));
                for (0..4) |st| {
                    const prev = xp[st * hidden + h];
                    const upd = @as(f64, self.post_prev[r * 4 + st]) * z + prev;
                    xp[st * hidden + h] = if (acc != null) c.fromBf16(c.toBf16(@as(f32, @floatCast(upd)))) else upd;
                    if (m == 0) {
                        want[r * width + st * hidden + h] = upd;
                        mag[r * width + st * hidden + h] = @abs(@as(f64, self.post_prev[r * 4 + st]) * z) + @abs(prev) + 1e-3;
                    }
                }
            }
            if (m == 0) return;
        }

        const terms = try a.alloc(f64, width);
        defer a.free(terms);
        for (0..width) |k| terms[k] = xp[k] * xp[k];
        const sq = if (acc) |ac| rows_.sumIn(ac, order, terms) else blk: {
            var s: f64 = 0;
            for (terms) |t| s += t;
            break :blk s;
        };
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + @as(f64, self.eps));
        var pre: [4]f64 = undefined;
        var wmag: f64 = 0;
        for (0..4) |i| {
            var mdot: f64 = 0;
            for (0..width) |k| {
                const p = xp[k] * @as(f64, self.fn_w[i * width + k]);
                terms[k] = p;
                mdot += @abs(p);
            }
            const dot = if (acc) |ac| rows_.sumIn(ac, order, terms) else blk: {
                var s: f64 = 0;
                for (terms) |t| s += t;
                break :blk s;
            };
            const z = (dot * inv) * @as(f64, self.scale_w[0]) + @as(f64, self.base_w[i]);
            const e = @exp(-@abs(z));
            const sig = if (z < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
            pre[i] = sig + @as(f64, self.hc_eps);
            wmag += 0.25 * (mdot * inv + @abs(dot * inv) * 0.5) * @abs(@as(f64, self.scale_w[0]));
        }

        var o_row: [hidden]f64 = undefined;
        var omag: [hidden]f64 = undefined;
        const terms_o = try a.alloc(f64, hidden);
        defer a.free(terms_o);
        for (0..hidden) |h| {
            var s: f64 = 0;
            var xmax: f64 = 0;
            var smag: f64 = 0;
            for (0..4) |st| {
                const xv = xp[st * hidden + h];
                s += pre[st] * xv;
                xmax = @max(xmax, @abs(xv));
                smag += @abs(pre[st] * xv);
            }
            const o_val = if (acc != null) c.fromBf16(c.toBf16(@as(f32, @floatCast(s)))) else s;
            o_row[h] = o_val;
            omag[h] = wmag * xmax + smag + 1e-3;
            terms_o[h] = o_val * o_val;
        }
        const so = if (acc) |ac| rows_.sumIn(ac, order, terms_o) else blk: {
            var s: f64 = 0;
            for (terms_o) |t| s += t;
            break :blk s;
        };
        const rinv_o = 1.0 / @sqrt(so / @as(f64, @floatFromInt(hidden)) + @as(f64, self.eps));
        for (0..hidden) |h| {
            const nw: f64 = c.fromBf16(self.next_nw[h]);
            const normed = if (acc != null) c.fromBf16(c.toBf16(@as(f32, @floatCast(nw * (o_row[h] * rinv_o))))) else nw * (o_row[h] * rinv_o);
            want[r * hidden + h] = normed;
            mag[r * hidden + h] = @abs(nw) * omag[h] * rinv_o * 2.0 + @abs(normed);
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| try self.evalRow(r, null, .sequential, want, mag);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.gpu.gpa.alloc(f64, out.len);
        defer self.rig.gpu.gpa.free(mag);
        for (0..self.shape.rows) |r| try self.evalRow(r, acc, order, out, mag);
    }

    /// Gate 3 checks 0-bit row invariance of `R = 2, 5, 16, 17, 65, 100` against solo `R = 1` ticket.
    pub fn check(self: *Case, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const m = self.mixesM();
        const row_elems = if (m > 0) hidden else 4 * hidden;
        const row_bytes = row_elems * 2;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (gate == .nonfinite) {
            const words: []align(1) const u16 = std.mem.bytesAsSlice(u16, got);
            var row0_nan = false;
            for (words[0..row_elems]) |w| if (!std.math.isFinite(c.fromBf16(w))) {
                row0_nan = true;
                break;
            };
            if (!row0_nan) return "hy4_ihc_fused dropped non-finite input on row 0";
            for (words[row_elems..]) |w| if (!std.math.isFinite(c.fromBf16(w))) return "hy4_ihc_fused leaked NaN across rows";
            return null;
        }
        if (gate != .shapes) return null;
        const n = self.count();
        const want = try a.alloc(f64, n);
        defer a.free(want);
        const mag = try a.alloc(f64, n);
        defer a.free(mag);
        try self.reference(want, mag);
        const words: []align(1) const u16 = std.mem.bytesAsSlice(u16, got);
        for (words, want, mag) |w, ref_v, mg| {
            const err = @abs(c.fromBf16(w) - ref_v);
            const allow = @max(@abs(ref_v), mg) * 3e-2 + 2e-2;
            if (!(err <= allow)) return "hy4_ihc_fused output exceeds fp64 bf16 error bound";
        }
        if (self.fault != .none or self.shape.rows <= 1) return null;

        var solo_x = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, 4 * hidden * 2);
        defer solo_x.free();
        var solo_a = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, row_bytes);
        defer solo_a.free();
        var solo_p = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, 16);
        defer solo_p.free();
        const solo_host = try a.alloc(u8, row_bytes);
        defer a.free(solo_host);

        const check_rows = [_]usize{ 0, self.shape.rows / 2, self.shape.rows - 1 };
        for (check_rows) |r| {
            const x_dst = if (m == 0) solo_a.ptr else solo_x.ptr;
            try self.launchIhc(
                1,
                self.dev_g + r * hidden * 4,
                self.dev_x + r * 4 * hidden * 2,
                x_dst,
                self.dev_pp + r * 16,
                if (m > 0) solo_a.ptr else 0,
                if (m == 8) solo_p.ptr else 0,
            );
            try self.rig.s.synchronize();
            try solo_a.download(0, solo_host);
            if (!std.mem.eql(u8, got[r * row_bytes ..][0..row_bytes], solo_host))
                return "hy4_ihc_fused multi-row output differs from R=1 solo launch";
        }
        return null;
    }
};
