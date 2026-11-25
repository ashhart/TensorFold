//! hy4_flash/index_paged.cu query M-RoPE and mma.m16n8k16 16-head ReLU-weighted pool score kernels under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_index_paged");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_index_paged.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;

const qmrope_served = [_]c.Shape{ .{ .rows = 8, .n = hy4.heads * hy4.dim, .k = 1 }, .{ .rows = 1, .n = hy4.heads * hy4.dim, .k = 1 }, .{ .rows = 64, .n = hy4.heads * hy4.dim, .k = 1 } };

pub const QMrope = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fault: set.Fault = .none,
    shape: c.Shape = undefined,
    iq: []u16 = &.{},
    pos: []i32 = &.{},
    cs: []f32 = &.{},
    in: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *QMrope) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *QMrope) void {
        const a = self.rig.gpu.gpa;
        a.free(self.iq);
        a.free(self.pos);
        a.free(self.cs);
        self.iq = &.{};
        self.pos = &.{};
        self.cs = &.{};
    }

    pub fn kernel(self: *QMrope) g.Kernel {
        var s: c.Spec = .{
            .name = "cuda hy4_index_paged tf_hy4_idx_q_mrope (trailing 64-dim interleaved RoPE)",
            .accum = .none,
            .depth = zeroDepth,
            .ops = 2,
            .out = .bf16,
            .served = &qmrope_served,
        };
        if (self.fault != .none) s.name = set.controlName("q_mrope", self.fault);
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(QMrope);

    fn zeroDepth(_: c.Shape) usize {
        return 0;
    }

    pub fn count(self: *const QMrope) usize {
        return self.shape.rows * hy4.heads * hy4.dim;
    }

    pub fn prepare(self: *QMrope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hy4.heads * hy4.dim) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.iq = try a.alloc(u16, rows * hy4.heads * hy4.dim);
        self.pos = try a.alloc(i32, rows);
        const max_pos = 2100 + rows;
        self.cs = try a.alloc(f32, max_pos * hy4.rope_pairs * 2);
        for (self.iq) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (self.pos, 0..) |*p, r| p.* = @intCast(2048 + r);
        const freqs = cuda.mla.ropeFreqs(1e7);
        for (0..max_pos) |p| for (0..hy4.rope_pairs) |j| {
            const ang = @as(f64, @floatFromInt(p)) * @as(f64, freqs[j]);
            self.cs[(p * hy4.rope_pairs + j) * 2] = @floatCast(@cos(ang));
            self.cs[(p * hy4.rope_pairs + j) * 2 + 1] = @floatCast(@sin(ang));
        };
        if (fill == .nonfinite) {
            const w = hy4.heads * hy4.dim;
            self.iq[0 * w + 5] = c.toBf16(std.math.nan(f32));
            self.iq[0 * w + 126] = c.toBf16(std.math.nan(f32));
            if (rows > 1) self.iq[1 * w + 7] = c.toBf16(std.math.inf(f32));
            if (rows > 2) self.iq[2 * w + 9] = c.toBf16(-std.math.inf(f32));
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.iq));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.cs));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *QMrope) !void {
        try hy4.qMrope(self.f, self.rig.s, 0, self.shape.rows, self.in[0], self.in[1], 0, self.shape.rows, self.in[2], self.out);
        try self.rig.s.synchronize();
    }

    pub fn reference(self: *QMrope, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| {
            const p: usize = @intCast(self.pos[r]);
            for (0..hy4.heads) |h| {
                const at = (r * hy4.heads + h) * hy4.dim;
                for (0..64) |d| {
                    want[at + d] = c.fromBf16(self.iq[at + d]);
                    mag[at + d] = @abs(want[at + d]);
                }
                for (0..hy4.rope_pairs) |pr| {
                    const x0: f64 = c.fromBf16(self.iq[at + 64 + 2 * pr]);
                    const x1: f64 = c.fromBf16(self.iq[at + 64 + 2 * pr + 1]);
                    const cv: f64 = self.cs[(p * hy4.rope_pairs + pr) * 2];
                    const sv: f64 = self.cs[(p * hy4.rope_pairs + pr) * 2 + 1];
                    want[at + 64 + 2 * pr] = x0 * cv - x1 * sv;
                    want[at + 64 + 2 * pr + 1] = x1 * cv + x0 * sv;
                    const mm = @abs(x0 * cv) + @abs(x1 * sv);
                    mag[at + 64 + 2 * pr] = mm;
                    mag[at + 64 + 2 * pr + 1] = mm;
                }
            }
        }
    }
};

pub const Kind = enum { decode, decode_rope, prefill };

fn scoreDepth(_: c.Shape) usize {
    return hy4.dim + hy4.heads;
}

const score_served = [_]c.Shape{ .{ .rows = 8, .n = 128, .k = hy4.dim }, .{ .rows = 1, .n = 64, .k = hy4.dim }, .{ .rows = 4, .n = 65, .k = hy4.dim }, .{ .rows = 17, .n = 257, .k = hy4.dim } };

pub const Score = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    q_mrope: cuda.Function,
    base_decode: cuda.Function,
    fault: set.Fault = .none,
    fill: g.Fill = .random,
    shape: c.Shape = undefined,
    iq: []u16 = &.{},
    hw: []f32 = &.{},
    pos: []i32 = &.{},
    gkc: []u16 = &.{},
    cs: []f32 = &.{},
    in: [5]u64 = @splat(0),
    out: u64 = 0,
    qi: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Score) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Score) void {
        const a = self.rig.gpu.gpa;
        a.free(self.iq);
        a.free(self.hw);
        a.free(self.pos);
        a.free(self.gkc);
        a.free(self.cs);
        self.iq = &.{};
        self.hw = &.{};
        self.pos = &.{};
        self.gkc = &.{};
        self.cs = &.{};
    }

    pub fn kernel(self: *Score) g.Kernel {
        const name = switch (self.kind) {
            .decode => "cuda hy4_index_paged tf_hy4_idx_score_decode (mma.m16n8k16 16-head relu score)",
            .decode_rope => "cuda hy4_index_paged tf_hy4_idx_score_decode_rope (fused M-RoPE + score)",
            .prefill => "cuda hy4_index_paged tf_hy4_idx_score_prefill (8 rows/CTA mma.m16n8k16 score)",
        };
        const rep: ?[]const u8 = if (self.fault != .none) null else switch (self.kind) {
            .decode => null,
            .decode_rope => "tf_hy4_idx_q_mrope + tf_hy4_idx_score_decode",
            .prefill => "tf_hy4_idx_score_decode",
        };
        const cname = switch (self.kind) {
            .decode => "score_decode",
            .decode_rope => "score_decode_rope",
            .prefill => "score_prefill",
        };
        const s: c.Spec = .{
            .name = if (self.fault == .none) name else set.controlName(cname, self.fault),
            .accum = .f32,
            .depth = scoreDepth,
            .ops = 4,
            .out = .f32,
            .replaces = rep,
            .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 64, .step = 1, .min = 1 }, .{} },
            .served = &score_served,
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(Score);

    pub fn count(self: *const Score) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn prepare(self: *Score, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != hy4.dim) return false;
        self.free();
        self.rig.reset();
        self.fill = fill;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const own = shape.n;
        const gpool = own + 16;
        const max_pos = 4 * own + 4;
        self.iq = try a.alloc(u16, rows * hy4.heads * hy4.dim);
        self.hw = try a.alloc(f32, rows * hy4.heads);
        self.pos = try a.alloc(i32, rows);
        self.gkc = try a.alloc(u16, gpool * hy4.dim);
        self.cs = try a.alloc(f32, max_pos * hy4.rope_pairs * 2);
        for (self.pos, 0..) |*p, r| p.* = @intCast(4 * own + (r % 4));
        const freqs = cuda.mla.ropeFreqs(1e7);
        for (0..max_pos) |p| for (0..hy4.rope_pairs) |j| {
            const ang = @as(f64, @floatFromInt(p)) * @as(f64, freqs[j]);
            self.cs[(p * hy4.rope_pairs + j) * 2] = @floatCast(@cos(ang));
            self.cs[(p * hy4.rope_pairs + j) * 2 + 1] = @floatCast(@sin(ang));
        };
        if (fill == .stress) {
            @memset(self.iq, 0);
            @memset(self.gkc, 0);
            @memset(self.hw, 1.0);
            for (0..rows) |r| for (0..hy4.heads) |h| {
                const row_q = self.iq[(r * hy4.heads + h) * hy4.dim ..];
                for (0..8) |i| row_q[i] = c.toBf16(1.0);
                for (16..32) |i| row_q[i] = c.toBf16(64.0);
                for (32..48) |i| row_q[i] = c.toBf16(-64.0);
            };
            for (0..own) |gg| {
                const row_k = self.gkc[gg * hy4.dim ..];
                for (0..8) |i| row_k[i] = c.toBf16(1.0);
                for (16..48) |i| row_k[i] = c.toBf16(128.0);
            }
        } else {
            for (self.iq) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (self.hw) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
            for (self.gkc[0 .. own * hy4.dim]) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            if (fill == .nonfinite) {
                self.iq[0 * hy4.heads * hy4.dim + 5] = c.toBf16(std.math.nan(f32));
                if (rows > 1) {
                    for (0..hy4.dim) |d| self.gkc[0 * hy4.dim + d] = c.toBf16(1.0);
                    self.iq[1 * hy4.heads * hy4.dim + 0] = c.toBf16(std.math.inf(f32));
                    self.hw[1 * hy4.heads + 0] = 0.5;
                }
                if (rows > 2) {
                    self.iq[2 * hy4.heads * hy4.dim + 0] = c.toBf16(std.math.inf(f32));
                    self.iq[2 * hy4.heads * hy4.dim + 1 * hy4.dim + 0] = c.toBf16(std.math.inf(f32));
                    self.hw[2 * hy4.heads + 0] = 0.5;
                    self.hw[2 * hy4.heads + 1] = -0.5;
                }
                if (rows > 3) {
                    self.iq[3 * hy4.heads * hy4.dim + 0] = c.toBf16(std.math.inf(f32));
                    self.hw[3 * hy4.heads + 0] = -0.5;
                }
            }
        }
        // Unlisted group keys g >= own carry NaN; any readPastOwn leaks into finite rows.
        for (self.gkc[own * hy4.dim ..]) |*v| v.* = c.toBf16(std.math.nan(f32));
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.iq));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.hw));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.gkc));
        self.in[4] = try self.rig.input(std.mem.sliceAsBytes(self.cs));
        self.out = try self.rig.output(self.count() * 4);
        self.qi = try self.rig.scratch(rows * hy4.heads * hy4.dim * 2);
        self.base = try self.rig.scratch(self.count() * 4);
        return true;
    }

    fn launch(self: *const Score, f: cuda.Function, kind: Kind, out: u64) !void {
        const rows = self.shape.rows;
        const own = self.shape.n;
        const gpool = own + 16;
        const rb: usize = if (kind == .prefill) 8 else 4;
        const nb = (rows + rb - 1) / rb;
        const split = hy4.scoreSplit(nb, own, 48, 3);
        const bx: u32 = if (kind == .prefill) 256 else 128;
        if (kind == .decode_rope) {
            var a: cuda.Args = .{};
            inline for (.{ @as(u64, 0), @as(c_int, @intCast(rows)), self.in[0], self.in[1], @as(u64, 0), self.in[2], self.in[3], @as(c_int, @intCast(gpool)), out, @as(c_int, @intCast(own)), self.in[4], @as(u64, 0), @as(c_int, @intCast(rows)) }) |v| a.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = split, .y = @intCast(nb) }, .block = .{ .x = bx } }, self.rig.s, &a);
        } else {
            var a: cuda.Args = .{};
            inline for (.{ @as(u64, 0), @as(c_int, @intCast(rows)), self.in[0], self.in[1], @as(u64, 0), self.in[2], self.in[3], @as(c_int, @intCast(gpool)), out, @as(c_int, @intCast(own)) }) |v| a.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = split, .y = @intCast(nb) }, .block = .{ .x = bx } }, self.rig.s, &a);
        }
    }

    pub fn run(self: *Score) !void {
        try self.launch(self.f, self.kind, self.out);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Score) ![]const u8 {
        if (self.kind == .decode_rope) {
            try hy4.qMrope(self.q_mrope, self.rig.s, 0, self.shape.rows, self.in[0], self.in[2], 0, self.shape.rows, self.in[4], self.qi);
            const own = self.shape.n;
            const gpool = own + 16;
            const nb = (self.shape.rows + 3) / 4;
            const split = hy4.scoreSplit(nb, own, 48, 3);
            var a: cuda.Args = .{};
            inline for (.{ @as(u64, 0), @as(c_int, @intCast(self.shape.rows)), self.qi, self.in[1], @as(u64, 0), self.in[2], self.in[3], @as(c_int, @intCast(gpool)), self.base, @as(c_int, @intCast(own)) }) |v| a.add(v);
            try cuda.launch.launch(self.base_decode, .{ .grid = .{ .x = split, .y = @intCast(nb) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        } else {
            try self.launch(self.base_decode, .decode, self.base);
        }
        return self.rig.bytes(self.base, self.count() * 4);
    }

    fn eval(self: *const Score, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        var dots: [hy4.dim]f64 = undefined;
        var hterms: [hy4.heads]f64 = undefined;
        for (0..self.shape.rows) |r| {
            const p: usize = @intCast(self.pos[r]);
            for (0..self.shape.n) |gg| {
                var mm: f64 = 0;
                for (0..hy4.heads) |h| {
                    const qa = (r * hy4.heads + h) * hy4.dim;
                    var abs: f64 = 0;
                    for (0..64) |d| {
                        const x = c.fromBf16(self.iq[qa + d]) * c.fromBf16(self.gkc[gg * hy4.dim + d]);
                        dots[d] = x;
                        abs += @abs(x);
                    }
                    for (0..hy4.rope_pairs) |pr| {
                        const q0: f64 = c.fromBf16(self.iq[qa + 64 + 2 * pr]);
                        const q1: f64 = c.fromBf16(self.iq[qa + 64 + 2 * pr + 1]);
                        var rq0 = q0;
                        var rq1 = q1;
                        if (self.kind == .decode_rope) {
                            const cv: f64 = self.cs[(p * hy4.rope_pairs + pr) * 2];
                            const sv: f64 = self.cs[(p * hy4.rope_pairs + pr) * 2 + 1];
                            rq0 = c.fromBf16(c.toBf16(@floatCast(q0 * cv - q1 * sv)));
                            rq1 = c.fromBf16(c.toBf16(@floatCast(q1 * cv + q0 * sv)));
                        }
                        const x0 = rq0 * c.fromBf16(self.gkc[gg * hy4.dim + 64 + 2 * pr]);
                        const x1 = rq1 * c.fromBf16(self.gkc[gg * hy4.dim + 64 + 2 * pr + 1]);
                        dots[64 + 2 * pr] = x0;
                        dots[64 + 2 * pr + 1] = x1;
                        abs += @abs(x0) + @abs(x1);
                    }
                    const dot = if (acc) |ka| sumIn(ka, order, &dots) else sumIn(.none, .sequential, &dots);
                    const relu = if (std.math.isNan(dot)) dot else if (dot < 0) 0 else dot;
                    const w: f64 = self.hw[r * hy4.heads + h];
                    hterms[h] = w * relu;
                    mm += if (std.math.isFinite(abs)) @abs(w) * abs else 0;
                }
                const idx = r * self.shape.n + gg;
                out[idx] = if (acc) |ka| sumIn(ka, order, &hterms) else sumIn(.none, .sequential, &hterms);
                if (mag) |m| m[idx] = mm;
            }
        }
    }

    pub fn reference(self: *Score, want: []f64, mag: []f64) !void {
        self.eval(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Score, acc: c.Accum, order: g.Order, out: []f64) !void {
        self.eval(acc, order, out, null);
    }
};
