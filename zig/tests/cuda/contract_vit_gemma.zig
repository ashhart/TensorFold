//! vit.cu's Gemma 4 tower glue under the contract: q/k/v norms with the 2D RoPE, torch's GeGLU, 3 x 3 pooling.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;

pub const Kind = enum { qkv, gelu, pool };

/// A registration: a served kernel (keep null), or a control keeping all of its points but one.
pub const Entry = struct { name: []const u8, kind: Kind, keep: ?i32 = null };

pub const entries = [_]Entry{
    .{ .name = "cuda vit tf_vit_g4_qkv (Gemma 4 tower: q/k/v RMSNorm, 2D RoPE, in place)", .kind = .qkv },
    .{ .name = "control: tower qkv without the normed row's rounding", .kind = .qkv, .keep = 2 },
    .{ .name = "control: tower qkv without the RoPE products' rounding", .kind = .qkv, .keep = 1 },
    .{ .name = "cuda vit tf_vit_g4_gelu_mul (torch's GeGLU on bf16)", .kind = .gelu },
    .{ .name = "control: tower GeGLU without the gelu's rounding", .kind = .gelu, .keep = 0 },
    .{ .name = "cuda vit tf_vit_g4_pool (3 x 3 means, standardized)", .kind = .pool },
    .{ .name = "control: tower pooling without the block mean's rounding", .kind = .pool, .keep = 0 },
};

const heads = 16;
const dim = 72;
const width = heads * dim;
const eps: f32 = 1e-6;

/// The kernels' own float32 steps on the host: the qkv rows' norm sum in the warp's butterfly order.
fn qkvDepth(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    const name = e.name;
    return switch (e.kind) {
        // the 31B's patches: up to 2,520 (280 tokens x 9), a 16 x 16 grid, one row
        .qkv => .{ .name = name, .accum = .none, .depth = qkvDepth, .out = .raw, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{ .{ .rows = 2520, .n = 3 * width, .k = dim }, .{ .rows = 256, .n = 3 * width, .k = dim }, .{ .rows = 9, .n = 3 * width, .k = dim } } },
        .gelu => .{ .name = name, .accum = .none, .depth = qkvDepth, .out = .raw, .served = &.{ .{ .rows = 2520, .n = 4304, .k = 1 }, .{ .rows = 9, .n = 4304, .k = 1 } } },
        // tokens (rows) and the grid's blocks across (k): one block, the processor's 42 x 57 patches (266), 280 x 1
        .pool => .{ .name = name, .accum = .none, .depth = qkvDepth, .out = .raw, .served = &.{ .{ .rows = 1, .n = width, .k = 1 }, .{ .rows = 266, .n = width, .k = 19 }, .{ .rows = 280, .n = width, .k = 1 } } },
    };
}

fn f(v: u16) f32 {
    return bfr.value(v);
}

fn r(v: f32) f32 {
    return bfr.value(bfr.bits(v));
}

pub const Glue = struct {
    rig: rig_.Rig,
    entry: Entry,
    vit: *const cuda.vit.Kernels,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    in: [5][]const u16 = .{ &.{}, &.{}, &.{}, &.{}, &.{} },
    dev: [6]u64 = @splat(0),
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Glue) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Glue);

    pub fn kernel(self: *Glue) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    fn fill(rng: std.Random, v: []u16, scale: f32) void {
        for (v, 0..) |*x, i| x.* = bfr.bits((rng.float(f32) * 2 - 1) * scale * (if (i % 41 == 5) @as(f32, 12) else 1));
    }

    pub fn prepare(self: *Glue, shape: c.Shape, fl: g.Fill, seed: u64) !bool {
        switch (self.entry.kind) {
            .qkv => if (shape.k != dim or shape.n != 3 * width) return false,
            .gelu => {},
            .pool => if (shape.k == 0 or shape.rows % shape.k != 0 or shape.n != width) return false,
        }
        if (shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows * shape.n;
        switch (self.entry.kind) {
            .qkv => {
                const x = try a.alloc(u16, n);
                fill(rng, x, 3);
                const qw = try a.alloc(u16, dim);
                const kw = try a.alloc(u16, dim);
                fill(rng, qw, 2);
                fill(rng, kw, 2);
                const cs = try a.alloc(u16, shape.rows * dim);
                const sn = try a.alloc(u16, shape.rows * dim);
                for (cs, sn, 0..) |*cv, *sv, i| {
                    const ang = rng.float(f32) * 40 + @as(f32, @floatFromInt(i % 7));
                    cv.* = bfr.bits(@cos(ang));
                    sv.* = bfr.bits(@sin(ang));
                }
                if (fl == .nonfinite) x[3 * width + 5] = bfr.bits(std.math.nan(f32));
                self.in = .{ x, qw, kw, cs, sn };
                for (self.in, 0..) |t, i| self.dev[i] = try self.rig.input(std.mem.sliceAsBytes(t));
                self.dev[5] = try self.rig.output(n * 2);
            },
            .gelu => {
                const gate = try a.alloc(u16, n);
                const up = try a.alloc(u16, n);
                fill(rng, gate, 4);
                fill(rng, up, 2);
                if (fl == .nonfinite) gate[n / 2] = bfr.bits(std.math.inf(f32));
                self.in = .{ gate, up, &.{}, &.{}, &.{} };
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(gate));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(up));
                self.dev[5] = try self.rig.output(n * 2);
            },
            .pool => {
                const x = try a.alloc(u16, 9 * shape.rows * width);
                fill(rng, x, 2);
                const bias = try a.alloc(u16, width);
                const scale = try a.alloc(u16, width);
                fill(rng, bias, 30);
                for (scale) |*v| v.* = bfr.bits(0.02 + rng.float(f32) * 0.2);
                if (fl == .nonfinite) x[7] = bfr.bits(std.math.nan(f32));
                self.in = .{ x, bias, scale, &.{}, &.{} };
                for (self.in[0..3], 0..) |t, i| self.dev[i] = try self.rig.input(std.mem.sliceAsBytes(t));
                self.dev[5] = try self.rig.output(shape.rows * width * 2);
            },
        }
        return true;
    }

    pub fn run(self: *Glue) !void {
        const s = self.rig.s;
        const sh = self.shape;
        var args: cuda.Args = .{};
        switch (self.entry.kind) {
            .qkv => {
                // in place: the kernel works on a copy of the input rows in the output buffer
                const d = self.rig.gpu.d;
                try d.check(d.api.cuMemcpyDtoDAsync_v2(self.dev[5], self.dev[0], sh.rows * sh.n * 2, s.handle), "cuMemcpyDtoDAsync");
                if (self.ctl) |fun| {
                    for ([_]u64{ self.dev[5], self.dev[1], self.dev[2], self.dev[3], self.dev[4] }) |p| args.add(p);
                    for ([_]usize{ sh.rows, heads, dim }) |v| args.add(@as(c_int, @intCast(v)));
                    args.add(eps);
                    args.add(@as(c_int, self.entry.keep.?));
                    try cuda.launch.launch(fun, .{ .grid = .{ .x = @intCast(sh.rows * heads * 3), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &args);
                } else try self.vit.g4Qkv(s, self.dev[5], self.dev[1], self.dev[2], self.dev[3], self.dev[4], sh.rows, heads, dim, eps);
            },
            .gelu => {
                const n = sh.rows * sh.n;
                if (self.ctl) |fun| {
                    for ([_]u64{ self.dev[0], self.dev[1], self.dev[5] }) |p| args.add(p);
                    args.add(@as(i64, @intCast(n)));
                    args.add(@as(c_int, self.entry.keep.?));
                    try cuda.launch.launch(fun, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
                } else try self.vit.g4GeluMul(s, self.dev[0], self.dev[1], self.dev[5], n);
            },
            .pool => {
                const gw = 3 * sh.k;
                const gh = 9 * sh.rows / gw;
                if (self.ctl) |fun| {
                    for ([_]u64{ self.dev[0], self.dev[1], self.dev[2], self.dev[5] }) |p| args.add(p);
                    args.add(@as(c_int, @intCast(gw)));
                    args.add(@as(c_int, width));
                    args.add(@as(f32, @floatCast(@sqrt(@as(f64, width)))));
                    args.add(@as(c_int, self.entry.keep.?));
                    try cuda.launch.launch(fun, .{ .grid = .{ .x = @intCast(gh * gw / 9), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
                } else try self.vit.g4Pool(s, self.dev[0], self.dev[1], self.dev[2], self.dev[5], gh, gw, width);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Glue) usize {
        const sh = self.shape;
        return 2 * sh.rows * sh.n;
    }

    pub fn reference(_: *Glue, out: []f64, mag: []f64) !void {
        @memset(out, 0);
        @memset(mag, 0);
    }

    /// Every output as the host's float32 steps give it, bit for bit; the GeGLU within tanhf's ulps.
    pub fn check(self: *Glue, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u16, try self.rig.bytes(self.dev[5], self.count()));
        return switch (self.entry.kind) {
            .qkv => self.checkQkv(got),
            .gelu => self.checkGelu(got),
            .pool => self.checkPool(got),
        };
    }

    fn same(want: f32, got: u16) bool {
        const w = bfr.bits(want);
        if (std.math.isNan(want)) return std.math.isNan(f(got));
        return w == got;
    }

    fn checkQkv(self: *Glue, got: []align(1) const u16) !?[]const u8 {
        const x = self.in[0];
        for (0..self.shape.rows) |row| for (0..3) |part| for (0..heads) |h| {
            const at = (row * 3 + part) * width + h * dim;
            // the warp's three values a lane, summed in order, then the butterfly
            var lanes: [32]f32 = @splat(0);
            for (0..32) |lane| for (0..3) |i| {
                const d = lane + 32 * i;
                const v: f32 = if (d < dim) f(x[at + d]) else 0;
                lanes[lane] = lanes[lane] + v * v;
            };
            var off: usize = 16;
            while (off > 0) : (off >>= 1) {
                var next: [32]f32 = undefined;
                for (0..32) |l| next[l] = lanes[l] + lanes[l ^ off];
                lanes = next;
            }
            const inv = 1.0 / @sqrt(lanes[0] / @as(f32, dim) + eps);
            var nrm: [dim]f32 = undefined;
            const w = if (part == 0) self.in[1] else self.in[2];
            for (0..dim) |d| nrm[d] = r(if (part == 2) f(x[at + d]) * inv else f(x[at + d]) * inv * f(w[d]));
            for (0..dim) |d| {
                var want = nrm[d];
                if (part != 2) {
                    const half = dim / 2;
                    const q = half / 2;
                    const base = d / half * half;
                    const j = d % half;
                    const other = if (j < q) -nrm[base + j + q] else nrm[base + j - q];
                    want = r(nrm[d] * f(self.in[3][row * dim + d])) + r(other * f(self.in[4][row * dim + d]));
                }
                if (!same(want, got[at + d])) return try std.fmt.bufPrint(&self.msg, "row {d} part {d} head {d} [{d}] = {e}, host {e}", .{ row, part, h, d, f(got[at + d]), want });
            }
        };
        return null;
    }

    /// The kernel's gelu in float32 steps around a tanh `t` (float64's, nudged by tanhf's ulps).
    fn gelu32(x: f32, t: f32) f32 {
        return 0.5 * x * (1.0 + t);
    }

    fn inner32(x: f32) f32 {
        const beta: f32 = @floatCast(std.math.sqrt2 * std.math.two_sqrtpi * 0.5);
        const cube = x * x * x;
        return beta * (x + 0.044715 * cube);
    }

    /// Every output within the products' roundings around tanhf two ulps either way of float64's tanh.
    fn checkGelu(self: *Glue, got: []align(1) const u16) !?[]const u8 {
        for (self.in[0], self.in[1], got, 0..) |gv, uv, o, i| {
            const x = f(gv);
            const u = f(uv);
            const v = f(o);
            if (!std.math.isFinite(x) or !std.math.isFinite(u)) {
                if (std.math.isFinite(v) and !(std.math.isInf(x) and x < 0)) return try std.fmt.bufPrint(&self.msg, "out[{d}] = {e} finite for gate {e}, up {e}", .{ i, v, x, u });
                continue;
            }
            const t: f32 = @floatCast(std.math.tanh(@as(f64, inner32(x))));
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            var tt = t;
            for (0..2) |_| tt = std.math.nextAfter(f32, tt, -std.math.inf(f32));
            for (0..5) |_| {
                const out = r(r(gelu32(x, @max(-1, @min(1, tt)))) * u);
                lo = @min(lo, out);
                hi = @max(hi, out);
                tt = std.math.nextAfter(f32, tt, std.math.inf(f32));
            }
            if (!(v >= lo and v <= hi)) return try std.fmt.bufPrint(&self.msg, "out[{d}] = {e} for gate {e}, up {e}: outside [{e}, {e}]", .{ i, v, x, u, lo, hi });
        }
        return null;
    }

    fn checkPool(self: *Glue, got: []align(1) const u16) !?[]const u8 {
        const bw = self.shape.k;
        const gw = 3 * bw;
        const root: f32 = @floatCast(@sqrt(@as(f64, width)));
        const x = self.in[0];
        for (0..self.shape.rows) |t| for (0..width) |ch| {
            const ty = t / bw;
            const tx = t % bw;
            var s: f32 = 0;
            for (0..3) |dy| for (0..3) |dx| {
                s = s + f(x[((3 * ty + dy) * gw + 3 * tx + dx) * width + ch]) * (1.0 / 9.0);
            };
            const want = (r(s) * root - f(self.in[1][ch])) * f(self.in[2][ch]);
            if (!same(want, got[t * width + ch])) return try std.fmt.bufPrint(&self.msg, "out[{d}][{d}] = {e}, host {e}", .{ t, ch, f(got[t * width + ch]), want });
        };
        return null;
    }
};

test "every control keeps all of its kernel's points but one" {
    var seen = [_]i32{ 0, 0, 0 };
    const all = [_]i32{ 3, 1, 1 };
    for (entries) |e| if (e.keep) |k| {
        const kind = @backingInt(e.kind);
        const dropped = all[kind] & ~k;
        try std.testing.expectEqual(@as(u32, 1), @popCount(dropped));
        seen[kind] |= dropped;
    };
    try std.testing.expectEqualSlices(i32, &all, &seen);
}
