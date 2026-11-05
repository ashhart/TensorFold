//! Ling-3.1-flash's own glue kernels under the contract: ling_glue.cu's rows and glm_glue's tf_ling_* SwiGLU and rope.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

pub const Kind = enum { add, gate, combine, copy, swiglu, xs, rope }; // copy: the Copy type below

/// wide: swiglu/xs the 512 block (else 64), copy 16-byte rows (else bytes), combine by expert (else by slot).
pub const Entry = struct { name: []const u8, kind: Kind, wide: bool = true, limit: f32 = std.math.inf(f32), scale: f32 = 1, pos: i32 = 0 };

pub const entries = [_]Entry{
    .{ .name = "cuda ling glue tf_ling_add (bf16 x + y)", .kind = .add },
    .{ .name = "cuda ling mla tf_mla_gate (head-wise sigmoid gate, in place)", .kind = .gate },
    .{ .name = "cuda ling glue tf_ling_combine (by slot, + shared)", .kind = .combine, .wide = false },
    .{ .name = "cuda ling glue tf_ling_combine (by expert, + shared)", .kind = .combine },
    .{ .name = "cuda ling glue tf_ling_copy_rows (16-byte rows)", .kind = .copy },
    .{ .name = "cuda ling glue tf_ling_copy_rows (byte rows)", .kind = .copy, .wide = false },
    .{ .name = "cuda ling ling_act tf_ling_swiglu_512 (no limit)", .kind = .swiglu },
    .{ .name = "cuda ling ling_act tf_ling_swiglu_512 (limit 7)", .kind = .swiglu, .limit = 7 },
    .{ .name = "cuda ling ling_act tf_ling_swiglu_64 (no limit)", .kind = .swiglu, .wide = false },
    .{ .name = "cuda ling ling_act tf_ling_swiglu_512 64-sums", .kind = .xs },
    .{ .name = "cuda ling ling_act tf_ling_swiglu_64 64-sums", .kind = .xs, .wide = false },
    .{ .name = "cuda ling mla tf_mla_rope (plain, positions from 1,000)", .kind = .rope, .pos = 1000 },
    .{ .name = "cuda ling mla tf_mla_rope (YaRN's attention factor, positions from 900,000)", .kind = .rope, .scale = 1.138629436111989, .pos = 900000 },
};

const experts = 16; // combine by expert: y [experts][rows][d]
const head_w = 192; // rope: q heads 192 wide, the 64 rope columns from column 128
const rope_at = 128;

fn depthNone(_: c.Shape) usize {
    return 0;
}

/// combine: each slot's product rounded, then the chain over the slots in use, then the shared add.
fn depthCombine(s: c.Shape) usize {
    return s.k + 2;
}

/// xs: four values a thread then a 16-lane butterfly (512), or a warp's butterfly then two warps (64).
fn depthXs(_: c.Shape) usize {
    return 8;
}

const shapes_of = struct {
    const add = [_]c.Shape{ .{ .rows = 1, .n = 4096, .k = 1 }, .{ .rows = 60, .n = 4096, .k = 1 } };
    const gate = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 128 }, .{ .rows = 40, .n = 64, .k = 128 } };
    const combine = [_]c.Shape{ .{ .rows = 1, .n = 4096, .k = 8 }, .{ .rows = 33, .n = 4096, .k = 8 } };
    const copy16 = [_]c.Shape{ .{ .rows = 1, .n = 8192, .k = 1 }, .{ .rows = 40, .n = 1152, .k = 1 } };
    const copy1 = [_]c.Shape{ .{ .rows = 3, .n = 1150, .k = 1 }, .{ .rows = 1, .n = 77, .k = 1 } };
    const sw512 = [_]c.Shape{ .{ .rows = 1, .n = 5120, .k = 64 }, .{ .rows = 20, .n = 10240, .k = 64 } };
    const sw64 = [_]c.Shape{ .{ .rows = 1, .n = 640, .k = 64 }, .{ .rows = 20, .n = 1280, .k = 64 } };
    const rope = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 64 }, .{ .rows = 40, .n = 64, .k = 64 } };
};

pub fn spec(e: Entry) c.Spec {
    const no: c.Tile = .{};
    return switch (e.kind) {
        .add => .{ .name = e.name, .accum = .none, .depth = depthNone, .ops = 1, .out = .bf16, .in_place = true, .tiles = .{ no, .{ .size = 256 }, no }, .served = &shapes_of.add },
        // exp (2 ulp), the add and the division: sigmoid within 8 units; bf16(sigmoid) is the mid rounding
        .gate => .{ .name = e.name, .accum = .none, .depth = depthNone, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .served = &shapes_of.gate },
        .combine => .{ .name = e.name, .accum = .f32, .depth = depthCombine, .out = .bf16, .tiles = .{ no, .{ .size = 256 }, .{ .size = 8, .min = 1 } }, .served = &shapes_of.combine },
        .copy => .{ .name = e.name, .accum = .none, .depth = depthNone, .out = .raw, .nan = .{ .stated = "bytes move as they are, NaN payloads included" }, .tiles = .{ no, if (e.wide) .{ .size = 4096, .step = 16, .min = 16 } else .{ .size = 256 }, no }, .served = if (e.wide) &shapes_of.copy16 else &shapes_of.copy1 },
        // ex2.approx after a rounded x log2(e) (|g| <= 16: 16 units), div.full (4 units), the product: 24 units
        .swiglu => .{ .name = e.name, .accum = .none, .depth = depthNone, .ops = 24, .mids = 1, .out = .bf16, .tiles = .{ no, if (e.wide) .{ .size = 512, .step = 512, .min = 512 } else .{ .size = 64, .step = 64, .min = 64 }, no }, .served = if (e.wide) &shapes_of.sw512 else &shapes_of.sw64 },
        .xs => .{ .name = e.name, .accum = .f32, .depth = depthXs, .out = .f32, .tiles = .{ no, if (e.wide) .{ .size = 512, .step = 512, .min = 512 } else .{ .size = 64, .step = 64, .min = 64 }, no }, .served = if (e.wide) &shapes_of.sw512 else &shapes_of.sw64 },
        .rope => .{ .name = e.name, .accum = .none, .depth = depthNone, .out = .bf16, .served = &shapes_of.rope },
    };
}

pub const Fns = struct { add: cuda.Function, gate: cuda.Function, combine: cuda.Function, copy: cuda.Function, sw: [2]cuda.Function, rope: cuda.Function };

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// torch.clamp's min(v, hi) with a NaN kept.
fn below(v: f64, hi: f64) f64 {
    return if (v > hi) hi else v;
}

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

pub const Glue = struct {
    rig: rig_.Rig,
    entry: Entry,
    fns: *const Fns,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    a16: []u16 = &.{}, // add x, gate g, combine y, swiglu gu, rope x
    b16: []u16 = &.{}, // add y, gate o (start), combine shared
    f: []f32 = &.{}, // combine wts, rope inv_freq
    ids: []i32 = &.{}, // combine pick
    bytes: []u8 = &.{}, // copy src, then the expected dst
    want_bytes: []u8 = &.{},
    extra: []f64 = &.{}, // per output: the allowance beyond the spec's bound (mid roundings, cancellation)
    out: u64 = 0,
    in: [4]u64 = @splat(0),
    scratch: u64 = 0,

    pub fn deinit(self: *Glue) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Glue) void {
        const a = self.rig.gpu.gpa;
        a.free(self.a16);
        a.free(self.b16);
        a.free(self.f);
        a.free(self.ids);
        a.free(self.bytes);
        a.free(self.want_bytes);
        a.free(self.extra);
        self.a16 = &.{};
        self.b16 = &.{};
        self.f = &.{};
        self.ids = &.{};
        self.bytes = &.{};
        self.want_bytes = &.{};
        self.extra = &.{};
    }

    pub fn kernel(self: *Glue) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Glue);

    fn slots(self: *const Glue) usize {
        return self.shape.k + 1;
    }

    fn pitches(self: *const Glue) [2]usize {
        return if (self.entry.wide) .{ self.shape.n + 32, self.shape.n + 16 } else .{ self.shape.n + 3, self.shape.n + 5 };
    }

    pub fn count(self: *const Glue) usize {
        const s = self.shape;
        return switch (self.entry.kind) {
            .add, .combine, .swiglu => s.rows * s.n,
            .gate => s.rows * s.n * s.k,
            .copy => s.rows * self.pitches()[1],
            .xs => s.rows * (s.n / 64),
            .rope => s.rows * s.n * 64,
        };
    }

    pub fn prepare(self: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const e = self.entry;
        if ((e.kind == .swiglu or e.kind == .xs) and shape.n % @as(usize, if (e.wide) 512 else 64) != 0) return false;
        if (e.kind == .copy and e.wide and shape.n % 16 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const n = shape.n;
        const sym = struct {
            fn v(r: std.Random, w: f32) u16 {
                return c.toBf16((r.float(f32) * 2 - 1) * w);
            }
        };
        switch (e.kind) {
            .add => {
                self.a16 = try a.alloc(u16, rows * n);
                self.b16 = try a.alloc(u16, rows * n);
                for (self.a16, self.b16) |*x, *y| {
                    x.* = sym.v(rng, 4);
                    y.* = sym.v(rng, 4);
                }
                if (fill == .nonfinite) {
                    self.a16[1] = c.toBf16(std.math.nan(f32));
                    self.a16[n + 2] = c.toBf16(std.math.inf(f32));
                    self.a16[2 * n] = c.toBf16(std.math.inf(f32));
                    self.b16[2 * n] = c.toBf16(-std.math.inf(f32));
                }
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.b16));
                self.out = try self.rig.output(rows * n * 2);
            },
            .gate => {
                const gs = n + 3;
                self.a16 = try a.alloc(u16, rows * gs);
                self.b16 = try a.alloc(u16, rows * n * shape.k);
                for (self.a16) |*x| x.* = sym.v(rng, 6);
                for (self.b16) |*x| x.* = sym.v(rng, 3);
                if (fill == .nonfinite) {
                    self.b16[5] = c.toBf16(std.math.nan(f32));
                    self.b16[n * shape.k + 7] = c.toBf16(std.math.inf(f32));
                    self.a16[2 * gs + 1] = c.toBf16(-std.math.inf(f32));
                    self.a16[3 * gs + 2] = c.toBf16(std.math.nan(f32));
                }
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                self.out = try self.rig.output(self.b16.len * 2);
            },
            .combine => try self.prepareCombine(rng, fill),
            .copy => {
                const p = self.pitches();
                self.bytes = try a.alloc(u8, rows * p[0]);
                rng.bytes(self.bytes);
                self.want_bytes = try a.alloc(u8, rows * p[1]);
                @memset(self.want_bytes, 0xff);
                for (0..rows) |r| @memcpy(self.want_bytes[r * p[1] ..][0..n], self.bytes[r * p[0] ..][0..n]);
                self.in[0] = try self.rig.input(self.bytes);
                self.out = try self.rig.output(self.want_bytes.len);
            },
            .swiglu, .xs => {
                self.a16 = try a.alloc(u16, rows * 2 * n);
                for (self.a16) |*x| x.* = sym.v(rng, 12);
                if (fill == .stress) for (0..rows) |r| for (0..n) |d| {
                    // silu(1024) is 1024 in fp32 and u * 1024 a bf16 value: each a exact, only sums round
                    self.a16[r * 2 * n + d] = c.toBf16(1024);
                    const u: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127);
                    self.a16[r * 2 * n + n + d] = c.toBf16(u);
                };
                if (fill == .nonfinite) {
                    self.a16[3] = c.toBf16(std.math.nan(f32)); // a gate NaN
                    self.a16[2 * n + n + 70] = c.toBf16(std.math.nan(f32)); // an up NaN
                    self.a16[4 * n + 130] = c.toBf16(std.math.inf(f32));
                    self.a16[6 * n + n + 200] = c.toBf16(-std.math.inf(f32));
                }
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                if (e.kind == .swiglu) {
                    self.out = try self.rig.output(rows * n * 2);
                    self.scratch = try self.rig.scratch(rows * (n / 64) * 4);
                } else {
                    self.scratch = try self.rig.scratch(rows * n * 2);
                    self.out = try self.rig.output(rows * (n / 64) * 4);
                }
            },
            .rope => {
                self.a16 = try a.alloc(u16, rows * n * head_w);
                for (self.a16) |*x| x.* = sym.v(rng, 4);
                if (fill == .nonfinite) {
                    self.a16[rope_at + 4] = c.toBf16(std.math.nan(f32));
                    self.a16[n * head_w + rope_at + 9] = c.toBf16(std.math.inf(f32));
                }
                self.f = try a.alloc(f32, 32);
                for (self.f, 0..) |*x, i| x.* = @floatCast(1.0 / std.math.pow(f64, 6.0e6, @as(f64, @floatFromInt(2 * i)) / 64.0));
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                self.in[1] = try self.rig.input(std.mem.asBytes(&e.pos));
                self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.f));
                self.out = try self.rig.output(rows * n * 64 * 2);
            },
        }
        return true;
    }

    /// Random slots; stress: integer weights near 2^19 and +-1, 2 or 4 per column, every partial sum an integer < 2^24.
    fn prepareCombine(self: *Glue, rng: std.Random, fill: g.Fill) !void {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        const d = self.shape.n;
        const k = self.shape.k;
        const ns = self.slots();
        const ny = if (self.entry.wide) experts * rows * d else rows * ns * d;
        self.a16 = try a.alloc(u16, ny);
        self.b16 = try a.alloc(u16, rows * d);
        self.f = try a.alloc(f32, rows * ns);
        self.ids = try a.alloc(i32, rows * ns);
        for (self.a16) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (self.b16) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (0..rows) |r| {
            var used: [experts]bool = @splat(false);
            for (0..ns) |j| {
                var x = rng.uintLessThan(u32, experts);
                while (used[x]) x = (x + 1) % experts;
                used[x] = true;
                self.ids[r * ns + j] = @intCast(x);
                self.f[r * ns + j] = if (j < k) rng.float(f32) * 0.5 else std.math.nan(f32); // past `used`: never read
            }
        }
        if (fill == .stress) {
            // slot j + 4 weighs 1-3 more than slot j and takes -y: each output is a small integer, every partial < 2^24
            if (k != 8) return error.StressNeedsEightSlots;
            const pw = [_]f32{ 1, 2, 4, -1, -2, -4 };
            for (0..rows) |r| {
                for (0..4) |j| {
                    const w: f32 = @floatFromInt((@as(u32, 1) << 18) + rng.uintLessThan(u32, 1 << 17));
                    self.f[r * ns + j] = w;
                    self.f[r * ns + j + 4] = w + @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 3)));
                }
                for (0..d) |col| for (0..4) |j| {
                    const y = pw[rng.uintLessThan(usize, pw.len)];
                    self.a16[self.yAt(r, j, col)] = c.toBf16(y);
                    self.a16[self.yAt(r, j + 4, col)] = c.toBf16(-y);
                };
            }
            @memset(self.b16, 0);
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 0, std.math.nan(f32) }, .{ 1, 1, std.math.inf(f32) }, .{ 2, 0, std.math.inf(f32) }, .{ 2, 2, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < rows) {
                self.a16[self.yAt(p[0], p[1], 17)] = c.toBf16(p[2]);
            };
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.f));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.b16));
        self.out = try self.rig.output(rows * d * 2);
    }

    fn yAt(self: *const Glue, r: usize, slot: usize, col: usize) usize {
        const d = self.shape.n;
        if (self.entry.wide) return (@as(usize, @intCast(self.ids[r * self.slots() + slot])) * self.shape.rows + r) * d + col;
        return (r * self.slots() + slot) * d + col;
    }

    pub fn run(self: *Glue) !void {
        const s = self.shape;
        const e = self.entry;
        var a: cuda.Args = .{};
        const st = self.rig.s;
        const cdiv = struct {
            fn f(x: usize, y: usize) u32 {
                return @intCast((x + y - 1) / y);
            }
        }.f;
        switch (e.kind) {
            .add => {
                inline for (.{ self.in[0], self.in[1], self.out, @as(i64, @intCast(s.rows * s.n)) }) |v| a.add(v);
                try cuda.launch.launch(self.fns.add, .{ .grid = .{ .x = cdiv(s.rows * s.n, 256) }, .block = .{ .x = 256 } }, st, &a);
            },
            .gate => {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.b16.len * 2 }).upload(0, std.mem.sliceAsBytes(self.b16));
                inline for (.{ self.out, @as(i64, @intCast(s.n)), self.in[0], @as(i64, @intCast(s.n + 3)) }) |v| a.add(v);
                try cuda.launch.launch(self.fns.gate, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = @intCast(s.k) } }, st, &a);
            },
            .combine => {
                inline for (.{ self.in[0], self.in[1], self.in[2], self.in[3], self.out, @as(i64, @intCast(s.rows)), @as(i64, @intCast(s.n)), @as(c_int, @intCast(self.slots())), @as(c_int, @intCast(s.k)), @as(c_int, @intFromBool(e.wide)) }) |v| a.add(v);
                try cuda.launch.launch(self.fns.combine, .{ .grid = .{ .x = @intCast(s.rows), .y = cdiv(s.n, 256) }, .block = .{ .x = 256 } }, st, &a);
            },
            .copy => {
                const p = self.pitches();
                inline for (.{ self.in[0], @as(c_longlong, @intCast(p[0])), self.out, @as(c_longlong, @intCast(p[1])), @as(c_int, @intCast(s.rows)), @as(c_int, @intCast(s.n)) }) |v| a.add(v);
                try cuda.launch.launch(self.fns.copy, .{ .grid = .{ .x = cdiv(s.n, 256 * 16), .y = @intCast(@min(s.rows, 65535)) }, .block = .{ .x = 256 } }, st, &a);
            },
            .swiglu, .xs => {
                const out16 = if (e.kind == .swiglu) self.out else self.scratch;
                const xs = if (e.kind == .swiglu) self.scratch else self.out;
                const block: usize = if (e.wide) 512 else 64;
                inline for (.{ self.in[0], out16, xs, e.limit, @as(i64, @intCast(s.n)) }) |v| a.add(v);
                try cuda.launch.launch(self.fns.sw[@intFromBool(!e.wide)], .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n / block) }, .block = .{ .x = if (e.wide) 128 else 64 } }, st, &a);
            },
            .rope => {
                inline for (.{ self.in[0] + rope_at * 2, @as(i64, @intCast(s.n * head_w)), @as(i64, head_w), self.out, @as(i64, @intCast(s.n * 64)), @as(i64, 64), self.in[1], self.in[2], e.scale }) |v| a.add(v);
                try cuda.launch.launch(self.fns.rope, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = 32 } }, st, &a);
            },
        }
        try st.synchronize();
    }

    /// One SwiGLU value in float64, torch's order: min(silu(g), L) * clamp(u, -L, L); its mid rounding is bf16(silu).
    fn swiglu1(self: *const Glue, gv: f64, uv: f64) f64 {
        const lim: f64 = self.entry.limit;
        const u = if (uv > lim) lim else if (uv < -lim) -lim else uv;
        return u * below(silu(gv), lim);
    }

    /// combine's terms of output o; returns the shared value.
    fn combineTerms(self: *const Glue, o: usize, t: []f64) f64 {
        const d = self.shape.n;
        const r = o / d;
        const col = o % d;
        for (t[0..self.shape.k], 0..) |*v, j| v.* = bf(self.a16[self.yAt(r, j, col)]) * self.f[r * self.slots() + j];
        return bf(self.b16[o]);
    }

    pub fn reference(self: *Glue, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        @memset(self.extra, 0);
        @memset(mag, 0);
        switch (self.entry.kind) {
            .add => for (want, 0..) |*w, i| {
                w.* = bf(self.a16[i]) + bf(self.b16[i]);
            },
            .gate => for (want, 0..) |*w, i| {
                const r = i / (s.n * s.k);
                const h = (i / s.k) % s.n;
                const z = bf(self.a16[r * (s.n + 3) + h]);
                w.* = bf(self.b16[i]) * (1 / (1 + @exp(-z)));
            },
            .combine => {
                var t: [64]f64 = undefined;
                for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
                    const sh = self.combineTerms(o, &t);
                    var sum: f64 = 0;
                    var abs: f64 = 0;
                    for (t[0..s.k]) |v| {
                        sum += v;
                        abs += @abs(v);
                    }
                    w.* = sum + sh;
                    m.* = abs + @abs(sh);
                    // bf16(sum) before the shared add; the stress sums are small integers, which bf16 holds
                    x.* = if (self.fill == .stress) 0 else 0x1p-8 * (@abs(sum) + 0x1p-20 * abs);
                }
            },
            .copy => for (want, 0..) |*w, i| {
                w.* = @floatFromInt(self.want_bytes[i]);
            },
            .swiglu => for (want, self.extra, 0..) |*w, *x, i| {
                const r = i / s.n;
                const d = i % s.n;
                w.* = self.swiglu1(bf(self.a16[r * 2 * s.n + d]), bf(self.a16[r * 2 * s.n + s.n + d]));
                x.* = 0;
            },
            .xs => for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
                const groups = s.n / 64;
                const r = o / groups;
                const d0 = (o % groups) * 64;
                var t: [64]f64 = undefined;
                var abs: f64 = 0;
                for (&t, 0..) |*v, j| {
                    v.* = self.swiglu1(bf(self.a16[r * 2 * s.n + d0 + j]), bf(self.a16[r * 2 * s.n + s.n + d0 + j]));
                    abs += @abs(v.*);
                }
                var sum: f64 = 0;
                for (t) |v| sum += v;
                w.* = sum;
                m.* = abs;
                // each value's own error (two bf16 roundings, 24 units of fp32 ops); the stress values are exact
                x.* = if (self.fill == .stress) 0 else (2 * 0x1p-8 + 24 * 0x1p-24) * abs;
            },
            .rope => for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
                const r = o / (s.n * 64);
                const h = (o / 64) % s.n;
                const col = o % 64;
                const i = col / 2;
                const src = (r * s.n + h) * head_w + rope_at + 2 * i;
                const x0 = bf(self.a16[src]);
                const x1 = bf(self.a16[src + 1]);
                const p: f32 = @floatFromInt(@as(i64, self.entry.pos) + @as(i64, @intCast(r)));
                const angle: f64 = self.f[i] * p; // the fp32 product, as the kernel and transformers form it
                const sc: f64 = self.entry.scale;
                const co = @cos(angle) * sc;
                const si = @sin(angle) * sc;
                const p0 = if (col % 2 == 0) x0 * co else x1 * co;
                const p1 = if (col % 2 == 0) -x1 * si else x0 * si;
                w.* = p0 + p1;
                m.* = @abs(p0) + @abs(p1);
                // sincosf's 2 ulp, the scale, two products and the add: within 8 units of the terms
                x.* = 8 * 0x1p-24 * m.*;
            },
        }
    }

    pub fn bounds(self: *Glue, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    pub fn simulate(self: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        var t: [64]f64 = undefined;
        for (out, 0..) |*v, o| switch (self.entry.kind) {
            .combine => {
                const sh = self.combineTerms(o, &t);
                v.* = contract.rows.sumIn(acc, order, t[0..s.k]) + sh;
            },
            .xs => {
                const groups = s.n / 64;
                const r = o / groups;
                const d0 = (o % groups) * 64;
                for (&t, 0..) |*x, j| x.* = self.swiglu1(bf(self.a16[r * 2 * s.n + d0 + j]), bf(self.a16[r * 2 * s.n + s.n + d0 + j]));
                v.* = contract.rows.sumIn(acc, order, &t);
            },
            else => return error.NoSums,
        };
    }
};

/// copy_rows moves bytes: its own check, every byte as moved and the gaps between rows untouched.
pub const Copy = struct {
    glue: Glue,

    pub fn kernel(self: *Copy) g.Kernel {
        return .{ .spec = spec(self.glue.entry), .ctx = &self.glue, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Glue);
        v.check = bytesCheck;
        break :blk v;
    };

    fn bytesCheck(ctx: *anyopaque, at: g.Gate) anyerror!?[]const u8 {
        _ = at;
        const self: *Glue = @ptrCast(@alignCast(ctx));
        const out = try self.rig.outputBytes();
        if (!std.mem.eql(u8, out, self.want_bytes)) return "a byte differs from the source row's, or a gap was written";
        return null;
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mods: [4]cuda.Module,
    fns: Fns = undefined,
    glue: [entries.len]Glue = undefined,
    copies: [entries.len]Copy = undefined,
    mla: *@import("contract_ling_mla.zig").Set,
    more: *@import("contract_ling_more.zig").Set,
    pattn: @import("contract_ling_pattn.zig").Pattn = undefined,

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var lg = try cuda.Module.load(gpu.d, cuda.kernels.ling_glue);
        errdefer lg.unload();
        var gg = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
        errdefer gg.unload();
        var mm = try cuda.Module.load(gpu.d, cuda.kernels.mla);
        errdefer mm.unload();
        var am = try cuda.Module.load(gpu.d, cuda.kernels.ling_act);
        errdefer am.unload();
        set.* = .{ .gpa = gpu.gpa, .mods = .{ lg, gg, mm, am }, .mla = undefined, .more = undefined };
        set.fns = .{
            .add = try lg.function("tf_ling_add"),
            .gate = try mm.function("tf_mla_gate"),
            .combine = try lg.function("tf_ling_combine"),
            .copy = try lg.function("tf_ling_copy_rows"),
            .sw = .{ try am.function("tf_ling_swiglu_512"), try am.function("tf_ling_swiglu_64") },
            .rope = try mm.function("tf_mla_rope"),
        };
        for (&set.glue, &set.copies, entries) |*x, *y, e| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .fns = &set.fns };
            y.* = .{ .glue = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .fns = &set.fns } };
        }
        for (&set.glue, &set.copies, entries) |*x, *y, e| try list.append(a, if (e.kind == .copy) y.kernel() else x.kernel());
        set.mla = try @import("contract_ling_mla.zig").Set.init(gpu, s, a, mm, list);
        set.more = try @import("contract_ling_more.zig").Set.init(gpu, s, a, gg, mm, list);
        set.pattn = try @import("contract_ling_pattn.zig").init(gpu, s, lg);
        try list.append(a, set.pattn.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.pattn.deinit();
        set.more.deinit();
        set.mla.deinit();
        for (&set.glue, &set.copies) |*x, *y| {
            x.deinit();
            y.glue.deinit();
        }
        for (&set.mods) |*m| m.unload();
        set.gpa.destroy(set);
    }
};

/// tf-cuda-test ling-contract: the Ling family's own kernels through gates 1-4.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6b63, &t);
    std.debug.print("SUMMARY ling: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings\n", .{ t.kernels, t.passed, t.failed, t.known });
    try check.expect(t.problems() == 0, "ling-contract: {d} gate failures", .{t.failed});
}
