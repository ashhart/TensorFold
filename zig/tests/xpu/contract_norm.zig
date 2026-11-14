//! The glue's norms on XPU: the residual add and the norm in one launch, wide and decode.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref;

/// The normed row the model reads, or the residual the kernel writes in place; wide (128 work-items) or decode (512).
pub const Which = enum { add_rmsnorm_h, add_rmsnorm_x, add_rmsnorm_dec_h, add_rmsnorm_dec_x, gated_norm_h };

const D = 5120; // Q27_D, the widest row the norms take
const VD = 6144; // Q27_VD, the gated row's width
const GDN_IN = 16640; // Q27_GDN_IN, the GDN input row the gated norm reads
const Z_AT = 10240; // Q27_NCONV, where z starts in that row
const EPS: f32 = 1e-6;
const GS: usize = 64;
const MAX_ROWS = 8320; // a prompt chunk of 8192 rows plus 128

const why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const why_bf16 = why ++ "; fp16's overflowing sums give 0, which the bound's residual roundings put at 120x";

fn zero(_: c.Shape) usize {
    return 0;
}
fn wideDepth(s: c.Shape) usize {
    return ref.normDepth(false, s);
}
fn decDepth(s: c.Shape) usize {
    return ref.normDepth(true, s);
}

fn isGated(w: Which) bool {
    return w == .gated_norm_h;
}
fn isH(w: Which) bool {
    return w != .add_rmsnorm_x and w != .add_rmsnorm_dec_x;
}
fn isDec(w: Which) bool {
    return w == .add_rmsnorm_dec_h or w == .add_rmsnorm_dec_x;
}
fn symbol(w: Which) []const u8 {
    return switch (w) {
        .add_rmsnorm_h, .add_rmsnorm_x => "add_rmsnorm",
        .add_rmsnorm_dec_h, .add_rmsnorm_dec_x => "add_rmsnorm_dec",
        .gated_norm_h => "gated_norm",
    };
}

const wide_served = [_]c.Shape{
    .{ .rows = 1, .n = D, .k = D },
    .{ .rows = 16, .n = D, .k = D },
    .{ .rows = 1152, .n = D, .k = D },
    .{ .rows = MAX_ROWS, .n = D, .k = D },
};
const dec_served = [_]c.Shape{
    .{ .rows = 1, .n = D, .k = D },
    .{ .rows = 16, .n = D, .k = D },
};
const gated_served = [_]c.Shape{
    .{ .rows = 1, .n = VD, .k = VD },
    .{ .rows = 16, .n = VD, .k = VD },
    .{ .rows = 1152, .n = VD, .k = VD },
};

pub fn spec(w: Which) c.Spec {
    var s = c.Spec{ .name = "", .accum = .f32, .depth = wideDepth, .ops = 8, .mids = 16, .out = .f16, .served = &wide_served };
    s.tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{ .size = 512, .step = 512, .min = 512 } };
    switch (w) {
        .add_rmsnorm_h => s.name = "xpu add_rmsnorm (the normed row, fp16)",
        .add_rmsnorm_dec_h => {
            s.name = "xpu add_rmsnorm_dec (the normed row, fp16)";
            s.served = &dec_served;
            s.depth = decDepth;
        },
        .add_rmsnorm_x => {
            s.name = "xpu add_rmsnorm (the residual in place, fp32)";
            s.accum = .none;
            s.depth = zero;
            s.ops = 0;
            s.store = .bf16; // the residual is a bf16 value in an fp32 word, as glue.cl's rbf stores it
            s.mids = 1; // the tie the card's fp32 sum and the bf16 store can differ by
            s.out = .f32;
        },
        .add_rmsnorm_dec_x => {
            s.name = "xpu add_rmsnorm_dec (the residual in place, fp32)";
            s.accum = .none;
            s.depth = zero;
            s.ops = 0;
            s.store = .bf16;
            s.mids = 1;
            s.out = .f32;
            s.served = &dec_served;
        },
        .gated_norm_h => {
            s.name = "xpu gated_norm (the gated row, fp16)";
            s.ops = 16; // rsqrt and silu's exp
            s.depth = ref.gatedDepth;
            s.served = &gated_served;
            s.tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{} };
        },
    }
    s.sums = .{ .bf16 = 0, .why = why_bf16 };
    return s;
}

/// A row of the kind: the normed row fp16, or the residual fp32, or the gated row fp16.
pub const Norm = struct {
    rig: rig_.Rig,
    k: xpu.Kernel,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    r: []f64 = &.{},
    y: []f64 = &.{},
    z: []f64 = &.{},
    w: []f64 = &.{},
    xb: []u8 = &.{},
    rb: []u8 = &.{},
    yb: []u8 = &.{},
    pb: []u8 = &.{},
    wb: []u8 = &.{},
    xat: ?*anyopaque = null,
    rat: ?*anyopaque = null,
    yat: ?*anyopaque = null,
    pat: ?*anyopaque = null,
    wat: ?*anyopaque = null,
    hout: ?*anyopaque = null,
    ahi: ?*anyopaque = null,
    alo: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, which: Which) !Norm {
        return .{ .rig = .{ .a = a, .dev = dev }, .k = try dev.kernelFor(.glue, symbol(which), a), .which = which, .arena = .init(a) };
    }

    pub fn deinit(self: *Norm) void {
        self.k.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    fn signed(rng: std.Random, lo: f64, hi: f64) f64 {
        const v = lo + rng.float(f64) * (hi - lo);
        return if (rng.boolean()) v else -v;
    }

    /// NaN in row 0's row, both infinities in rows 1 and 2, and a value whose square overflows fp32 in row 3.
    fn plant(self: *Norm) void {
        const n = self.shape.n;
        self.x[5] = std.math.nan(f64);
        if (isH(self.which)) {
            self.x[n + 3] = std.math.inf(f64);
            self.x[2 * n + 1] = -std.math.inf(f64);
            self.x[3 * n + 2] = 3e38;
        } else {
            self.r[n + 3] = std.math.inf(f64);
            self.r[2 * n + 1] = -std.math.inf(f64);
            self.x[3 * n + 2] = 3e38;
        }
    }

    /// The gated row's plants: NaN in row 0's y, both infinities in rows 1 and 2's z.
    fn plantGated(self: *Norm) void {
        const n = self.shape.n;
        self.y[5] = std.math.nan(f64);
        self.z[n + 3] = std.math.inf(f64);
        self.z[2 * n + 1] = -std.math.inf(f64);
        self.y[3 * n + 2] = 60000; // its square is past fp32's exact integers, so the sum carries an error
    }

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gated = isGated(self.which);
        const n: usize = if (gated) VD else D;
        if (shape.rows == 0 or shape.rows > MAX_ROWS) return false;
        if (shape.n != n) return false;
        if (!gated and (shape.n % 512 != 0 or shape.n > D)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const total = rows * n;
        self.w = try a.alloc(f64, n);
        for (self.w) |*v| v.* = kit.round(.bf16, 0.75 + rng.float(f64) * 0.5);
        if (gated) {
            self.y = try a.alloc(f64, total);
            self.z = try a.alloc(f64, total);
            for (self.y) |*v| v.* = kit.round(.bf16, 0.5 + rng.float(f64));
            for (self.z) |*v| v.* = kit.round(.f16, -2 + rng.float(f64) * 4);
            if (fill == .stress) {
                @memset(self.y, 64); // 128 squares of 64 are 524,288: exact in fp32, past fp16's range
                @memset(self.z, 1);
                @memset(self.w, 1);
            }
            if (fill == .nonfinite) self.plantGated();
            self.yb = try kit.bytes(a, self.y, .bf16);
            self.pb = try a.alloc(u8, rows * GDN_IN * 2);
            @memset(self.pb, 0);
            for (0..rows) |r| for (0..VD) |j| {
                kit.put(.f16, self.pb, r * GDN_IN + Z_AT + (j / 128) * 128 + j % 128, self.z[r * VD + j]);
            };
            self.wb = try kit.bytes(a, self.w, .bf16);
            self.yat = try self.rig.input(self.yb);
            self.pat = try self.rig.input(self.pb);
            self.wat = try self.rig.input(self.wb);
            self.hout = try self.rig.output(total * 2);
            self.ahi = try self.rig.scratch(rows * (VD / GS) * 2);
            self.alo = try self.rig.scratch(rows * (VD / GS) * 2);
            return true;
        }
        self.x = try a.alloc(f64, total);
        self.r = try a.alloc(f64, total);
        for (self.x) |*v| v.* = kit.round(.f32, signed(rng, 1, 2) * (if (rng.uintLessThan(u32, 64) == 7) @as(f64, 20) else @as(f64, 1)));
        for (self.x, self.r) |xv, *rv| {
            const sign: f64 = if (xv < 0) -1 else 1;
            rv.* = kit.round(.f16, sign * (0.25 + rng.float(f64) * 0.25) * @abs(xv));
        }
        if (fill == .stress) {
            @memset(self.r, 0); // nothing added: the norm's rows of integers, so its sums are exact in fp32
            for (self.x) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 64))) - 32);
        }
        if (fill == .nonfinite) self.plant();
        self.xb = try kit.bytes(a, self.x, .f32);
        self.rb = try kit.bytes(a, self.r, .f16);
        self.wb = try kit.bytes(a, self.w, .bf16);
        self.xat = if (isH(self.which)) try self.rig.scratch(total * 4) else try self.rig.output(total * 4);
        self.rat = try self.rig.input(self.rb);
        self.wat = try self.rig.input(self.wb);
        self.hout = if (isH(self.which)) try self.rig.output(total * 2) else try self.rig.scratch(total * 2);
        self.ahi = try self.rig.scratch(rows * (n / GS) * 2);
        self.alo = try self.rig.scratch(rows * (n / GS) * 2);
        return true;
    }

    pub fn run(self: *Norm) !void {
        const s = self.shape;
        const grid = [3]u32{ @intCast(s.rows), 1, 1 };
        if (isGated(self.which)) {
            try self.rig.launch(self.k, grid, .{
                self.yat, self.pat,     self.wat,    self.hout,              self.ahi,               self.alo,
                EPS,      @as(u32, GS), @as(u32, 0), @as(?*anyopaque, null), @as(?*anyopaque, null), @as(u32, 0),
            });
            return;
        }
        try self.rig.put(self.xat, self.xb); // the residual the kernel updates in place
        try self.rig.launch(self.k, grid, .{
            self.xat,                self.rat,    self.wat,                                 self.hout,    self.ahi,               self.alo,
            @as(u32, @intCast(s.n)), EPS,         @as(u32, @intFromBool(!isH(self.which))), @as(u32, GS), @as(?*anyopaque, null), @as(?*anyopaque, null),
            @as(u32, 0),             @as(u32, 0),
        });
    }

    pub fn count(self: *Norm) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        switch (self.which) {
            .add_rmsnorm_h => try ref.normedRow(a, self.x, self.r, self.w, .{ .dec = false, .eps = EPS }, self.shape, false, want, mag),
            .add_rmsnorm_dec_h => try ref.normedRow(a, self.x, self.r, self.w, .{ .dec = true, .eps = EPS }, self.shape, false, want, mag),
            .add_rmsnorm_x, .add_rmsnorm_dec_x => ref.residualRow(self.x, self.r, self.shape, want, mag),
            .gated_norm_h => try ref.gatedRow(a, self.y, self.z, self.w, EPS, self.shape, want, mag),
        }
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.arena.allocator();
        switch (self.which) {
            .add_rmsnorm_h => try ref.simulateNormed(a, self.x, self.r, self.w, .{ .dec = false, .eps = EPS }, self.shape, false, acc, order, out),
            .add_rmsnorm_dec_h => try ref.simulateNormed(a, self.x, self.r, self.w, .{ .dec = true, .eps = EPS }, self.shape, false, acc, order, out),
            .gated_norm_h => try ref.simulateGated(a, self.y, self.z, self.w, EPS, self.shape, acc, order, out),
            .add_rmsnorm_x, .add_rmsnorm_dec_x => @memset(out, 0),
        }
    }
};
