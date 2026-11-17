//! ZONOS3's pointwise and moving kernels on Metal under the contract: no sums, each output a few fp32 steps from its
//! inputs (contract_zonos3_point.zig's CUDA twin).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ref = @import("metal_zonos3_ref.zig");
const erf64 = @import("dsv41_vision_ref.zig").erf;

const c = contract.spec;
const g = contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

pub const Kind = enum { mul_silu, combine, softcap, add_scaled, add_plain, gelu, patch_up, gather, spec, snake, add, sub, patch_down, sub_code, swish, glu };

const Image = ref.File;

const Info = struct {
    symbol: [:0]const u8,
    image: Image,
    name: []const u8,
    /// What the control's launch gets wrong.
    fault: []const u8,
    in_place: bool = false,
    out: c.Elem = .f32,
    served: []const c.Shape,
};

fn sh(rows: usize, n: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = 1 };
}

/// Each kind's facts, built at compile time so the served lists outlive the call.
fn info(kind: Kind) Info {
    return switch (kind) {
        inline else => |k| comptime infoOf(k),
    };
}

fn infoOf(comptime kind: Kind) Info {
    return switch (kind) {
        // the dense feed-forward's hidden rows, and an expert's for each of a row's two picks
        .mul_silu => .{ .symbol = "z3_mul_silu", .image = .backbone, .name = "metal zonos3 z3_mul_silu", .fault = "reads its gate one value late", .served = &.{ sh(64, 2048), sh(1, 2048), sh(2, 1536), sh(128, 1536) } },
        .combine => .{ .symbol = "z3_combine", .image = .backbone, .name = "metal zonos3 z3_combine", .fault = "weighs each pick with the other pick's weight", .served = &.{ sh(64, 256), sh(1, 256) } },
        .softcap => .{ .symbol = "z3_softcap", .image = .backbone, .name = "metal zonos3 z3_softcap (in place)", .fault = "launched one value short", .in_place = true, .served = &.{ sh(8, 40980), sh(1, 40980), sh(64, 40980) } },
        // a stage's attention and feed-forward residuals with their layer scale, and the ConvNeXt block's plain one
        .add_scaled => .{ .symbol = "zc_add_scaled", .image = .codec, .name = "metal zonos3 zc_add_scaled (in place, layer scale)", .fault = "launched without its scale", .in_place = true, .served = &.{ sh(16, 768), sh(300, 1280), sh(1, 768) } },
        .add_plain => .{ .symbol = "zc_add_scaled", .image = .codec, .name = "metal zonos3 zc_add_scaled (in place, no scale)", .fault = "launched one column short", .in_place = true, .served = &.{ sh(16, 768), sh(300, 768) } },
        .gelu => .{ .symbol = "zc_gelu", .image = .codec, .name = "metal zonos3 zc_gelu (in place)", .fault = "launched one value short", .in_place = true, .served = &.{ sh(16, 2048), sh(300, 5120), sh(1, 3072) } },
        // rows are the doubled ones, n half the stage's output width
        .patch_up => .{ .symbol = "zc_patch_up", .image = .codec, .name = "metal zonos3 zc_patch_up", .fault = "launched one row short", .served = &.{ sh(16, 384), sh(600, 640), sh(2, 384) } },
        .gather => .{ .symbol = "zc_gather", .image = .codec, .name = "metal zonos3 zc_gather", .fault = "reads the next codebook's column", .served = &.{ sh(16, 768), sh(1, 768), sh(300, 768) } },
        // n is both halves: 369 and 1,105 bins
        .spec => .{ .symbol = "zc_spec", .image = .codec, .name = "metal zonos3 zc_spec (in place)", .fault = "launched one row short", .in_place = true, .served = &.{ sh(16, 738), sh(16, 2210), sh(1, 738) } },
        .snake => .{ .symbol = "ze_snake", .image = .encoder, .name = "metal zonos3 ze_snake", .fault = "launched one value short", .served = &.{ sh(64, 64), sh(2944, 64), sh(368, 128), sh(1, 256) } },
        .add => .{ .symbol = "ze_add", .image = .encoder, .name = "metal zonos3 ze_add (in place)", .fault = "launched one value short", .in_place = true, .served = &.{ sh(64, 64), sh(2944, 64), sh(368, 128) } },
        .sub => .{ .symbol = "ze_sub", .image = .encoder, .name = "metal zonos3 ze_sub (in place)", .fault = "launched one value short", .in_place = true, .served = &.{ sh(16, 1024), sh(300, 1024), sh(1, 1024) } },
        // k is the patch: 23 front-end frames a row, then 2 a stage
        .patch_down => .{ .symbol = "ze_patch_down", .image = .encoder, .name = "metal zonos3 ze_patch_down", .fault = "launched one row short", .served = &.{ .{ .rows = 16, .n = 5888, .k = 23 }, .{ .rows = 8, .n = 768, .k = 2 }, .{ .rows = 150, .n = 1280, .k = 2 }, .{ .rows = 1, .n = 768, .k = 2 } } },
        .sub_code => .{ .symbol = "ze_sub_code", .image = .encoder, .name = "metal zonos3 ze_sub_code (in place)", .fault = "reads the next codebook's column", .in_place = true, .served = &.{ sh(16, 1024), sh(300, 1024), sh(1, 1024) } },
        .swish => .{ .symbol = "zt_swish", .image = .teacher, .name = "metal zonos3 zt_swish (in place)", .fault = "launched one value short", .in_place = true, .served = &.{ sh(16, 4096), sh(999, 1024), sh(1, 4096) } },
        .glu => .{ .symbol = "zt_glu", .image = .teacher, .name = "metal zonos3 zt_glu", .fault = "launched one row short", .served = &.{ sh(16, 1024), sh(999, 1024), sh(2, 1024) } },
    };
}

const table_rows = 64;
const column = 3;
const cap: f32 = 15;
const sqrt_half: f32 = 0.70710678118654752;
const log_100: f32 = 4.605170185988092;

pub const Point = struct {
    pipe: mtl.Pipeline,
    kind: Kind,
    /// control: the launch differs from the reference as `Info.fault` says
    fault: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    y: []f32 = &.{},
    z: []f32 = &.{},
    ids: []i32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,
    label: [120]u8 = undefined,

    pub fn init(libs: *ref.Libs, kind: Kind, fault: bool) !Point {
        const i = info(kind);
        return .{ .pipe = try libs.pipe(i.image, i.symbol), .kind = kind, .fault = fault };
    }

    pub fn spec(self: *Point) c.Spec {
        const i = info(self.kind);
        const name = if (self.fault) std.fmt.bufPrint(&self.label, "control: metal zonos3 {s} {s}", .{ i.symbol, i.fault }) catch unreachable else i.name;
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = i.out, .in_place = i.in_place, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = i.served };
    }

    pub fn control(self: *const Point) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Point, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.y);
        a.free(self.z);
        a.free(self.ids);
        self.x = &.{};
        self.y = &.{};
        self.z = &.{};
        self.ids = &.{};
    }

    /// The width of a row of `x`.
    fn width(self: *const Point) usize {
        const n = self.shape.n;
        return switch (self.kind) {
            .mul_silu => 2 * n + 1,
            .glu, .patch_up => 2 * n,
            .patch_down => n / self.shape.k,
            else => n,
        };
    }

    fn xRows(self: *const Point) usize {
        const rows = self.shape.rows;
        return switch (self.kind) {
            .combine => 2 * rows,
            .patch_up => rows / 2,
            .gather => table_rows,
            .patch_down => rows * self.shape.k,
            else => rows,
        };
    }

    pub fn prepare(self: *Point, a: std.mem.Allocator, rig: *ref.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        switch (self.kind) {
            .patch_up => if (shape.rows % 2 != 0) return false,
            .spec => if (shape.n % 2 != 0) return false,
            .patch_down => if (shape.k == 0 or shape.n % shape.k != 0) return false,
            else => {},
        }
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const xr = self.xRows();
        const xw = self.width();
        self.x = try ref.values(a, rng, xr * xw, fill, 4);
        if (fill == .nonfinite) ref.plant(self.x, xr, xw);
        switch (self.kind) {
            .combine => {
                self.y = try a.alloc(f32, 2 * rows);
                for (self.y) |*v| v.* = rng.float(f32);
            },
            .add_scaled => {
                self.y = try ref.values(a, rng, rows * n, fill, 4);
                self.z = try ref.values(a, rng, n, fill, 1);
            },
            .add_plain, .add, .sub => self.y = try ref.values(a, rng, rows * n, fill, 4),
            .snake => {
                self.y = try a.alloc(f32, n);
                for (self.y) |*v| v.* = 0.5 + 1.5 * rng.float(f32); // the checkpoint's alphas are positive
            },
            .sub_code => self.y = try ref.values(a, rng, table_rows * n, fill, 4),
            else => {},
        }
        if (self.kind == .gather or self.kind == .sub_code) {
            self.ids = try a.alloc(i32, rows * 10);
            for (self.ids, 0..) |*v, i| v.* = if (fill == .nonfinite and i % 10 == column) @intCast(i / 10 % table_rows) else @intCast(rng.uintLessThan(u32, table_rows));
        }
        if (info(self.kind).in_place) {
            self.out = try rig.output(self.x.len * 4);
        } else {
            self.dev[0] = try ref.input(rig, self.x);
            self.out = try rig.output(self.count() * 4);
        }
        if (self.y.len > 0) {
            if (self.kind == .combine and self.fault) {
                const swapped = try a.dupe(f32, self.y);
                defer a.free(swapped);
                for (0..rows) |r| std.mem.swap(f32, &swapped[2 * r], &swapped[2 * r + 1]);
                self.dev[1] = try ref.input(rig, swapped);
            } else self.dev[1] = try ref.input(rig, self.y);
        }
        if (self.z.len > 0) self.dev[2] = try ref.input(rig, self.z);
        if (self.ids.len > 0) self.dev[2] = try ref.input(rig, self.ids);
        return true;
    }

    pub fn run(self: *Point, rig: *ref.Rig) !void {
        const rows = self.shape.rows;
        const n = self.shape.n;
        const total = rows * n;
        const short = ref.one(self.fault);
        if (info(self.kind).in_place) try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.x));
        const p = self.pipe;
        const int = ref.int;
        const flat = [3]usize{ (total + 255) / 256, 1, 1 };
        const across = [3]usize{ (n + 255) / 256, rows, 1 };
        const Count = extern struct { n: u32 };
        const Long = extern struct { n: i64 };
        switch (self.kind) {
            .mul_silu => {
                const A = extern struct { stride: i32, a_off: i32, b_off: i32, n: i32 };
                try rig.go(p, across, 256, &.{ self.dev[0], self.out }, A{ .stride = int(2 * n + 1), .a_off = 0, .b_off = int(n + short), .n = int(n) }, 0);
            },
            .combine => try rig.go(p, across, 256, &.{ self.dev[0], self.dev[1], self.out }, extern struct { n: i32 }{ .n = int(n) }, 0),
            .softcap => {
                const A = extern struct { n: u32, cap: f32 };
                try rig.go(p, flat, 256, &.{self.out}, A{ .n = @intCast(total - short), .cap = cap }, 0);
            },
            .add_scaled, .add_plain => {
                const A = extern struct { D: i32, has_scale: i32 };
                const scaled = self.kind == .add_scaled and !self.fault;
                const scale = if (self.kind == .add_scaled) self.dev[2] else self.dev[1];
                try rig.go(p, across, 256, &.{ self.out, self.dev[1], scale }, A{ .D = int(if (self.kind == .add_plain) n - short else n), .has_scale = @intFromBool(scaled) }, 0);
            },
            .gelu => try rig.go(p, flat, 256, &.{self.out}, Count{ .n = @intCast(total - short) }, 0),
            .swish => try rig.go(p, flat, 256, &.{self.out}, Long{ .n = ref.long(total - short) }, 0),
            .patch_up => try rig.go(p, .{ (n + 255) / 256, rows - short, 1 }, 256, &.{ self.dev[0], self.out }, extern struct { C: i32 }{ .C = int(2 * n) }, 0),
            .gather => {
                const A = extern struct { col: i32, D: i32 };
                try rig.go(p, across, 256, &.{ self.dev[2], self.dev[0], self.out }, A{ .col = int(column + short), .D = int(n) }, 0);
            },
            .spec => try rig.go(p, .{ (n / 2 + 255) / 256, rows - short, 1 }, 256, &.{self.out}, extern struct { half_: i32 }{ .half_ = int(n / 2) }, 0),
            .snake => {
                const A = extern struct { C: i32, pad: i32 = 0, n: i64 };
                try rig.go(p, flat, 256, &.{ self.dev[0], self.dev[1], self.out }, A{ .C = int(n), .n = ref.long(total - short) }, 0);
            },
            .add, .sub => try rig.go(p, flat, 256, &.{ self.out, self.dev[1] }, Long{ .n = ref.long(total - short) }, 0),
            .patch_down => {
                const A = extern struct { C: i32, h: i32 };
                try rig.go(p, .{ (n + 255) / 256, rows - short, 1 }, 256, &.{ self.dev[0], self.out }, A{ .C = int(n / self.shape.k), .h = int(self.shape.k) }, 0);
            },
            .sub_code => {
                const A = extern struct { col: i32, D: i32 };
                try rig.go(p, across, 256, &.{ self.out, self.dev[1], self.dev[2] }, A{ .col = int(column + short), .D = int(n) }, 0);
            },
            .glu => try rig.go(p, .{ (n + 255) / 256, rows - short, 1 }, 256, &.{ self.dev[0], self.out }, extern struct { C: i32 }{ .C = int(n) }, 0),
        }
    }

    pub fn count(self: *const Point) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Point, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const x = self.x;
        for (out, 0..) |*o, at| {
            const r = at / n;
            const i = at % n;
            o.* = switch (self.kind) {
                .mul_silu => Iv.of(x[r * (2 * n + 1) + i]).mul(ref.silu(Iv.of(x[r * (2 * n + 1) + n + i]))),
                .combine => Iv.of(x[2 * r * n + i]).mul(Iv.of(self.y[2 * r])).add(Iv.of(x[(2 * r + 1) * n + i]).mul(Iv.of(self.y[2 * r + 1]))),
                .softcap => Iv.of(cap).mul(ref.tanh(Iv.of(x[at]).div(Iv.of(cap)))),
                .add_scaled => Iv.of(x[at]).add(Iv.of(self.z[i]).mul(Iv.of(self.y[at]))),
                .add_plain, .add => Iv.of(x[at]).add(Iv.of(self.y[at])),
                .sub => Iv.of(x[at]).sub(Iv.of(self.y[at])),
                .gelu => Iv.of(0.5).mul(Iv.of(x[at])).mul(Iv.of(1).add(erf(Iv.of(x[at]).mul(Iv.of(sqrt_half))))),
                .patch_up => Iv.of(x[(r >> 1) * 2 * n + 2 * i + (r & 1)]),
                .gather => Iv.of(x[@as(usize, @intCast(self.ids[r * 10 + column])) * n + i]),
                .spec => blk: {
                    const half = n / 2;
                    const bin = i % half;
                    const mag = Iv.of(x[r * n + bin]).atMost(log_100).exp();
                    break :blk mag.mul(Iv.of(x[r * n + half + bin]).trig(i >= half));
                },
                .snake => blk: {
                    const alpha = Iv.of(self.y[i]);
                    const s = alpha.mul(Iv.of(x[at])).trig(true);
                    break :blk Iv.of(x[at]).add(ref.square(s).div(alpha.add(Iv.of(@as(f32, 1e-9)))));
                },
                .patch_down => blk: {
                    const h = self.shape.k;
                    break :blk Iv.of(x[(r * h + i % h) * (n / h) + i / h]);
                },
                .sub_code => Iv.of(x[at]).sub(Iv.of(self.y[@as(usize, @intCast(self.ids[r * 10 + column])) * n + i])),
                .swish => ref.silu(Iv.of(x[at])),
                .glu => Iv.of(x[r * 2 * n + i]).mul(Iv.of(x[r * 2 * n + n + i]).sigmoid()),
            };
        }
    }
};

/// The kernels a control is planted for: every kind.
pub const kinds = std.enums.values(Kind);

/// z3_erf's distance from erf at z: below 0.5 the Taylor series (its tail under 1e-11) in about 20 fp32 steps of
/// its value; above, Numerical Recipes' erfcc (erfc within 1.2e-7 of itself) through precise::exp (4 ulps) of an
/// exponent whose roundings scale with z^2, about 20 steps of the fit, then 1 - erfc's own rounding.
fn erfError(z: f64) f64 {
    const az = @abs(z);
    if (az < 0.5) return 24 * ref.unit * @abs(erf64(z)) + 0x1p-149;
    return (1 - erf64(az)) * (1.2e-7 + (4 * az * az + 48) * ref.unit) + ref.unit;
}

/// codec.metal's z3_erf of a range: erf is increasing, and each end moves by z3_erf's error there.
fn erf(a: Iv) Iv {
    if (std.math.isNan(a.m)) return a;
    const e = @max(@max(erfError(a.lo), erfError(a.hi)), erfError(a.m));
    return .{ .m = erf64(a.m), .lo = @max(-1.0, erf64(a.lo) - e), .hi = @min(1.0, erf64(a.hi) + e) };
}

/// The group's registrations, each followed by its control.
pub const Set = struct {
    points: [2 * kinds.len]ref.Gated(Point),

    pub fn init(set: *Set, libs: *ref.Libs, rig: ref.Rig, list: ref.List) !void {
        for (kinds, 0..) |kind, i| for (0..2) |j| {
            set.points[2 * i + j] = .{ .rig = rig, .k = try Point.init(libs, kind, j == 1) };
        };
        for (&set.points) |*x| try list.add(x.kernel());
    }

    pub fn deinit(set: *Set) void {
        for (&set.points) |*x| x.deinit();
    }
};
