//! SAM3's mixed-precision trunk tiles (trunk_tiles.metal): FP16 projections on the tensor units and the FP16 row norm.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const norm = @import("metal_sam3_trunk_norm.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const vt = g.hooks(Probe);
const period = 67;
const u: f64 = 0x1p-24;

const Case = enum { project, gelu, residual, scatter };

fn function(case: Case) []const u8 {
    return switch (case) {
        .project => "sam3_tile_project",
        .gelu => "sam3_tile_project_gelu",
        .residual, .scatter => "sam3_tile_project_residual",
    };
}

const Edit = struct { function: []const u8, from: []const u8, to: []const u8 };

/// One planted fault a kernel: no bias, GELU without its half, the residual left in tile order, the norm ungathered.
const edits = [_]Edit{
    .{ .function = "sam3_tile_project", .from = "acc[i][j][e] = acc[i][j][e] + b;", .to = "acc[i][j][e] = acc[i][j][e];" },
    .{ .function = "sam3_tile_project_gelu", .from = "(0.5f * v) * (1.0f + qwv_erf", .to = "v * (1.0f + qwv_erf" },
    .{ .function = "sam3_tile_project_residual", .from = "p.scatter != 0 ? sam3_trunk_grid_row(r, p.grid, p.tile) : r", .to = "r" },
    .{ .function = "sam3_tile_norm_f16", .from = "p.gather != 0 ? sam3_trunk_grid_row(r, p.grid, p.tile) : r", .to = "r" },
};

pub fn text(a: std.mem.Allocator, edit: ?Edit) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    const e = edit orelse return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, e.from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, e.from, e.to);
}

fn editFor(fun: []const u8) !Edit {
    for (edits) |e| if (std.mem.eql(u8, e.function, fun)) return e;
    return error.UnknownLaunch;
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    if (!control) return text(a, null);
    return text(a, try editFor(name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..]));
}

fn label(case: Case, control: bool) []const u8 {
    if (control) return switch (case) {
        .project => "control: sam3 trunk tile projection without bias",
        .gelu => "control: sam3 trunk tile GELU without its half",
        .residual, .scatter => "control: sam3 trunk tile residual left in tile order",
    };
    return switch (case) {
        .project => "metal sam3 trunk tile projection",
        .gelu => "metal sam3 trunk tile GELU projection",
        .residual => "metal sam3 trunk tile residual projection",
        .scatter => "metal sam3 trunk tile residual projection scattered",
    };
}

/// Each k16 step of the FP16 tensor product: mma_step roundings of 2^-24 of the sums so far; then bias, residual.
fn depth(s: c.Shape) usize {
    return ar.mma_step * ((s.k + 15) / 16) + 2;
}

/// Shapes: rows in 64-row blocks (the kernel's precondition), n = outputs, k = depth; vision width 1024, MLP 4736.
fn served(case: Case) []const c.Shape {
    return switch (case) {
        .project => &.{.{ .rows = 81, .n = 3072, .k = 1024 }},
        .gelu => &.{.{ .rows = 81, .n = 4736, .k = 1024 }},
        .residual => &.{ .{ .rows = 81, .n = 1024, .k = 4736 }, .{ .rows = 81, .n = 1024, .k = 1024 } },
        .scatter => &.{.{ .rows = 81, .n = 1024, .k = 1024 }},
    };
}

const Params = extern struct { rows: u32, cols: u32, depth: u32, scatter: u32, grid: u32, tile: u32 };

/// The exact GELU in float64 (core act_ref's erf, tested beside it).
const gelu = contract.rows.gelu;

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    case: Case,
    control: bool,
    shape: c.Shape = undefined,
    stress: bool = false,
    x: []f16 = &.{},
    w: []f16 = &.{},
    bias: []f32 = &.{},
    residual: []f32 = &.{},
    pre: []f64 = &.{},
    pre_mag: []f64 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.case, p.control),
            .accum = .f32,
            .depth = depth,
            .out = if (p.case == .gelu) .f16 else .f32,
            .served = served(p.case),
            .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256 } },
        };
    }

    fn rows(p: *const Probe) usize {
        return if (p.case == .scatter) 5184 else 64 * p.shape.rows;
    }

    fn reset(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.a.free(p.bias);
        p.a.free(p.residual);
        p.a.free(p.pre);
        p.a.free(p.pre_mag);
        p.x = &.{};
        p.w = &.{};
        p.bias = &.{};
        p.residual = &.{};
        p.pre = &.{};
        p.pre_mag = &.{};
        p.rig.reset();
    }

    /// The scattered residual runs the whole 72 x 72 grid in nine 24 x 24 tiles whatever rows the harness asks for.
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.n % 128 != 0 or s.k == 0 or (p.case == .scatter and s.rows > 81)) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        const m = p.rows();
        p.x = try p.a.alloc(f16, m * s.k);
        p.w = try p.a.alloc(f16, s.n * s.k);
        p.bias = try p.a.alloc(f32, s.n);
        p.residual = try p.a.alloc(f32, if (p.case == .residual or p.case == .scatter) m * s.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const half = s.k / 2;
        for (0..m) |r| for (0..s.k) |k| {
            p.x[r * s.k + k] = if (p.stress) 1 else if (r >= period) p.x[(r % period) * s.k + k] else @floatCast(rng.float(f32) * 2 - 1);
        };
        // stress: each output sums 1025 and 1024 over the first half and -1024 over its mirror, so it is exactly 4
        for (0..s.n) |col| for (0..s.k) |k| {
            const at = if (k < half) k else k - half;
            p.w[col * s.k + k] = if (!p.stress) @floatCast(rng.float(f32) * 2 - 1) else if (k >= 2 * half) 0 else if (k < half) (if (at < 8 and at % 2 == 0) 1025 else 1024) else -1024;
        };
        for (p.bias) |*v| v.* = if (p.stress) 0 else rng.float(f32) * 2 - 1;
        for (p.residual) |*v| v.* = if (p.stress) 0 else rng.float(f32) * 8 - 4;
        if (fill == .nonfinite) {
            p.x[0] = std.math.nan(f16);
            p.x[s.k + 1] = std.math.inf(f16);
            p.x[2 * s.k + 2] = -std.math.inf(f16);
            p.x[3 * s.k] = std.math.inf(f16);
            p.x[3 * s.k + 1] = -std.math.inf(f16);
            for (0..m) |r| if (r >= period) {
                @memcpy(p.x[r * s.k ..][0..4], p.x[(r % period) * s.k ..][0..4]);
            };
        }
        p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.ins[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.ins[2] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
        if (p.residual.len > 0) p.ins[3] = try p.rig.input(std.mem.sliceAsBytes(p.residual));
        p.out = try p.rig.output(p.count() * @as(usize, if (p.case == .gelu) 2 else 4));
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const m = p.rows();
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins[0..3], 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        const params = Params{ .rows = @intCast(m), .cols = @intCast(s.n), .depth = @intCast(s.k), .scatter = @intFromBool(p.case == .scatter), .grid = 72, .tile = 24 };
        if (p.residual.len > 0) {
            pass.e.setBuffer(p.ins[3].buf, p.ins[3].off, 4);
            pass.e.setValue(params, 5);
        } else pass.e.setValue(params, 4);
        pass.e.dispatchThreads(mtl.Size.of(m / 64 * 32, s.n / 128 * 4, 2), mtl.Size.of(32, 4, 2));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.rows() * p.shape.n;
    }

    /// x W^T + b in float64 over the FP16 operands; rows repeat with the period, so only the first period are summed.
    fn project(p: *Probe) !void {
        const s = p.shape;
        const m = p.rows();
        p.pre = try p.a.alloc(f64, m * s.n);
        p.pre_mag = try p.a.alloc(f64, m * s.n);
        const wide = try p.a.alloc(f64, s.n * s.k);
        defer p.a.free(wide);
        for (wide, p.w) |*y, x| y.* = x;
        const row = try p.a.alloc(f64, s.k);
        defer p.a.free(row);
        for (0..m) |r| {
            if (r >= period) {
                @memcpy(p.pre[r * s.n ..][0..s.n], p.pre[(r % period) * s.n ..][0..s.n]);
                @memcpy(p.pre_mag[r * s.n ..][0..s.n], p.pre_mag[(r % period) * s.n ..][0..s.n]);
                continue;
            }
            for (row, p.x[r * s.k ..][0..s.k]) |*y, x| y.* = x;
            for (0..s.n) |col| {
                var sum: f64 = p.bias[col];
                var abs: f64 = @abs(sum);
                for (row, wide[col * s.k ..][0..s.k]) |x, w| {
                    sum += x * w;
                    abs += @abs(x * w);
                }
                p.pre[r * s.n + col] = sum;
                p.pre_mag[r * s.n + col] = abs;
            }
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.a.free(p.pre);
        p.a.free(p.pre_mag);
        p.pre = &.{};
        p.pre_mag = &.{};
        try p.project();
        const n = p.shape.n;
        for (0..p.rows()) |r| {
            const d = if (p.case == .scatter) norm.gridRow(r, 72, 24) else r;
            for (0..n) |col| {
                const v = p.pre[r * n + col];
                switch (p.case) {
                    .project => {
                        want[r * n + col] = v;
                        mag[r * n + col] = p.pre_mag[r * n + col];
                    },
                    .gelu => {
                        want[r * n + col] = gelu(v);
                        mag[r * n + col] = p.pre_mag[r * n + col];
                    },
                    .residual, .scatter => {
                        const res = p.residual[d * n + col];
                        want[d * n + col] = v + res;
                        mag[d * n + col] = p.pre_mag[r * n + col] + @abs(res);
                    },
                }
            }
        }
    }

    /// GELU: spread through GELU' <= 1.13, qwv_erf's 1.5e-7 and 20 roundings, 3 for the product, then FP16.
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec_ = p.spec();
        for (allow, 0..) |*l, i| {
            if (p.case != .gelu) {
                l.* = if (p.stress) c.boundExact(spec_, want[i]) else c.bound(spec_, p.shape, want[i], mag[i]);
                continue;
            }
            const v = p.pre[i];
            const spread = if (p.stress) u * @abs(v) else gamma(@floatFromInt(depth(p.shape))) * p.pre_mag[i] + u * @abs(v);
            const work = (1.13 * spread + 0.5 * @abs(v) * (1.5e-7 + 20 * u) + 3 * u * @abs(want[i])) * (1 + 0x1p-10);
            l.* = if (std.math.isFinite(work)) work + 0x1p-11 * (@abs(want[i]) + work) + 0x1p-25 else std.math.floatMax(f64);
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const terms = try p.a.alloc(f64, s.k + 2);
        defer p.a.free(terms);
        for (0..p.rows()) |r| {
            const d = if (p.case == .scatter) norm.gridRow(r, 72, 24) else r;
            for (0..s.n) |col| {
                for (terms[0..s.k], p.x[r * s.k ..][0..s.k], p.w[col * s.k ..][0..s.k]) |*t, x, w| t.* = @as(f64, x) * w;
                terms[s.k] = p.bias[col];
                terms[s.k + 1] = if (p.residual.len > 0) p.residual[d * s.n + col] else 0;
                const sum = contract.rows.sumIn(acc, order, terms);
                out[d * s.n + col] = if (p.case == .gelu) gelu(sum) else sum;
            }
        }
    }

    fn deinit(p: *Probe) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
    }
};

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [7]?*Probe = @splat(null),
    norms: [3]?*norm.Probe = @splat(null),

    fn want(only: ?[]const u8, name: []const u8) bool {
        const needle = only orelse return true;
        return std.mem.indexOf(u8, name, needle) != null;
    }

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        const cases = [_]Case{ .project, .gelu, .residual, .scatter, .project, .gelu, .scatter };
        for (cases, 0..) |case, i| {
            const control = i >= 4;
            if (!want(only, label(case, control))) continue;
            const body = try text(s.a, if (control) try editFor(function(case)) else null);
            defer s.a.free(body);
            const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try mtl.Pipeline.init(device, lib, function(case), false);
            errdefer pipe.deinit();
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            try kernels.ensureUnusedCapacity(s.a, 1);
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .case = case, .control = control };
            s.probes[i] = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
        for (0..3) |i| {
            const gather = i != 0;
            const control = i == 2;
            if (!want(only, norm.Probe.label(gather, control))) continue;
            const body = try text(s.a, if (control) try editFor("sam3_tile_norm_f16") else null);
            defer s.a.free(body);
            const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try mtl.Pipeline.init(device, lib, "sam3_tile_norm_f16", false);
            errdefer pipe.deinit();
            const p = try s.a.create(norm.Probe);
            errdefer s.a.destroy(p);
            try kernels.ensureUnusedCapacity(s.a, 1);
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .gather = gather, .control = control };
            s.norms[i] = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &norm.vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
        for (s.norms) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
