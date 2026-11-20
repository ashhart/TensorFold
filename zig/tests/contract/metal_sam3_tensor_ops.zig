//! The SAM tensor library's generic ops (tensor_ops.metal) at the dtypes and widths the SAM3 family serves them.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

const u: f64 = 0x1p-24;

pub const Kind = enum { cast_down, cast_up, pad, unpad, add, add_rows, mul_rows, norm, gelu, gelu_half };

/// Each kernel's planted source edit; its control runs at the first kind naming the function.
const Edit = struct { function: []const u8, from: []const u8, to: []const u8, kind: Kind, gate: c.Gate, label: []const u8 };
const edits = [_]Edit{
    .{ .function = "im_cast", .kind = .cast_down, .gate = .shapes, .label = "control: sam3 tensor ops cast reads the output's kind", .from = "im_write(out, i, p[2], im_read(x, i, p[1]));", .to = "im_write(out, i, p[2], im_read(x, i, p[2]));" },
    .{ .function = "im_heads", .kind = .pad, .gate = .shapes, .label = "control: sam3 tensor ops heads leave padding unzeroed", .from = "at.x < p.src_dim ? im_read(x, src, p.kind) : 0.0f", .to = "im_read(x, src, p.kind)" },
    .{ .function = "im_point", .kind = .add, .gate = .shapes, .label = "control: sam3 tensor ops point swaps add and multiply", .from = "p.operation >= 2 ? x * y : x + y", .to = "p.operation >= 2 ? x + y : x * y" },
    .{ .function = "im_norm", .kind = .norm, .gate = .shapes, .label = "control: sam3 tensor ops norm drops its bias", .from = "value + (p.has_bias ? im_read(bias, d, p.kind) : 0.0f)", .to = "value" },
    .{ .function = "im_gelu", .kind = .gelu, .gate = .shapes, .label = "control: sam3 tensor ops gelu without its half", .from = "im_write(out, i, p[1], 0.5f * v * (1.0f + inner));", .to = "im_write(out, i, p[1], v * (1.0f + inner));" },
};

/// The engine's SAM library (tensor_library over sam3_source), with one kernel's planted edit for its control.
pub fn libraryText(a: std.mem.Allocator, edit: ?Edit) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    const e = edit orelse return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, e.from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, e.from, e.to);
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    if (!control) return libraryText(a, null);
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    for (edits) |e| if (std.mem.eql(u8, e.function, fun)) return libraryText(a, e);
    return error.UnknownLaunch;
}

fn function(kind: Kind) []const u8 {
    return switch (kind) {
        .cast_down, .cast_up => "im_cast",
        .pad, .unpad => "im_heads",
        .add, .add_rows, .mul_rows => "im_point",
        .norm => "im_norm",
        .gelu, .gelu_half => "im_gelu",
    };
}

fn label(kind: Kind) []const u8 {
    return switch (kind) {
        .cast_down => "metal sam3 tensor ops cast f32 to f16",
        .cast_up => "metal sam3 tensor ops cast f16 to f32",
        .pad => "metal sam3 tensor ops heads 16 to 32",
        .unpad => "metal sam3 tensor ops heads 32 to 16",
        .add => "metal sam3 tensor ops point add",
        .add_rows => "metal sam3 tensor ops point add a row",
        .mul_rows => "metal sam3 tensor ops point multiply a row",
        .norm => "metal sam3 tensor ops layer norm",
        .gelu => "metal sam3 tensor ops exact gelu f32",
        .gelu_half => "metal sam3 tensor ops exact gelu f16",
    };
}

fn exact(kind: Kind) bool {
    return switch (kind) {
        .cast_down, .cast_up, .pad, .unpad => true,
        else => false,
    };
}

// Shapes: rows = elements (heads: token rows), n = a broadcast row's width or the heads, k = a norm's width.
fn served(kind: Kind) []const c.Shape {
    return switch (kind) {
        .cast_down => &.{ .{ .rows = 5184 * 1024, .n = 1, .k = 1 }, .{ .rows = 5184 * 256, .n = 1, .k = 1 } },
        .cast_up => &.{.{ .rows = 5184 * 256, .n = 1, .k = 1 }},
        .pad, .unpad => &.{ .{ .rows = 5184, .n = 8, .k = 1 }, .{ .rows = 7, .n = 8, .k = 1 } },
        .add => &.{ .{ .rows = 5184 * 1024, .n = 1, .k = 1 }, .{ .rows = 5184 * 256, .n = 1, .k = 1 }, .{ .rows = 32 * 1024, .n = 1, .k = 1 } },
        .add_rows => &.{ .{ .rows = 5184 * 64, .n = 64, .k = 1 }, .{ .rows = 201 * 256, .n = 256, .k = 1 } },
        .mul_rows => &.{.{ .rows = 5184 * 256, .n = 256, .k = 1 }},
        .norm => &.{ .{ .rows = 5184, .n = 1024, .k = 1024 }, .{ .rows = 5184, .n = 256, .k = 256 }, .{ .rows = 32, .n = 1024, .k = 1024 }, .{ .rows = 201, .n = 256, .k = 256 } },
        .gelu => &.{ .{ .rows = 5184 * 2048, .n = 1, .k = 1 }, .{ .rows = 32 * 4096, .n = 1, .k = 1 } },
        .gelu_half => &.{.{ .rows = 5184 * 1024, .n = 1, .k = 1 }},
    };
}

fn one(_: c.Shape) usize {
    return 1;
}

fn zero(_: c.Shape) usize {
    return 0;
}

// tf_vis_sum: each thread's chain over width / 256 values, then two five-level simd sums.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

fn specOf(kind: Kind, control: ?Edit) c.Spec {
    return .{
        .name = if (control) |e| e.label else label(kind),
        .accum = switch (kind) {
            .add, .add_rows, .norm => .f32,
            else => .none,
        },
        .depth = switch (kind) {
            .add, .add_rows => one,
            .norm => normDepth,
            else => zero,
        },
        // Layer norm past its sums: mean, difference, scale, offset, precise rsqrt (4), times inv and weight, bias.
        .ops = if (kind == .norm) 11 else 0,
        .out = switch (kind) {
            .cast_down, .gelu_half => .f16,
            else => .f32,
        },
        .nan = if (exact(kind)) .{ .stated = "every finite word converts or copies exactly; NaN stays NaN" } else .propagate,
        .tiles = if (kind == .norm) .{ .{ .size = 4 }, .{}, .{ .size = 256 } } else if (kind == .pad or kind == .unpad) .{ .{ .size = 64 }, .{}, .{} } else .{ .{ .size = 256 }, .{}, .{} },
        .served = served(kind),
    };
}

fn fillValue(rng: std.Random, i: usize) f32 {
    const scale: f32 = switch (i % 5) {
        0 => 1,
        1 => 0x1p-14,
        2 => 0x1p10,
        3 => 0x1p17,
        else => 0x1p-20,
    };
    return (rng.float(f32) * 2 - 1) * scale;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    control: ?Edit = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []f32 = &.{},
    y: []f32 = &.{},
    rows: ?contract.rows.Rows = null,
    expect: []u32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = specOf(p.kind, p.control), .ctx = p, .vt = if (exact(p.kind)) &exact_vt else &ranged_vt, .control = if (p.control) |e| e.gate else null };
    }

    fn src(p: *const Probe) usize {
        return if (p.kind == .unpad) 32 else 16;
    }

    fn dst(p: *const Probe) usize {
        return if (p.kind == .unpad) 16 else 32;
    }

    pub fn reset(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.y);
        p.a.free(p.expect);
        if (p.rows) |r| r.deinit();
        p.x = &.{};
        p.y = &.{};
        p.expect = &.{};
        p.rows = null;
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (p.kind == .norm and shape.k == 0) return false;
        p.reset();
        p.shape = shape;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (p.kind) {
            .norm => {
                const r = try contract.rows.Rows.init(p.a, .layer_norm, shape);
                p.rows = r;
                r.fill(fill, seed);
                for ([_][]const u16{ r.x, r.w, r.b }, &p.ins) |values, *input| {
                    const floats = try p.a.alloc(f32, values.len);
                    defer p.a.free(floats);
                    for (values, floats) |v, *f| f.* = c.fromBf16(v);
                    input.* = try p.rig.input(std.mem.sliceAsBytes(floats));
                }
                p.out = try p.rig.output(r.outputs() * 4);
                return true;
            },
            .cast_up => {
                const halves = try p.a.alloc(u16, shape.rows);
                defer p.a.free(halves);
                for (halves) |*h| {
                    h.* = rng.int(u16);
                    if (h.* & 0x7c00 == 0x7c00) h.* &= 0xbfff; // finite words, subnormals included
                }
                if (fill == .nonfinite) for ([_]u16{ 0x7e00, 0x7c00, 0xfc00, 0xfe01 }, 0..) |h, i| {
                    halves[i % halves.len] = h;
                };
                p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(halves));
                p.expect = try p.a.alloc(u32, shape.rows);
                for (halves, p.expect) |h, *e| e.* = @bitCast(@as(f32, @as(f16, @bitCast(h))));
                p.out = try p.rig.output(shape.rows * 4);
                return true;
            },
            else => {},
        }
        const xs: usize = if (p.kind == .pad or p.kind == .unpad) shape.rows * shape.n * p.src() else shape.rows;
        p.x = try p.a.alloc(f32, xs);
        for (p.x, 0..) |*v, i| v.* = switch (p.kind) {
            .gelu, .gelu_half => (rng.float(f32) * 2 - 1) * (if (i % 97 == 0) @as(f32, 64) else 8),
            else => fillValue(rng, i),
        };
        if (p.kind == .gelu_half) for (p.x) |*v| {
            v.* = @as(f16, @floatCast(v.*));
        };
        if (p.kind == .cast_down) for ([_]f32{ 1 + 0x1p-11, 1 + 3 * 0x1p-11, 65504, 65520, 0x1p-25, 0x1.8p-24, -0x1p-24 }, 0..) |v, i| {
            p.x[(i * 131) % p.x.len] = v;
        };
        if (p.kind == .add or p.kind == .add_rows or p.kind == .mul_rows) {
            p.y = try p.a.alloc(f32, if (p.kind == .add) shape.rows else shape.n);
            for (p.y, 0..) |*v, i| v.* = fillValue(rng, i + 3);
        }
        if (fill == .stress and (p.kind == .add or p.kind == .add_rows)) {
            // Integer pairs whose sum is small: exact in fp32, past fp16's range and bf16's precision.
            for (p.x, 0..) |*v, i| v.* = 0x1p20 + @as(f32, @floatFromInt(1 + i % 7));
            @memset(p.y, -0x1p20);
        }
        if (fill == .nonfinite) {
            const plant = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (plant, 0..) |v, i| p.x[(i * 7) % p.x.len] = v;
            if (p.y.len > 0) p.y[p.y.len - 1] = std.math.nan(f32);
        }
        if (p.kind == .gelu_half) {
            const halves = try p.a.alloc(f16, p.x.len);
            defer p.a.free(halves);
            for (halves, p.x) |*h, v| h.* = @floatCast(v);
            p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(halves));
        } else p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        if (p.y.len > 0) p.ins[1] = try p.rig.input(std.mem.sliceAsBytes(p.y));
        if (p.kind == .cast_down) {
            p.expect = try p.a.alloc(u32, shape.rows);
            for (p.x, p.expect) |v, *e| e.* = @as(u16, @bitCast(@as(f16, @floatCast(v))));
        } else if (p.kind == .pad or p.kind == .unpad) {
            p.expect = try p.a.alloc(u32, p.count());
            for (p.expect, 0..) |*e, i| {
                const at = i % p.dst();
                const row_head = i / p.dst();
                e.* = if (at < p.src()) @bitCast(p.x[row_head * p.src() + at]) else 0;
            }
        }
        p.out = try p.rig.output(p.count() * @as(usize, if (p.kind == .cast_down or p.kind == .gelu_half) 2 else 4));
        return true;
    }

    pub fn count(p: *const Probe) usize {
        return switch (p.kind) {
            .pad, .unpad => p.shape.rows * p.shape.n * p.dst(),
            .norm => p.rows.?.outputs(),
            else => p.shape.rows,
        };
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        switch (p.kind) {
            .cast_down, .cast_up => {
                pass.e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                pass.e.setBuffer(p.out.buf, p.out.off, 1);
                pass.e.setValue([3]u32{ @intCast(s.rows), if (p.kind == .cast_down) 1 else 2, if (p.kind == .cast_down) 2 else 1 }, 2);
                pass.e.dispatchThreads(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .pad, .unpad => {
                pass.e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                pass.e.setBuffer(p.out.buf, p.out.off, 1);
                const stride = s.n * p.src();
                pass.e.setValue([8]u32{ @intCast(s.rows), @intCast(s.n), @intCast(p.src()), @intCast(p.dst()), @intCast(stride), 0, 1, @intCast(p.src()) }, 2);
                pass.e.dispatchThreads(mtl.Size.of(p.dst(), s.n, s.rows), mtl.Size.of(@min(128, p.dst()), 1, 1));
            },
            .add, .add_rows, .mul_rows => {
                for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.out }, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
                const op: u32 = switch (p.kind) {
                    .add => 0,
                    .add_rows => 1,
                    else => 3,
                };
                pass.e.setValue([4]u32{ @intCast(s.rows), @intCast(if (p.kind == .add) 1 else s.n), 1, op }, 3);
                pass.e.dispatchThreads(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .norm => {
                const r = p.rows.?;
                for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.ins[2], p.out }, 0..) |ref, i| pass.e.setBuffer(ref.buf, ref.off, i);
                const Params = extern struct { width: u32, kind: u32, rms: u32, has_bias: u32, eps: f32 };
                pass.e.setValue(Params{ .width = @intCast(s.k), .kind = 1, .rms = 0, .has_bias = 1, .eps = r.eps }, 4);
                pass.e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
            .gelu, .gelu_half => {
                pass.e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                pass.e.setBuffer(p.out.buf, p.out.off, 1);
                pass.e.setValue([3]u32{ @intCast(s.rows), if (p.kind == .gelu_half) 2 else 1, 1 }, 2);
                pass.e.dispatchThreads(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    /// Exact kinds: each output word as converted or copied on the host; any NaN where the host has NaN.
    pub fn check(p: *Probe, _: c.Gate) !?[]const u8 {
        const out = try p.rig.outputBytes();
        const half = p.kind == .cast_down;
        for (p.expect, 0..) |want, i| {
            const got: u32 = if (half) std.mem.readInt(u16, out[2 * i ..][0..2], .little) else std.mem.readInt(u32, out[4 * i ..][0..4], .little);
            const nan = if (half) want & 0x7c00 == 0x7c00 and want & 0x3ff != 0 else want & 0x7f800000 == 0x7f800000 and want & 0x7fffff != 0;
            const got_nan = if (half) got & 0x7c00 == 0x7c00 and got & 0x3ff != 0 else got & 0x7f800000 == 0x7f800000 and got & 0x7fffff != 0;
            if (if (nan) !got_nan else got != want) return try std.fmt.allocPrint(p.a, "word {d} is 0x{x} where the host gives 0x{x}", .{ i, got, want });
        }
        return null;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.rows) |r| return r.reference(want, mag);
        for (want, mag, 0..) |*w, *m, i| {
            const x: f64 = p.x[i];
            switch (p.kind) {
                .add, .add_rows, .mul_rows => {
                    const y: f64 = p.y[if (p.kind == .add) i else i % p.shape.n];
                    w.* = if (p.kind == .mul_rows) x * y else x + y;
                    m.* = if (p.kind == .mul_rows) @abs(w.*) else @abs(x) + @abs(y);
                },
                else => {
                    w.* = contract.rows.gelu(x);
                    m.* = 0;
                },
            }
        }
    }

    /// GELU: the formula's erf (Abramowitz-Stegun 7.1.26, within 1.5e-7) and its fp32 evaluation, then the stores.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        // Layer norm's steps past the sums round relative to (x - mean) inv w, which the bias may cancel in the output.
        if (p.rows) |r| for (want, allow, 0..) |w, *l, i| {
            l.* += 11 * u * @abs(w - c.fromBf16(r.b[i % r.shape.k]));
        };
        if (p.kind != .gelu and p.kind != .gelu_half) return;
        const out_unit: f64 = if (p.kind == .gelu_half) 0x1p-11 else u;
        for (want, allow, 0..) |w, *l, i| {
            const x: f64 = p.x[i];
            const erf_error = 1.5e-7 + 64 * u;
            const work = 0.5 * @abs(x) * erf_error + 3 * u * @abs(w);
            const floor: f64 = if (p.kind == .gelu_half) 0x1p-24 else 0x1p-149;
            l.* = work + out_unit * (@abs(w) + work) + floor;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (p.rows) |r| return r.simulate(acc, order, out);
        for (out, 0..) |*o, i| o.* = contract.rows.sumIn(acc, order, &.{ p.x[i], p.y[if (p.kind == .add) i else i % p.shape.n] });
    }
};

const exact_vt = blk: {
    var v = g.hooks(Probe);
    v.bounds = null;
    v.simulate = null;
    break :blk v;
};

const ranged_vt = blk: {
    var v = g.hooks(Probe);
    v.check = null;
    break :blk v;
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        const wanted = struct {
            fn f(needle: ?[]const u8, name: []const u8) bool {
                const n = needle orelse return true;
                return std.mem.indexOf(u8, name, n) != null;
            }
        }.f;
        var any = false;
        for (std.enums.values(Kind)) |k| any = any or wanted(only, label(k));
        if (any) {
            const text = try libraryText(s.a, null);
            defer s.a.free(text);
            const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (std.enums.values(Kind)) |k| if (wanted(only, label(k))) try s.add(device, queue, lib, k, null, kernels);
        }
        for (edits) |e| if (wanted(only, e.label)) {
            const text = try libraryText(s.a, e);
            defer s.a.free(text);
            const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            try s.add(device, queue, lib, e.kind, e, kernels);
        };
    }

    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, kind: Kind, control: ?Edit, kernels: *std.ArrayList(g.Kernel)) !void {
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try mtl.Pipeline.init(device, lib, function(kind), false), .kind = kind, .control = control };
        try s.probes.append(s.a, p);
        try kernels.append(s.a, p.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
