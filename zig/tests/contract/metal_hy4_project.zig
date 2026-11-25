//! Hy4-flash's MLA projection kernels (mla_project.metal, loaded by mla_metal.zig's Projections) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const h = @import("metal_hy4.zig");
const rope_ = @import("metal_hy4_project_rope.zig");
const keys_ = @import("metal_hy4_project_keys.zig");
const dots_ = @import("metal_hy4_project_dots.zig");

const fl = h.fl;
const g = contract.gates;
const Range = fl.Range;

/// Gate 5's names for this source; a spec's name is "metal " and this, its function and its variant.
pub const stem = "hy4 mla_project ";

/// A planted control: its gate-5 slug, its function, the gate that must go red, what it breaks, and the edit.
pub const Control = struct { slug: []const u8, fun: []const u8, gate: g.Gate, what: []const u8, edit: fl.Edit };

pub const controls = [_]Control{
    .{ .slug = "time-axis", .fun = "tf_hy4_rotary_table", .gate = .shapes, .what = "every pair's phase from the time position", .edit = .{ .kernel = "tf_hy4_rotary_table", .from = "positions[ulong(row) * 3 + p.axes[pair]]", .to = "positions[ulong(row) * 3]" } },
    .{ .slug = "turned-back", .fun = "tf_hy4_rotate", .gate = .shapes, .what = "the pair turned the other way", .edit = .{ .kernel = "tf_hy4_rotate", .from = "a * c - b * s", .to = "a * c + b * s" } },
    .{ .slug = "norm-unrounded", .fun = "tf_hy4_cache_keys", .gate = .shapes, .what = "x * inverse's bf16 rounding dropped", .edit = .{ .kernel = "tf_hy4_cache_keys", .from = "hy4_mla_value(hy4_mla_round(v[i] * inverse))", .to = "(v[i] * inverse)" } },
    .{ .slug = "gate-unrounded", .fun = "tf_hy4_gate_values", .gate = .shapes, .what = "the gate's bf16 rounding dropped", .edit = .{ .kernel = "tf_hy4_gate_values", .from = "hy4_mla_value(hy4_mla_round(gate))", .to = "gate" } },
    .{ .slug = "short-sum", .fun = "tf_hy4_absorb_q", .gate = .shapes, .what = "the last of its 192 terms dropped", .edit = .{ .kernel = "tf_hy4_absorb_q", .from = "i < 192", .to = "i < 191" } },
    .{ .slug = "half-sum", .fun = "tf_hy4_absorb_q", .gate = .sums, .what = "summed in half", .edit = .{ .kernel = "tf_hy4_absorb_q", .from = "sum = fma(", .to = "sum = (half)fma(" } },
    .{ .slug = "short-sum", .fun = "tf_hy4_expand_v", .gate = .shapes, .what = "the last of its 512 terms dropped", .edit = .{ .kernel = "tf_hy4_expand_v", .from = "i < 512", .to = "i < 511" } },
    .{ .slug = "gate-unrounded", .fun = "tf_hy4_expand_v_gated", .gate = .shapes, .what = "the gate's bf16 rounding dropped", .edit = .{ .kernel = "tf_hy4_expand_v_gated", .from = "hy4_mla_value(hy4_mla_round(gate))", .to = "gate" } },
};

/// The control planted in `fun` under `slug`; a comptime miss is a build error.
pub fn control(comptime fun: []const u8, comptime slug: []const u8) *const Control {
    inline for (&controls) |*x| if (comptime std.mem.eql(u8, x.fun, fun) and std.mem.eql(u8, x.slug, slug)) return x;
    @compileError("no control " ++ slug ++ " in " ++ fun);
}

/// One of mla_project.metal's kernels compiled as Projections.load compiles it, or a control's edited copy.
pub fn pipe(a: std.mem.Allocator, device: mtl.Device, fun: []const u8, ctl: ?*const Control) !mtl.Pipeline {
    const text = try fl.edited(a, sources.hy4_flash_mla_project, if (ctl) |x| x.edit else null);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, fun, false);
}

/// A test's name: "metal hy4 mla_project <fun> <variant>", or its control's.
pub fn name(buf: []u8, fun: []const u8, variant: []const u8, ctl: ?*const Control) []const u8 {
    var base: [160]u8 = undefined;
    const b = std.fmt.bufPrint(&base, "metal {s}{s}{s}{s}", .{ stem, fun, if (variant.len > 0) " " else "", variant }) catch return fun;
    if (ctl == null) return std.fmt.bufPrint(buf, "{s}", .{b}) catch fun;
    return fl.named(b, ctl.?.what, buf);
}

/// Metal's precise functions land within `n` ulps of the true value (an ulp at most 2^-23 of it), or flush to zero.
pub fn ulps(r: Range, n: f64) Range {
    const e = n * 0x1p-23;
    return .{ .lo = r.lo - e * @abs(r.lo) - fl.tiny, .hi = r.hi + e * @abs(r.hi) + fl.tiny };
}

/// 1 / (1 + precise::exp(-x)) in fp32 at a finite x: exp within 4 ulps, the add's rounding, the divide's 2.5 ulps.
pub fn sigmoid(x: f64) Range {
    const den = ulps(Range.point(@exp(-x)), 4).plus(Range.point(1)).fp32();
    return ulps(.{ .lo = 1 / den.hi, .hi = 1 / den.lo }, 2.5);
}

/// The float64 sigmoid the gates' references take before its bf16 point.
pub fn sigmoid64(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// One rotated pair (a, b) by bf16 (c, s): each product of two bf16 values is exact, so this is the fp32 sum's input.
pub fn turn(a: f64, b: f64, cs: f64, sn: f64) [2]f64 {
    return .{ a * cs - b * sn, b * cs + a * sn };
}

/// A rotated value's range: one fp32 add of exact products, then the bf16 store; any range at a non-finite value.
pub fn turned(v: f64) Range {
    if (!std.math.isFinite(v)) return .{ .lo = -std.math.floatMax(f64), .hi = std.math.floatMax(f64) };
    return Range.point(v).fp32().bf16();
}

/// A rotary table's rows as tf_hy4_rotary_table leaves them: bf16 cos and sin of each pair's angle.
pub fn angles(rng: std.Random, tab: []u16) void {
    var i: usize = 0;
    while (i < tab.len) : (i += 2) {
        const t = rng.float(f64) * 400;
        tab[i] = contract.spec.toBf16(@floatCast(@cos(t)));
        tab[i + 1] = contract.spec.toBf16(@floatCast(@sin(t)));
    }
}

/// Uniform bf16 values in [-r, r).
pub fn uniform(rng: std.Random, out: []u16, r: f32) void {
    for (out) |*v| v.* = contract.spec.toBf16((rng.float(f32) * 2 - 1) * r);
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tables: fl.List(rope_.Table) = .{},
    turns: fl.List(rope_.Rotate) = .{},
    keys: fl.List(keys_.Keys) = .{},
    gates: fl.List(keys_.Gate) = .{},
    dots: fl.List(dots_.Dot) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try rope_.collect(s, only);
        try keys_.collect(s, only);
        try dots_.collect(s, only);
    }
    pub fn deinit(s: *Suite) void {
        s.tables.deinit(s.a);
        s.turns.deinit(s.a);
        s.keys.deinit(s.a);
        s.gates.deinit(s.a);
        s.dots.deinit(s.a);
    }
};

/// Gate 5's texts for "hy4 mla_project [slug] <fun>": the source as Projections.load compiles it, or a control's edit.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, label: []const u8) !?fl.Texts {
    if (!std.mem.startsWith(u8, label, stem)) return null;
    const fun = label[std.mem.lastIndexOfScalar(u8, label, ' ').? + 1 ..];
    const slug = std.mem.trim(u8, label[stem.len .. label.len - fun.len], " ");
    if (slug.len == 0) return .{ .here = sources.hy4_flash_mla_project, .pre_m5 = sources.hy4_flash_mla_project };
    for (&controls) |*x| if (std.mem.eql(u8, x.fun, fun) and std.mem.eql(u8, x.slug, slug)) {
        const text = try fl.edited(a, sources.hy4_flash_mla_project, x.edit);
        return .{ .here = text, .pre_m5 = text };
    };
    return error.UnknownLaunch;
}
