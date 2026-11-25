//! MiniMax-H3-Image's decoder kernels (image_decoder.metal) under the contract: each against its formula in float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const common = @import("metal_h3_image_common.zig");
const parts = @import("metal_h3_image_decoder_parts.zig");
const rows = @import("metal_h3_image_decoder_rows.zig");
const norms = @import("metal_h3_image_decoder_norms.zig");
const gemm = @import("metal_h3_image_decoder_gemm.zig");
const heads = @import("metal_h3_image_decoder_heads.zig");
const g = contract.gates;

/// A registration: one kernel at one form; the gemm's LayerScale form is mode 1 of h3d_gemm.
const Kind = enum { post_quant, embed, rms_norm, layer_norm, gemm, gemm_scale, heads, attention, swiglu };

const kinds = [_]Kind{ .post_quant, .embed, .rms_norm, .layer_norm, .gemm, .gemm_scale, .heads, .attention, .swiglu };

fn Of(comptime kind: Kind) type {
    return switch (kind) {
        .post_quant => rows.PostQuant,
        .embed => rows.Embed,
        .rms_norm => norms.RmsNorm,
        .layer_norm => norms.LayerNorm,
        .gemm, .gemm_scale => gemm.Gemm,
        .heads => heads.Heads,
        .attention => heads.Attention,
        .swiglu => rows.Swiglu,
    };
}

fn function(comptime kind: Kind) []const u8 {
    return switch (kind) {
        .gemm, .gemm_scale => "h3d_gemm",
        else => "h3d_" ++ @tagName(kind),
    };
}

fn label(comptime kind: Kind) []const u8 {
    return comptime common.name(.decoder, function(kind)) ++ (if (kind == .gemm_scale) " layer scale" else "");
}

fn initial(comptime kind: Kind) Of(kind) {
    return switch (kind) {
        .gemm => .{ .layer_scale = false },
        .gemm_scale => .{ .layer_scale = true },
        else => .{},
    };
}

/// A planted control: the registration it breaks, the gate that must catch it, and its one source edit.
const Control = struct { kind: Kind, gate: g.Gate, what: []const u8, old: []const u8, new: []const u8 };

const controls = [_]Control{
    .{ .kind = .post_quant, .gate = .shapes, .what = "drops the latent's bf16 rounding point", .old = "float(bfloat(lat[k * plane + r * p.cols + c]))", .new = "lat[k * plane + r * p.cols + c]" },
    .{ .kind = .embed, .gate = .shapes, .what = "puts the registers after the zero token", .old = "} else if (t < cells + 4) {\n    v = reg[(t - cells) * 1024 + at.x];", .new = "} else if (t > cells) {\n    v = reg[(t - cells - 1) * 1024 + at.x];" },
    .{ .kind = .rms_norm, .gate = .sums, .what = "sums its simdgroups in half", .old = "v = simd_sum(v);", .new = "v = float(simd_sum(half(v)));" },
    .{ .kind = .layer_norm, .gate = .shapes, .what = "drops its mean", .old = "const float mean = h3d_total(s, sums, lane, sg) / 1024.0f;", .new = "const float mean = 0.0f * h3d_total(s, sums, lane, sg);" },
    .{ .kind = .gemm, .gate = .shapes, .what = "drops its last K step", .old = "for (uint k0 = 0; k0 < p.k; k0 += BK) {", .new = "for (uint k0 = 0; k0 + BK < p.k; k0 += BK) {" },
    .{ .kind = .gemm_scale, .gate = .shapes, .what = "adds LayerScale's product unrounded", .old = "float(bfloat(out * float(s[col])))", .new = "out * float(s[col])" },
    .{ .kind = .heads, .gate = .shapes, .what = "turns the second half of each head", .old = "const float a = float(src[part * 64 + lane]);\n    const float z = float(src[part * 64 + 32 + lane]);", .new = "const float a = float(src[part * 64 + 32 + lane]);\n    const float z = float(src[part * 64 + lane]);" },
    .{ .kind = .attention, .gate = .shapes, .what = "drops its last key", .old = "for (uint key = 0; key < T; key++) {", .new = "for (uint key = 0; key + 1 < T; key++) {" },
    .{ .kind = .swiglu, .gate = .shapes, .what = "swaps gate and value", .old = "const float g = float(r[at.x]);\n  const float silu = float(bfloat(g / (1.0f + precise::exp(-g))));\n  act[ulong(at.y) * 4096 + at.x] = bfloat(silu * float(r[4096 + at.x]));", .new = "const float g = float(r[4096 + at.x]);\n  const float silu = float(bfloat(g / (1.0f + precise::exp(-g))));\n  act[ulong(at.y) * 4096 + at.x] = bfloat(silu * float(r[at.x]));" },
    .{ .kind = .swiglu, .gate = .nonfinite, .what = "takes a NaN gate as zero", .old = "const float g = float(r[at.x]);", .new = "const float g = isnan(float(r[at.x])) ? 0.0f : float(r[at.x]);" },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    libs: std.ArrayList(mtl.Library) = .empty,
    names: std.ArrayList([]u8) = .empty,
    cases: std.ArrayList(Owned) = .empty,

    /// A registration's test, freed with the suite.
    const Owned = struct { ptr: *anyopaque, free: *const fn (*anyopaque, std.mem.Allocator) void };

    fn Free(comptime T: type) type {
        return struct {
            fn free(p: *anyopaque, a: std.mem.Allocator) void {
                const t: *T = @ptrCast(@alignCast(p));
                t.deinit();
                a.destroy(t);
            }
        };
    }

    /// `kind`'s test on `lib`'s pipeline, named `name`, appended to `out` (a control when `control` names a gate).
    fn add(s: *Suite, comptime kind: Kind, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, name: []const u8, control: ?g.Gate, out: *std.ArrayList(g.Kernel)) !void {
        const T = parts.Case(Of(kind));
        const pipe = try mtl.Pipeline.init(device, lib, function(kind), false);
        const t = s.a.create(T) catch |err| {
            pipe.deinit();
            return err;
        };
        t.* = T.init(s.a, device, queue, pipe, name, initial(kind));
        s.cases.append(s.a, .{ .ptr = t, .free = Free(T).free }) catch |err| {
            Free(T).free(t, s.a);
            return err;
        };
        try out.append(s.a, t.kernel(control));
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const lib = try common.library(device, .decoder, common.source(.decoder));
        s.libs.append(a, lib) catch |err| {
            lib.deinit();
            return err;
        };
        inline for (kinds) |kind| if (common.keep(only, label(kind))) try s.add(kind, device, queue, lib, label(kind), null, kernels);
        inline for (controls) |x| {
            const name = try std.fmt.allocPrint(a, "{s} control: {s}", .{ label(x.kind), x.what });
            s.names.append(a, name) catch |err| {
                a.free(name);
                return err;
            };
            if (common.keep(only, name)) {
                const text = try common.edited(a, .decoder, x.old, x.new);
                defer a.free(text);
                const edited = try common.library(device, .decoder, text);
                s.libs.append(a, edited) catch |err| {
                    edited.deinit();
                    return err;
                };
                try s.add(x.kind, device, queue, edited, name, x.gate, kernels);
            }
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        for (s.cases.items) |o| o.free(o.ptr, a);
        s.cases.deinit(a);
        for (s.libs.items) |l| l.deinit();
        s.libs.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }
};
