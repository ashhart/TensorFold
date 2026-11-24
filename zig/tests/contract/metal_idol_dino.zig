//! Idol's DINOv3 encoder kernels under the contract, each at transformers' bf16 points; controls break one point each.
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("rig.zig");
const dn = @import("metal_idol_dino_common.zig");
const gemm = @import("metal_idol_dino_gemm.zig");
const norms = @import("metal_idol_dino_norms.zig");
const elems = @import("metal_idol_dino_rows.zig");
const attn = @import("metal_idol_dino_attention.zig");

const g = dn.g;
const Control = dn.Control;
const Lib = dn.Lib;

const controls = [_]Control{
    .{ .kernel = "dn_linear_0", .what = "adds bf16(x w) before the bias", .edits = &.{.{ .from = "acc[i][j][e] = dn_b(acc[i][j][e] + b);", .to = "acc[i][j][e] = dn_b(dn_b(acc[i][j][e]) + b);" }} },
    .{ .kernel = "dn_linear_1", .what = "adds bf16(x w) before the bias", .edits = &.{.{ .from = "acc[i][j][e] = dn_b(acc[i][j][e] + b);", .to = "acc[i][j][e] = dn_b(dn_b(acc[i][j][e]) + b);" }} },
    .{ .kernel = "dn_norm", .what = "adds bf16(n) before the affine", .edits = &.{.{ .from = "yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));", .to = "yr[i] = bfloat(dn_b((float(xr[i]) - mean) * inv) * float(w[i]) + float(b[i]));" }} },
    .{ .kernel = "dn_norm", .what = "sums the simdgroups in fp16", .gate = .sums, .edits = &.{.{ .from = "  v = simd_sum(v);\n  if (lane == 0) part[sg] = v;", .to = "  v = float(simd_sum(half(v)));\n  if (lane == 0) part[sg] = v;" }} },
    .{ .kernel = "dn_final", .what = "drops bf16(h) before F.layer_norm", .edits = &.{.{ .from = "    v[i] = float(o);", .to = "    v[i] = (float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]);" }} },
    .{ .kernel = "dn_final", .what = "sums F.layer_norm's squares in fp16", .gate = .sums, .edits = &.{.{ .from = "q2 += (v[i] - mean2) * (v[i] - mean2);", .to = "q2 = float(half(q2 + (v[i] - mean2) * (v[i] - mean2)));" }} },
    .{ .kernel = "dn_heads", .what = "drops bf16(q cos)", .edits = &.{.{ .from = "Q[to] = bfloat(dn_b(q0 * c0) + dn_b(-q1 * s0));", .to = "Q[to] = bfloat((q0 * c0) + dn_b(-q1 * s0));" }} },
    .{ .kernel = "dn_scale_add", .what = "drops bf16(g y)", .edits = &.{.{ .from = "x[i] = bfloat(float(x[i]) + dn_b(float(g[at.x]) * float(y[i])));", .to = "x[i] = bfloat(float(x[i]) + (float(g[at.x]) * float(y[i])));" }} },
    .{ .kernel = "dn_swiglu", .what = "drops bf16(silu(g))", .edits = &.{.{ .from = "bfloat(dn_b(g / (1.0f + metal::precise::exp(-g))) * float(gu[i + width]));", .to = "bfloat((g / (1.0f + metal::precise::exp(-g))) * float(gu[i + width]));" }} },
    .{ .kernel = "dn_attention", .what = "keys past Lk unmasked", .edits = &.{.{ .from = "s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kDnMasked;", .to = "s[f][e] = s[f][e];" }} },
    .{ .kernel = "dn_attention", .what = "rounds the P V sums to bf16 each key block", .gate = .sums, .edits = &.{.{ .from = "for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];", .to = "for (short e = 0; e < 8; e++) acc[i][e] = float(bfloat(acc[i][e] * scale_old[e >> 2]));" }} },
};

/// A test the suite made, and how it goes.
const Made = struct { ptr: *anyopaque, free: *const fn (std.mem.Allocator, *anyopaque) void };

pub const Suite = struct {
    a: std.mem.Allocator,
    libs: std.ArrayList(*Lib) = .empty,
    names: std.ArrayList([]u8) = .empty,
    made: std.ArrayList(Made) = .empty,

    fn label(s: *Suite, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.allocPrint(s.a, fmt, args);
        errdefer s.a.free(name);
        try s.names.append(s.a, name);
        return name;
    }

    fn make(s: *Suite, comptime T: type, value: T) !*T {
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = value;
        const f = struct {
            fn free(a: std.mem.Allocator, p: *anyopaque) void {
                const x: *T = @ptrCast(@alignCast(p));
                x.deinit();
                a.destroy(x);
            }
        };
        try s.made.append(s.a, .{ .ptr = t, .free = f.free });
        return t;
    }

    /// Each kernel (or, with `ctl`, the one it breaks) on `lib`'s pipelines.
    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: *Lib, ctl: ?Control, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        const suffix = if (ctl) |x| try s.label(" control: {s}", .{x.what}) else "";
        const gate: ?g.Gate = if (ctl) |x| x.gate else null;
        const rig: rig_.Rig = .{ .a = s.a, .device = device, .queue = queue };
        const ends: dn.Ends = .{ .a = s.a };
        const W = struct {
            s: *Suite,
            ctl: ?Control,
            suffix: []const u8,
            only: ?[]const u8,
            /// The test's name when this pass registers `kernel`, else null.
            fn name(w: @This(), kernel: []const u8) !?[]const u8 {
                if (w.ctl) |x| if (!std.mem.eql(u8, x.kernel, kernel)) return null;
                const full = try w.s.label("metal idol dino {s}{s}", .{ kernel, w.suffix });
                if (w.only) |needle| if (std.mem.indexOf(u8, full, needle) == null) return null;
                return full;
            }
        };
        const w = W{ .s = s, .ctl = ctl, .suffix = suffix, .only = only };
        for (gemm.forms, 0..) |fun, f| {
            const name = (try w.name(fun)) orelse continue;
            const t = try s.make(gemm.Linear, .{ .a = s.a, .rig = rig, .lib = lib, .form = @intCast(f), .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]norms.Kind{ .norm, .final }, [_][]const u8{ "dn_norm", "dn_final" }) |kind, fun| {
            const name = (try w.name(fun)) orelse continue;
            const t = try s.make(norms.Norm, .{ .a = s.a, .rig = rig, .lib = lib, .kind = kind, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        inline for (.{ .{ elems.Heads, "dn_heads" }, .{ elems.ScaleAdd, "dn_scale_add" }, .{ elems.Swiglu, "dn_swiglu" } }) |e| {
            if (try w.name(e[1])) |name| {
                const t = try s.make(e[0], .{ .a = s.a, .rig = rig, .lib = lib, .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        if (try w.name("dn_attention")) |name| {
            const t = try s.make(attn.Attention, .{ .a = s.a, .rig = rig, .lib = lib, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        if (!device.tensorUnits()) {
            std.debug.print("UNSUPPORTED metal idol dino: the encoder runs on tensor units (M5) alone; its kernels unchecked here\n", .{});
            return s;
        }
        for ([_]?Control{null} ++ controls) |x| {
            const lib = try a.create(Lib);
            lib.* = .{ .a = a, .device = device, .ctl = x };
            s.libs.append(a, lib) catch |err| {
                a.destroy(lib);
                return err;
            };
            try s.add(device, queue, lib, x, out, only);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        for (s.made.items) |m| m.free(a, m.ptr);
        s.made.deinit(a);
        for (s.libs.items) |l| {
            l.deinit();
            a.destroy(l);
        }
        s.libs.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }
};
