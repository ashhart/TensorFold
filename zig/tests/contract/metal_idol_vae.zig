//! Idol's image encoder kernels (idol_vae.metal, fp32) under the contract; each control breaks one gate.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const st = @import("metal_idol_vae_steps.zig");
const wn = @import("metal_idol_vae_wino.zig");
const rw = @import("metal_idol_vae_rows.zig");

const g = id.g;
const Ends = id.Ends;

/// A control: the kernel it breaks, the gate that must catch it, and the one text change that breaks it.
const Control = struct { kernel: []const u8, gate: g.Gate, what: []const u8, from: []const u8, to: []const u8 };

const wino_mma = "for (uint i = 0; i < 2; i++) simdgroup_multiply_accumulate(acc[i][j], vm[i], um, acc[i][j]);";
const wino_mma_half = "for (uint i = 0; i < 2; i++) { simdgroup_multiply_accumulate(acc[i][j], vm[i], um, acc[i][j]); " ++
    "acc[i][j].thread_elements()[0] = float(half(acc[i][j].thread_elements()[0])); " ++
    "acc[i][j].thread_elements()[1] = float(half(acc[i][j].thread_elements()[1])); }";
const patch_axes = "const uint py = (patch / w2) * 2 + (at.x % 4) / 2;\n  const uint px = (patch % w2) * 2 + at.x % 2;";
const patch_swapped = "const uint py = (patch / w2) * 2 + at.x % 2;\n  const uint px = (patch % w2) * 2 + (at.x % 4) / 2;";

const controls = [_]Control{
    .{ .kernel = "iv_conv_in", .gate = .shapes, .what = "stages 2 x - 1 in half", .from = "s = fma(v[c] * 2.0f - 1.0f,", .to = "s = fma(float(half(v[c] * 2.0f - 1.0f))," },
    .{ .kernel = "iv_conv 1x1", .gate = .shapes, .what = "stages its inputs in half", .from = "for (uint i = 0; i < 4; i++) dst[i] = src[i];", .to = "for (uint i = 0; i < 4; i++) dst[i] = float4(half4(src[i]));" },
    .{ .kernel = "iv_conv 3x3 stride 2", .gate = .shapes, .what = "pads above as padding=1 would", .from = "const uint iy = oy * stride + tap / 3;", .to = "const uint iy = oy * stride + tap / 3 - (p.taps == 9 ? 1 : 0);" },
    .{ .kernel = "iv_wino 3x3", .gate = .shapes, .what = "stages its patch in half", .from = "*(threadgroup float4*)(buf + px * BK + q * 4) = v;", .to = "*(threadgroup float4*)(buf + px * BK + q * 4) = float4(half4(v));" },
    .{ .kernel = "iv_wino 3x3 + residual", .gate = .shapes, .what = "drops a term of the output transform", .from = "s[j][1] = (m1 - m2) - m3;", .to = "s[j][1] = (m1 - m2);" },
    .{ .kernel = "iv_wino 3x3", .gate = .sums, .what = "sums in half", .from = wino_mma, .to = wino_mma_half },
    .{ .kernel = "iv_norm silu", .gate = .shapes, .what = "its SiLU in half", .from = "y[i] = p.silu ? v / (1.0f + precise::exp(-v)) : v;", .to = "y[i] = p.silu ? float(half(v) / (1.0h + exp(-half(v)))) : v;" },
    .{ .kernel = "iv_group_part", .gate = .sums, .what = "sums in half", .from = "s += p.centered ? d * d : v;", .to = "s = float(half(s) + half(p.centered ? d * d : v));" },
    .{ .kernel = "iv_group_finish", .gate = .sums, .what = "sums in half", .from = "for (uint i = lane; i < p.chunks; i += 32) s += part[sg * p.chunks + i];", .to = "for (uint i = lane; i < p.chunks; i += 32) s = float(half(s) + half(part[sg * p.chunks + i]));" },
    .{ .kernel = "iv_attention", .gate = .shapes, .what = "keeps stale maxima (no rescale)", .from = "    if (rescale) {", .to = "    if (false) {" },
    .{ .kernel = "iv_tokens", .gate = .shapes, .what = "swaps the patch's axes", .from = patch_axes, .to = patch_swapped },
    .{ .kernel = "iv_tokens", .gate = .nonfinite, .what = "clamps a NaN logvar (Metal's clamp)", .from = "const float logvar = isnan(raw) ? raw : clamp(raw, -30.0f, 20.0f);", .to = "const float logvar = clamp(raw, -30.0f, 20.0f);" },
};

/// The encoder's source as the engine compiles it, with `ctl`'s change made where it occurs, exactly once.
fn compile(a: std.mem.Allocator, device: mtl.Device, ctl: ?Control) !mtl.Library {
    const text = if (ctl) |x| blk: {
        if (std.mem.count(u8, sources.idol_vae, x.from) != 1) {
            std.debug.print("control {s}: text not found once: {s}\n", .{ x.what, x.from });
            return error.ControlTextNotFoundOnce;
        }
        break :blk try std.mem.replaceOwned(u8, a, sources.idol_vae, x.from, x.to);
    } else try a.dupe(u8, sources.idol_vae);
    defer a.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

/// A test the suite made, and how it goes.
const Made = struct { ptr: *anyopaque, free: *const fn (std.mem.Allocator, *anyopaque) void };

pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,
    names: std.ArrayList([]u8) = .empty,
    libs: std.ArrayList(mtl.Library) = .empty,
    made: std.ArrayList(Made) = .empty,

    fn pipeline(s: *Suite, device: mtl.Device, lib: mtl.Library, name: []const u8) !mtl.Pipeline {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        errdefer p.deinit();
        try s.pipes.append(s.a, p);
        return p;
    }

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
    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, ctl: ?Control, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        const suffix = if (ctl) |x| try s.label(" control: {s}", .{x.what}) else "";
        const gate: ?g.Gate = if (ctl) |x| x.gate else null;
        const rig: rig_.Rig = .{ .a = s.a, .device = device, .queue = queue };
        const ends: Ends = .{ .a = s.a };
        const W = struct {
            s: *Suite,
            ctl: ?Control,
            suffix: []const u8,
            only: ?[]const u8,
            /// The test's name when this pass registers `kernel`, else null.
            fn name(w: @This(), kernel: []const u8) !?[]const u8 {
                if (w.ctl) |x| if (!std.mem.eql(u8, x.kernel, kernel)) return null;
                const full = try w.s.label("metal idol vae {s}{s}", .{ kernel, w.suffix });
                if (w.only) |needle| if (std.mem.indexOf(u8, full, needle) == null) return null;
                return full;
            }
        };
        const w = W{ .s = s, .ctl = ctl, .suffix = suffix, .only = only };
        if (try w.name("iv_conv_in")) |name| {
            const t = try s.make(st.ConvIn, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_conv_in"), .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for (st.conv_forms) |f| {
            const name = (try w.name(f.label)) orelse continue;
            const t = try s.make(st.Conv, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_conv"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for (wn.forms) |f| {
            const name = (try w.name(f.label)) orelse continue;
            const t = try s.make(wn.Wino, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_wino"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ true, false }) |silu| {
            const name = (try w.name(if (silu) "iv_norm silu" else "iv_norm")) orelse continue;
            const t = try s.make(st.Norm, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_norm"), .silu = silu, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ false, true }) |centered| {
            const name = (try w.name(if (centered) "iv_group_part centered" else "iv_group_part")) orelse continue;
            const t = try s.make(rw.GroupPart, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_group_part"), .centered = centered, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("iv_group_finish")) |name| {
            const t = try s.make(rw.GroupFinish, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_group_finish"), .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("iv_attention")) |name| {
            const t = try s.make(rw.Attention, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_attention"), .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("iv_tokens")) |name| {
            const t = try s.make(st.Tokens, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "iv_tokens"), .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for ([_]?Control{null} ++ controls) |x| {
            const lib = try compile(a, device, x);
            s.libs.append(a, lib) catch |err| {
                lib.deinit();
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
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(a);
        for (s.libs.items) |l| l.deinit();
        s.libs.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }
};
