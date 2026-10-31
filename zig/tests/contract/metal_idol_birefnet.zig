//! Idol's background remover's kernels (idol_birefnet.metal, fp32) under the contract; each control breaks one gate.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const cv = @import("metal_idol_birefnet_conv.zig");
const rw = @import("metal_idol_birefnet_rows.zig");
const nm = @import("metal_idol_birefnet_norm.zig");
const wn = @import("metal_idol_birefnet_window.zig");

const g = id.g;
const Ends = id.Ends;

/// A control: the kernel it breaks, the gate that must catch it, and the one text change that breaks it.
const Control = struct { kernel: []const u8, gate: g.Gate, what: []const u8, from: []const u8, to: []const u8 };

const flush = "acc[i][j].thread_elements()[0] += part[i][j].thread_elements()[0];";
const flush_half = "acc[i][j].thread_elements()[0] = float(half(acc[i][j].thread_elements()[0]) + " ++
    "half(part[i][j].thread_elements()[0]));";
const gelu = "if (p.flags & 4) v = (v * 0.5f) * (1.0f + br_erf(v * 0.70710677f));";
const gelu_tanh = "if (p.flags & 4) v = (v * 0.5f) * (1.0f + precise::tanh(0.7978846f * (v + 0.044715f * v * v * v)));";
const bias_at = "const uint at = ((qi / 12 + 11 - j / 12) * 23 + (qi % 12 + 11 - j % 12)) * p.heads + h;";
const bias_swapped = "const uint at = ((qi % 12 + 11 - j % 12) * 23 + (qi / 12 + 11 - j / 12)) * p.heads + h;";
const norm_sum = "for (uint c = t; c < p.width; c += 256) s += br_norm_at(x, map, p, row, c);";
const norm_sum_half = "for (uint c = t; c < p.width; c += 256) s = float(half(s) + half(br_norm_at(x, map, p, row, c)));";
const pool_sum = "for (uint i = t; i < p.pixels; i += 256) s += x[ulong(i) * p.channels + ch];";
const pool_sum_half = "for (uint i = t; i < p.pixels; i += 256) s = float(half(s) + half(x[ulong(i) * p.channels + ch]));";
const outside = "if (!(h > -1.0f && h < float(H) && w > -1.0f && w < float(W))) {";
const outside_floor = "if (!(h >= 0.0f && h < float(H) && w >= 0.0f && w < float(W))) {";

const controls = [_]Control{
    .{ .kernel = "br_conv 1x1", .gate = .shapes, .what = "stages its inputs in half", .from = "for (uint i = 0; i < 4; i++) dst[i] = src[i];", .to = "for (uint i = 0; i < 4; i++) dst[i] = float4(half4(src[i]));" },
    .{ .kernel = "br_conv 3x3", .gate = .shapes, .what = "reads its taps a row low", .from = "const int iy = oy + int(tap / p.k) - pad;", .to = "const int iy = oy + int(tap / p.k) - pad + 1;" },
    .{ .kernel = "br_conv 1x1", .gate = .sums, .what = "adds its blocks in half", .from = flush, .to = flush_half },
    .{ .kernel = "br_conv 1x1 gelu", .gate = .shapes, .what = "GELU's tanh form", .from = gelu, .to = gelu_tanh },
    .{ .kernel = "br_conv 1x1 bn relu", .gate = .nonfinite, .what = "ReLU by max (drops a NaN)", .from = "if (p.flags & 2) v = v < 0.0f ? 0.0f : v;", .to = "if (p.flags & 2) v = max(v, 0.0f);" },
    .{ .kernel = "br_conv 1x1 + residual by map", .gate = .shapes, .what = "subtracts its residual", .from = "if (p.flags & 8) v = r[at] + v;", .to = "if (p.flags & 8) v = r[at] - v;" },
    .{ .kernel = "br_conv3", .gate = .shapes, .what = "drops its bias", .from = "y[ulong(at.y) * p.cout + at.x] = s + b[at.x];", .to = "y[ulong(at.y) * p.cout + at.x] = s;" },
    .{ .kernel = "br_conv3", .gate = .sums, .what = "sums in half", .from = "s = fma(v[c], w[((ky * 3 + kx) * 3 + c) * p.cout + at.x], s);", .to = "s = float(half(s) + half(v[c] * w[((ky * 3 + kx) * 3 + c) * p.cout + at.x]));" },
    .{ .kernel = "br_norm", .gate = .shapes, .what = "the unbiased variance", .from = "const float rstd = 1.0f / precise::sqrt(all / float(p.width) + p.eps);", .to = "const float rstd = 1.0f / precise::sqrt(all / float(p.width - 1) + p.eps);" },
    .{ .kernel = "br_norm", .gate = .sums, .what = "sums in half", .from = norm_sum, .to = norm_sum_half },
    .{ .kernel = "br_norm map", .gate = .shapes, .what = "leaves the padding rows unwritten", .from = "for (uint c = t; c < p.width; c += 256) out[c] = 0.0f;", .to = "" },
    .{ .kernel = "br_norm merge", .gate = .shapes, .what = "swaps x1 and x2", .from = "const uint sy = (r / half_w) * 2 + (part & 1);", .to = "const uint sy = (r / half_w) * 2 + (part >> 1);" },
    .{ .kernel = "br_window shift 6", .gate = .shapes, .what = "drops the shifted windows' mask", .from = "if (p.shift) v = v + (br_region(j, wy, wx, p) == mine ? 0.0f : -100.0f);", .to = "" },
    .{ .kernel = "br_window", .gate = .shapes, .what = "swaps the bias table's axes", .from = bias_at, .to = bias_swapped },
    .{ .kernel = "br_resize", .gate = .shapes, .what = "weighs columns by the row's lambda", .from = "const float top = (1.0f - lx) * v00 + lx * v01;", .to = "const float top = (1.0f - ly) * v00 + ly * v01;" },
    .{ .kernel = "br_tiles grid", .gate = .shapes, .what = "takes the tile's row from its column", .from = "const uint sy = p.mode == 0 ? oy * p.a + i : i * p.out_h + oy;", .to = "const uint sy = p.mode == 0 ? oy * p.a + i : j * p.out_h + oy;" },
    .{ .kernel = "br_deform 3x3", .gate = .shapes, .what = "a modulator of sigmoid, not twice it", .from = "const float mask = 2.0f * br_sigmoid(off[2 * taps + at.x]);", .to = "const float mask = br_sigmoid(off[2 * taps + at.x]);" },
    .{ .kernel = "br_deform 7x7", .gate = .shapes, .what = "zeroes samples within a pixel of the border", .from = outside, .to = outside_floor },
    .{ .kernel = "br_pool", .gate = .sums, .what = "sums in half", .from = pool_sum, .to = pool_sum_half },
    .{ .kernel = "br_gate", .gate = .shapes, .what = "drops the gate's bias", .from = "x[o] = x[o] * br_sigmoid(s + b[0]);", .to = "x[o] = x[o] * br_sigmoid(s);" },
    .{ .kernel = "br_head", .gate = .shapes, .what = "drops ipt_blk1's channels", .from = "for (uint c = 0; c < p.cq; c++) s = fma(qr[c], w[p.cp + c], s);", .to = "" },
};

/// The remover's source as the engine compiles it, with `ctl`'s change made where it occurs, exactly once.
fn compile(a: std.mem.Allocator, device: mtl.Device, ctl: ?Control) !mtl.Library {
    const text = if (ctl) |x| blk: {
        if (std.mem.count(u8, sources.idol_birefnet, x.from) != 1) {
            std.debug.print("control {s}: text not found once: {s}\n", .{ x.what, x.from });
            return error.ControlTextNotFoundOnce;
        }
        break :blk try std.mem.replaceOwned(u8, a, sources.idol_birefnet, x.from, x.to);
    } else try a.dupe(u8, sources.idol_birefnet);
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
                const full = try w.s.label("metal idol birefnet {s}{s}", .{ kernel, w.suffix });
                if (w.only) |needle| if (std.mem.indexOf(u8, full, needle) == null) return null;
                return full;
            }
        };
        const w = W{ .s = s, .ctl = ctl, .suffix = suffix, .only = only };
        for (cv.conv_forms) |f| {
            const name = (try w.name(f.label)) orelse continue;
            const t = try s.make(cv.Conv, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_conv"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("br_conv3")) |name| {
            const t = try s.make(cv.Conv3, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_conv3"), .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]nm.NormForm{ .rows, .map, .merge }, [_][]const u8{ "br_norm", "br_norm map", "br_norm merge" }) |f, label_| {
            const name = (try w.name(label_)) orelse continue;
            const t = try s.make(nm.Norm, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_norm"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]usize{ 0, 6 }, [_][]const u8{ "br_window", "br_window shift 6" }) |shift, label_| {
            const name = (try w.name(label_)) orelse continue;
            const t = try s.make(wn.Window, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_window"), .shift = shift, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("br_resize")) |name| {
            const t = try s.make(rw.Resize, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_resize"), .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ false, true }, [_][]const u8{ "br_tiles", "br_tiles grid" }) |grid, label_| {
            const name = (try w.name(label_)) orelse continue;
            const t = try s.make(rw.Tiles, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_tiles"), .grid = grid, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]usize{ 1, 3, 7 }, [_][]const u8{ "br_deform 1x1", "br_deform 3x3", "br_deform 7x7" }) |side, label_| {
            const name = (try w.name(label_)) orelse continue;
            const t = try s.make(nm.Deform, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_deform"), .side = side, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("br_pool")) |name| {
            const t = try s.make(rw.Pool, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_pool"), .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("br_gate")) |name| {
            const t = try s.make(rw.Gate, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_gate"), .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("br_head")) |name| {
            const t = try s.make(rw.Head, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "br_head"), .name = name });
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
