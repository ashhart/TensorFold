//! idol_decoder.metal's kernels under the contract at idol-sdk's points; controls break one rounding, sum or rule each.
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const steps = @import("metal_idol_decoder_steps.zig");
const attn = @import("metal_idol_decoder_attention.zig");

const g = id.g;
const Control = id.Control;
const Lib = steps.Lib;

const pv_lo = "tfp::mma_16x32<false, false>(acc[d], acc[d + 1], pl[k], v0, v1);";
const pv_half = pv_lo ++ " for (short e = 0; e < 8; e++) { acc[d][e] = float(half(acc[d][e])); " ++
    "acc[d + 1][e] = float(half(acc[d + 1][e])); }";
const embed_sum = "const float s = fma(x2, w[c * 3 + 2], fma(x1, w[c * 3 + 1], x0 * w[c * 3])) + b[c];";
const embed_half = "const float s = float(fma(half(x2), half(w[c * 3 + 2]), fma(half(x1), half(w[c * 3 + 1]), " ++
    "half(x0) * half(w[c * 3])))) + b[c];";
const kv_sum = "s = fma(float(ctx[r * 16 + k]), float(w[c * 16 + k]), s);";
const to_half = "o[i] = half(y[i] + b[at.x]);";
/// dec_attention_few's four P V chains, each step rounded to fp16.
const few_half = blk: {
    var out: [4]id.Edit = undefined;
    for (&out, [_][]const u8{ "", " + 1", " + 2", " + 3" }, "xyzw") |*e, at, x| {
        const slot = "o[4 * c" ++ at ++ "]";
        const step = "fma(e, vv." ++ [_]u8{x} ++ ", " ++ slot ++ ")";
        e.* = .{ .from = slot ++ " = " ++ step ++ ";", .to = slot ++ " = float(half(" ++ step ++ "));" };
    }
    break :blk out;
};

const controls = [_]Control{
    .{ .kernel = "dec_embed", .what = "adds fp16(pe)", .edits = &.{.{ .from = "h[ulong(r) * 1024 + c] = s + pe;", .to = "h[ulong(r) * 1024 + c] = s + float(half(pe));" }} },
    .{ .kernel = "dec_embed", .what = "sums in_proj in fp16", .gate = .sums, .edits = &.{.{ .from = embed_sum, .to = embed_half }} },
    .{ .kernel = "dec_half", .what = "adds fp16(y) before the bias", .edits = &.{.{ .from = to_half, .to = "o[i] = half(float(half(y[i])) + b[at.x]);" }} },
    .{ .kernel = "dec_half", .what = "writes its input", .gate = .repeat, .edits = &.{.{ .from = to_half, .to = to_half ++ " ((device float*)y)[i] = 0.0f;" }} },
    .{ .kernel = "dec_kv", .what = "adds fp16(ctx w) before the bias", .edits = &.{.{ .from = "half(s + float(b[c]));", .to = "half(float(half(s)) + float(b[c]));" }} },
    .{ .kernel = "dec_kv", .what = "sums in fp16", .gate = .sums, .edits = &.{.{ .from = kv_sum, .to = "s = float(half(fma(float(ctx[r * 16 + k]), float(w[c * 16 + k]), s)));" }} },
    .{ .kernel = "dec_heads", .what = "clamps the norm by max, which drops a NaN", .gate = .nonfinite, .edits = &.{.{ .from = "const float n = isnan(len) ? len : max(len, 1e-12f);", .to = "const float n = max(len, 1e-12f);" }} },
    .{ .kernel = "dec_heads", .what = "adds fp16(x / n)", .edits = &.{.{ .from = "dst[to] = half(a0 / n * gamma[g] * 8.0f);", .to = "dst[to] = half(float(half(a0 / n)) * gamma[g] * 8.0f);" }} },
    .{ .kernel = "dec_heads", .what = "sums the squares in fp16", .gate = .sums, .edits = &.{.{ .from = "simd_sum(a0 * a0 + a1 * a1)", .to = "float(simd_sum(half(a0 * a0 + a1 * a1)))" }} },
    .{ .kernel = "dec_attention_few", .what = "rounds P to fp16", .edits = &.{.{ .from = "const float e = metal::precise::exp(s[i] - top);", .to = "const float e = float(half(metal::precise::exp(s[i] - top)));" }} },
    .{ .kernel = "dec_attention_few", .what = "sums P V in fp16", .gate = .sums, .edits = &few_half },
    .{ .kernel = "dec_attention_10", .what = "P's lo half meets infinite values", .gate = .nonfinite, .edits = &.{.{ .from = "dec_finite(v0);", .to = "" }} },
    .{ .kernel = "dec_attention_00", .what = "keys past Lk unmasked", .edits = &.{.{ .from = "s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kDecMasked;", .to = "s[f][e] = s[f][e];" }} },
    .{ .kernel = "dec_attention_01", .what = "sums P V in fp16", .gate = .sums, .edits = &.{.{ .from = pv_lo, .to = pv_half }} },
    .{ .kernel = "dec_attention_11", .what = "drops P's lo half", .edits = &.{.{ .from = pv_lo, .to = "" }} },
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
        const ends: id.Ends = .{ .a = s.a };
        const W = struct {
            s: *Suite,
            ctl: ?Control,
            suffix: []const u8,
            only: ?[]const u8,
            /// The test's name when this pass registers `kernel`, else null.
            fn name(w: @This(), kernel: []const u8) !?[]const u8 {
                if (w.ctl) |x| if (!std.mem.eql(u8, x.kernel, kernel)) return null;
                const full = try w.s.label("metal idol decoder {s}{s}", .{ kernel, w.suffix });
                if (w.only) |needle| if (std.mem.indexOf(u8, full, needle) == null) return null;
                return full;
            }
        };
        const w = W{ .s = s, .ctl = ctl, .suffix = suffix, .only = only };
        inline for (.{ .{ steps.Embed, "dec_embed" }, .{ steps.Half, "dec_half" }, .{ steps.Kv, "dec_kv" }, .{ steps.Heads, "dec_heads" } }) |e| {
            if (try w.name(e[1])) |name| {
                const t = try s.make(e[0], .{ .a = s.a, .rig = rig, .lib = lib, .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        for (attn.forms, 0..) |f, i| {
            const name = (try w.name(f.name)) orelse continue;
            const t = try s.make(attn.Attention, .{ .a = s.a, .rig = rig, .lib = lib, .form = i, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        if (!device.tensorUnits()) {
            std.debug.print("UNSUPPORTED metal idol decoder: the family runs on tensor units (M5) alone; its kernels unchecked here\n", .{});
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
