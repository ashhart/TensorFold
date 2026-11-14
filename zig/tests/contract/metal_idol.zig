//! Idol's Metal kernels under the contract, each at idol-sdk's fp16 points; controls drop or add one rounding point.
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const gemm = @import("metal_idol_gemm.zig");
const norms = @import("metal_idol_norms.zig");
const steps = @import("metal_idol_steps.zig");
const elems = @import("metal_idol_rows.zig");
const attn = @import("metal_idol_attention.zig");

const g = id.g;
const ks = id.ks;
const Control = id.Control;
const Lib = id.Lib;

/// The NAX fp32 GEMM (the tensor unit's fp32 path, not fp32-accurate on M5) in id_gemm32's place.
const nax_gemm32 =
    \\[[max_total_threads_per_threadgroup(128)]] [[kernel]] void id_gemm32(
    \\    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], device float* y [[buffer(2)]],
    \\    constant IdGemm32& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    \\    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    \\  const int row = int(tg.y) * 64 + 32 * int(sg / 2), col = int(tg.x) * 64 + 32 * int(sg % 2);
    \\  const int rows = min(32, int(p.M) - row), cols = min(32, int(p.N) - col);
    \\  const short2 home = tfp::frag_home(ushort(lane));
    \\  tfp::frag<float> acc[2][2];
    \\  tfp::gemm_block<float, false, true, false, 32>(acc, x + long(row) * p.K, w + long(col) * p.K, int(p.K), int(p.K),
    \\                                                 int(p.K), int(p.K) / 32, rows, cols, home);
    \\  if (rows > 0 && cols > 0) tfp::gemm_store<float, false>(acc, y + long(row) * p.ldy + col, int(p.ldy), rows, cols, home);
    \\}
;

const mma_line = "for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);";
const mma_half = "for (uint j = 0; j < 4; j++) { simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]); " ++
    "acc[i][j].thread_elements()[0] = id_h(acc[i][j].thread_elements()[0]); " ++
    "acc[i][j].thread_elements()[1] = id_h(acc[i][j].thread_elements()[1]); }";

const controls = [_]Control{
    .{ .kernel = "id_linear_gelu_01", .what = "drops fp16(x w + b) before the GELU", .edits = &.{.{ .from = "const float v = id_h(acc[i][j][e] + b);", .to = "const float v = (acc[i][j][e] + b);" }} },
    .{ .kernel = "id_linear_01", .what = "adds fp16(x w) before the bias", .edits = &.{.{ .from = "const float v = id_h(acc[i][j][e] + b);", .to = "const float v = id_h(id_h(acc[i][j][e]) + b);" }} },
    .{ .kernel = "id_norm modulated", .what = "drops fp16(y s1)", .edits = &.{.{ .from = "v = id_h(id_h(v * float(s1[i])) + float(sh[i]));", .to = "v = id_h((v * float(s1[i])) + float(sh[i]));" }} },
    .{ .kernel = "id_norm affine", .what = "adds fp16(y) before the affine", .edits = &.{.{ .from = "if (p.mode == 1) v = v * w[i] + b[i];", .to = "if (p.mode == 1) v = id_h(v) * w[i] + b[i];" }} },
    .{ .kernel = "id_norm32 modulated", .what = "adds fp16(y) before the modulation", .edits = &.{.{ .from = "yr[i] = p.mode == 2 ? v * (1.0f + w[i])", .to = "yr[i] = p.mode == 2 ? id_h(v) * (1.0f + w[i])" }} },
    .{ .kernel = "id_norm32 modulated", .what = "sums the simdgroups in fp16", .gate = .sums, .edits = &.{.{ .from = "  v = simd_sum(v);\n  if (lane == 0) part[sg] = v;", .to = "  v = float(simd_sum(half(v)));\n  if (lane == 0) part[sg] = v;" }} },
    .{ .kernel = "id_norm_split", .what = "rounds the row to fp16 before its split", .edits = &.{.{ .from = "const float v = (float(xr[i]) - mean) * inv * w[i] + b[i];", .to = "const float v = id_h((float(xr[i]) - mean) * inv * w[i] + b[i]);" }} },
    .{ .kernel = "id_repo", .what = "adds fp16(u)", .edits = &.{.{ .from = "u[i] = (a / (1.0f + metal::precise::exp(-a))) * (g[128 + i] * p.inv);", .to = "u[i] = id_h((a / (1.0f + metal::precise::exp(-a))) * (g[128 + i] * p.inv));" }} },
    .{ .kernel = "id_repo", .what = "sums d in fp16", .gate = .sums, .edits = &.{.{ .from = "s = fma(u[k], wf[t * 128 + k], s);", .to = "s = id_h(fma(u[k], wf[t * 128 + k], s));" }} },
    .{ .kernel = "id_qk", .what = "drops fp16(q0 c - q1 s)", .edits = &.{.{ .from = "const float rq0 = id_h(q0 * c - q1 * s), rq1", .to = "const float rq0 = (q0 * c - q1 * s), rq1" }} },
    .{ .kernel = "id_qk", .what = "sums q's squares in fp16", .gate = .sums, .edits = &.{.{ .from = "simd_sum(rq0 * rq0 + rq1 * rq1)", .to = "float(simd_sum(half(rq0 * rq0 + rq1 * rq1)))" }} },
    .{ .kernel = "id_gate_add", .what = "drops fp16(g y)", .edits = &.{.{ .from = "half(float(x[i]) + id_h(float(g[at.x]) * float(y[i])));", .to = "half(float(x[i]) + (float(g[at.x]) * float(y[i])));" }} },
    .{ .kernel = "id_embed", .what = "drops fp16(z w + b)", .edits = &.{.{ .from = "x[i] = half(float(half(s + b[c])) + float(pe[i]));", .to = "x[i] = half((s + b[c]) + float(pe[i]));" }} },
    .{ .kernel = "id_embed", .what = "sums z w in fp16", .gate = .sums, .edits = &.{.{ .from = "s = fma(z[r * 16 + k], w[c * 16 + k], s);", .to = "s = id_h(fma(z[r * 16 + k], w[c * 16 + k], s));" }} },
    .{ .kernel = "id_cond", .what = "adds fp16(a + ba)", .edits = &.{.{ .from = "y[i] = half((a[i] + ba[at.x]) + (c[i] + bc[at.x]));", .to = "y[i] = half(id_h(a[i] + ba[at.x]) + (c[i] + bc[at.x]));" }} },
    .{ .kernel = "id_bias32", .what = "adds fp16(b)", .edits = &.{.{ .from = "y[ulong(at.y) * width + at.x] += b[at.x];", .to = "y[ulong(at.y) * width + at.x] += id_h(b[at.x]);" }} },
    .{ .kernel = "id_gemm32", .what = "sums in fp16", .gate = .sums, .edits = &.{.{ .from = mma_line, .to = mma_half }} },
    .{ .kernel = "id_gemm32", .what = "takes the NAX fp32 path", .gate = .sums, .edits = &.{.{ .from = "void id_gemm32(", .to = "void id_gemm32_simd(" }}, .append = nax_gemm32 },
    .{ .kernel = "id_attention_00", .what = "keys past Lk unmasked", .edits = &.{.{ .from = "s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kIdMasked;", .to = "s[f][e] = s[f][e];" }} },
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
                const full = try w.s.label("metal idol {s}{s}", .{ kernel, w.suffix });
                if (w.only) |needle| if (std.mem.indexOf(u8, full, needle) == null) return null;
                return full;
            }
        };
        const w = W{ .s = s, .ctl = ctl, .suffix = suffix, .only = only };
        for ([_]gemm.Kind{ .linear, .gelu, .split }) |kind| for (0..4) |f| {
            const form: u2 = @intCast(f);
            if (kind == .split and form & 1 == 0) continue;
            const fun = gemm.pipeline(kind, form);
            const name = (try w.name(fun)) orelse continue;
            const t = try s.make(gemm.Gemm16, .{ .a = s.a, .rig = rig, .lib = lib, .kind = kind, .form = form, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        };
        if (try w.name("id_gemm32")) |name| {
            const t = try s.make(gemm.Gemm32, .{ .a = s.a, .rig = rig, .lib = lib, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        const norm_kinds = [_]struct { kind: norms.Kind, mode: ks.Norm, fun: []const u8 }{
            .{ .kind = .norm, .mode = .plain, .fun = "id_norm plain" },
            .{ .kind = .norm, .mode = .affine, .fun = "id_norm affine" },
            .{ .kind = .norm, .mode = .modulated, .fun = "id_norm modulated" },
            .{ .kind = .norm32, .mode = .affine, .fun = "id_norm32 affine" },
            .{ .kind = .norm32, .mode = .modulated, .fun = "id_norm32 modulated" },
            .{ .kind = .split, .mode = .affine, .fun = "id_norm_split" },
        };
        for (norm_kinds) |nk| {
            const name = (try w.name(nk.fun)) orelse continue;
            const t = try s.make(norms.Norm, .{ .a = s.a, .rig = rig, .lib = lib, .kind = nk.kind, .mode = nk.mode, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("id_repo")) |name| {
            const t = try s.make(steps.Repo, .{ .a = s.a, .rig = rig, .lib = lib, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        if (try w.name("id_qk")) |name| {
            const t = try s.make(steps.Qk, .{ .a = s.a, .rig = rig, .lib = lib, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        inline for (.{ .{ elems.GateAdd, "id_gate_add" }, .{ elems.Embed, "id_embed" }, .{ elems.Cond, "id_cond" }, .{ elems.Bias32, "id_bias32" } }) |e| {
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
            std.debug.print("UNSUPPORTED metal idol: the family runs on tensor units (M5) alone; its kernels unchecked here\n", .{});
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
