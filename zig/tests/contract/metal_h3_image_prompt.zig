//! MiniMax-H3-Image's prompt kernels (h3_prompt.metal) under the contract: each against its own formula in float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const common = @import("metal_h3_image_common.zig");
const case = @import("metal_h3_image_prompt_case.zig");
const g = contract.gates;
const c = contract.spec;
const Case = case.Case;
const Form = case.Form;

/// A short prompt's rows (a 64-row tile and 13 more) and a long presentation's (sixteen tiles and 7 more).
const short = 77;
const long = 1031;

/// The text model's projections (q, k and v, o, gate and up, down) and the refiner's (qkv, out, fc1, fc2) as N, K.
const projections = [_][2]usize{ .{ 1024, 5120 }, .{ 8192, 5120 }, .{ 5120, 8192 }, .{ 25600, 5120 }, .{ 5120, 25600 }, .{ 21504, 5376 }, .{ 5376, 7168 }, .{ 28672, 5376 }, .{ 5376, 14336 } };

const gemm_served = blk: {
    var out: []const c.Shape = &.{};
    for (projections) |p| out = out ++ &[_]c.Shape{.{ .rows = short, .n = p[0], .k = p[1] }};
    break :blk out ++ &[_]c.Shape{ .{ .rows = long, .n = 1024, .k = 5120 }, .{ .rows = long, .n = 5376, .k = 7168 } };
};

fn rowsAt(comptime n: usize, comptime k: usize) []const c.Shape {
    return &.{ .{ .rows = short, .n = n, .k = k }, .{ .rows = long, .n = n, .k = k } };
}

fn headsAt(comptime heads: usize, comptime rows: []const usize) []const c.Shape {
    var out: []const c.Shape = &.{};
    for (rows) |t| out = out ++ &[_]c.Shape{.{ .rows = heads, .n = 128, .k = t }};
    return out;
}

/// The released checkpoint's launches in prompt.zig (text 5120 wide, 64 over 8 heads; refiner 5376, 56 heads).
const forms = [_]Form{
    .{ .kind = .gemm, .served = gemm_served },
    .{ .kind = .gemm, .label = "bias", .bias = true, .served = rowsAt(5376, 5120) },
    .{ .kind = .norm, .label = "qwen", .mode = .qwen, .eps = 1e-6, .served = rowsAt(5120, 5120) },
    .{ .kind = .norm, .label = "h3", .mode = .h3, .eps = 1e-5, .served = rowsAt(5376, 5376) },
    .{ .kind = .heads, .label = "qwen rotary", .mode = .qwen, .eps = 1e-6, .served = rowsAt(64, 128) },
    .{ .kind = .heads, .label = "h3 fused", .mode = .h3, .eps = 1e-5, .fused = true, .served = rowsAt(56, 128) },
    .{ .kind = .attention, .label = "causal", .causal = true, .served = headsAt(64, &.{ short, long }) },
    .{ .kind = .attention, .label = "fused", .fused = true, .served = headsAt(56, &.{ short, long }) },
    .{ .kind = .attention, .label = "causal last key", .causal = true, .last_key = true, .served = headsAt(64, &.{short}) },
    .{ .kind = .attention, .label = "fused last key", .fused = true, .last_key = true, .served = headsAt(56, &.{short}) },
    .{ .kind = .glu, .label = "qwen", .mode = .qwen, .served = rowsAt(25600, 1) },
    .{ .kind = .glu, .label = "h3 fused", .mode = .h3, .fused = true, .served = rowsAt(14336, 1) },
    .{ .kind = .add, .served = rowsAt(5120, 1) ++ rowsAt(5376, 1) },
    .{ .kind = .scatter, .served = &.{ .{ .rows = 1024, .n = 5120, .k = 1 }, .{ .rows = 2560, .n = 5120, .k = 1 } } },
};

/// A planted control: the registration it breaks (its label after the function), the one source edit, and its gate.
const Control = struct { entry: []const u8, gate: g.Gate, what: []const u8, old: []const u8, new: []const u8 };

const controls = [_]Control{
    .{ .entry = "hp_gemm", .gate = .shapes, .what = "drops the last K block", .old = "k0 < K; k0 += BK", .new = "k0 + BK < K; k0 += BK" },
    .{ .entry = "hp_gemm bias", .gate = .shapes, .what = "swaps a column pair's biases", .old = "float2(float(bias[c]), float(bias[c + 1]))", .new = "float2(float(bias[c + 1]), float(bias[c]))" },
    .{ .entry = "hp_norm qwen", .gate = .shapes, .what = "rounds as H3's blocks", .old = "return mode == 0 ?", .new = "return mode != 0 ?" },
    .{ .entry = "hp_norm h3", .gate = .shapes, .what = "rounds as the Qwen model", .old = "return mode == 0 ?", .new = "return mode != 0 ?" },
    .{ .entry = "hp_norm qwen", .gate = .sums, .what = "sums its squares in half", .old = "float s = 0.0f;", .new = "half s = 0.0h;" },
    .{ .entry = "hp_heads qwen rotary", .gate = .shapes, .what = "turns the wrong way", .old = "lane < 16 ? -1.0f : 1.0f", .new = "lane < 16 ? 1.0f : -1.0f" },
    .{ .entry = "hp_heads h3 fused", .gate = .shapes, .what = "rounds as the Qwen model", .old = "return mode == 0 ?", .new = "return mode != 0 ?" },
    .{ .entry = "hp_attention causal", .gate = .shapes, .what = "sees the next key", .old = "? row + 1 : p.rows;", .new = "? min(row + 2, p.rows) : p.rows;" },
    .{ .entry = "hp_attention fused", .gate = .shapes, .what = "skips the last key", .old = "? row + 1 : p.rows;", .new = "? row + 1 : p.rows - 1;" },
    .{ .entry = "hp_attention causal last key", .gate = .nonfinite, .what = "adds its padded keys' values at weight 0", .old = "if (j0 + u < end) acc += e", .new = "acc += e" },
    .{ .entry = "hp_attention causal", .gate = .nonfinite, .what = "weighs a NaN score 0", .old = "const float e = metal::precise::exp(s[u] - top);", .new = "const float e = isnan(s[u]) ? 0.0f : metal::precise::exp(s[u] - top);" },
    .{ .entry = "hp_glu qwen", .gate = .shapes, .what = "swaps gate and value", .old = "g = float(G[ulong(at.y) * p.g_row + at.x]), v = float(V[ulong(at.y) * p.v_row + at.x])", .new = "g = float(V[ulong(at.y) * p.v_row + at.x]), v = float(G[ulong(at.y) * p.g_row + at.x])" },
    .{ .entry = "hp_glu h3 fused", .gate = .shapes, .what = "rounds as the Qwen model", .old = "const float s = p.mode == 0 ?", .new = "const float s = p.mode != 0 ?" },
    .{ .entry = "hp_add", .gate = .shapes, .what = "skips the last value", .old = "if (i < n)", .new = "if (i + 1 < n)" },
    .{ .entry = "hp_scatter_add", .gate = .shapes, .what = "adds into row i, not idx[i]", .old = "ulong(idx[at.y]) * width", .new = "ulong(at.y) * width" },
};

fn entry(comptime f: Form) []const u8 {
    return case.function(f.kind) ++ (if (f.label.len > 0) " " ++ f.label else "");
}

fn kChain(s: c.Shape) usize {
    return s.k + 2;
}
/// hp_norm: a thread's chain over every 256th value, simd_sum's tree, then the 8 simdgroups' partials in order.
fn normChain(s: c.Shape) usize {
    return (s.k + 255) / 256 + 6 + 8;
}
/// hp_heads: a lane's 4 squares, then simd_sum's tree.
fn headChain(_: c.Shape) usize {
    return 4 + 6;
}
/// hp_attention: a score's lane dot and tree, then the keys' sequential sum and its rescales.
fn attnChain(s: c.Shape) usize {
    return 4 + 6 + s.k + s.k / 4 + 3;
}
fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(comptime f: Form, name: []const u8) c.Spec {
    return switch (f.kind) {
        .gemm => .{ .name = name, .accum = .f32, .depth = kChain, .ops = if (f.bias) 1 else 0, .out = .bf16, .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 32, .step = 32, .min = 32 } }, .served = f.served },
        .norm => .{ .name = name, .accum = .f32, .depth = normChain, .ops = 8, .mids = if (f.mode == .qwen) 1 else 0, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = f.served },
        .heads => .{ .name = name, .accum = .f32, .depth = headChain, .ops = 8, .mids = 3, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{ .size = 8 }, .{} }, .served = f.served },
        .attention => .{ .name = name, .accum = .f32, .depth = attnChain, .ops = 8, .out = .bf16, .tiles = .{ .{ .size = 8 }, .{}, .{ .size = 4 } }, .served = f.served, .known = f.known },
        .glu => .{ .name = name, .accum = .none, .depth = none, .ops = 8, .mids = 2, .out = .bf16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = f.served },
        .add, .scatter => .{ .name = name, .accum = .none, .depth = none, .ops = 1, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = f.served },
    };
}

pub const Suite = struct {
    a: std.mem.Allocator,
    cases: std.ArrayList(*Case) = .empty,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,

    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, comptime f: Form, comptime name: []const u8, gate: ?g.Gate, out: *std.ArrayList(g.Kernel)) !void {
        const p = try mtl.Pipeline.init(device, lib, case.function(f.kind), false);
        {
            errdefer p.deinit();
            try s.pipes.append(s.a, p);
        }
        const t = try s.a.create(Case);
        t.* = Case.init(s.a, device, queue, p, f, spec(f, name), gate);
        {
            errdefer {
                t.deinit();
                s.a.destroy(t);
            }
            try s.cases.append(s.a, t);
        }
        try out.append(s.a, t.kernel());
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const lib = try common.library(device, .prompt, common.source(.prompt));
        defer lib.deinit();
        inline for (forms) |f| {
            const name = comptime common.name(.prompt, entry(f));
            if (common.keep(only, name)) try s.add(device, queue, lib, f, name, null, kernels);
        }
        inline for (controls) |x| inline for (forms) |f| if (comptime std.mem.eql(u8, entry(f), x.entry)) {
            const name = comptime common.name(.prompt, entry(f)) ++ " control: " ++ x.what;
            if (common.keep(only, name)) {
                const text = try common.edited(a, .prompt, x.old, x.new);
                defer a.free(text);
                const edited = try common.library(device, .prompt, text);
                defer edited.deinit();
                try s.add(device, queue, edited, f, name, x.gate, kernels);
            }
        };
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.cases.items) |t| {
            t.deinit();
            s.a.destroy(t);
        }
        s.cases.deinit(s.a);
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(s.a);
        s.a.destroy(s);
    }
};
