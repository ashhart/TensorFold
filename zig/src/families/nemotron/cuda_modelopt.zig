//! NVIDIA ModelOpt checkpoints: each projection FP8, NVFP4 or bf16 as its tensors say; the bf16 MTP head made 4-bit.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const Config = @import("config.zig").Config;
const draft_ids = @import("draft_ids.zig");
const host4 = core.affine4_host;

const Loader = W.Loader;
const Tensor = core.checkpoint.Tensor;
const Nvfp4Experts = cuda.experts.Nvfp4Experts;

const Kind = enum { nvfp4, fp8, bf16, q8_0, iq2xxs };

/// Bytes and weights of one ggml block: Q8_0 34 per 32, IQ2_XXS 66 per 256.
fn ggmlBlock(k: Kind) [2]usize {
    return switch (k) {
        .q8_0 => .{ 34, 32 },
        .iq2xxs => .{ 66, 256 },
        else => unreachable,
    };
}

fn kindOf(t: Tensor) !Kind {
    // ggml blocks (Bolt): [rows, blocks, block bytes]; NVFP4 packs pairs as [rows, k / 2]
    if (t.dtype == .u8 and t.rank == 3) return switch (t.dim(2)) {
        34 => .q8_0,
        66 => .iq2xxs,
        else => error.UnsupportedQuantization,
    };
    return switch (t.dtype) {
        .u8 => .nvfp4,
        .f8_e4m3 => .fp8,
        .bf16 => .bf16,
        else => error.UnsupportedQuantization,
    };
}

/// A part's weight, its e4m3 block scales (NVFP4) and its one fp32 scale (NVFP4's weight_scale_2, FP8's weight_scale).
const Part = struct { w: Tensor, blocks: ?Tensor = null, scale: f32 = 1.0, kind: Kind, n: usize, k: usize };

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// An fp32 tensor of one element, read through the map (a few bytes; the loader's O_DIRECT reads are for bulk).
fn scalar(ck: *core.Checkpoint, n: []const u8) !f32 {
    const t = try ck.get(n);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// `{pre}.weight` and its scales by the weight's dtype; static activation and FP8 cache scales are read and set aside.
fn part(ck: *core.Checkpoint, pre: []const u8) !Part {
    var buf: [192]u8 = undefined;
    const w = try ck.get(try name(&buf, "{s}.weight", .{pre}));
    if (w.rank != 2 and w.rank != 3) return error.UnexpectedTensor;
    // we keep bf16 activations and a bf16 KV cache, more precise than the W8A8 and FP8-cache math these scales are for
    for ([_][]const u8{ "input_scale", "k_scale", "v_scale" }) |extra| {
        const n = try name(&buf, "{s}.{s}", .{ pre, extra });
        if (ck.has(n)) _ = try ck.get(n);
    }
    const kind = try kindOf(w);
    if ((w.rank == 3) != (kind == .q8_0 or kind == .iq2xxs)) return error.UnexpectedTensor;
    const k = switch (kind) {
        .nvfp4 => w.dim(1) * 2,
        .q8_0, .iq2xxs => w.dim(1) * ggmlBlock(kind)[1],
        else => w.dim(1),
    };
    var p: Part = .{ .w = w, .kind = kind, .n = w.dim(0), .k = k };
    switch (kind) {
        .bf16, .q8_0, .iq2xxs => {},
        .fp8 => p.scale = try scalar(ck, try name(&buf, "{s}.weight_scale", .{pre})),
        .nvfp4 => {
            p.blocks = try ck.expect(try name(&buf, "{s}.weight_scale", .{pre}), .f8_e4m3, &.{ p.n, p.k / 16 });
            p.scale = try scalar(ck, try name(&buf, "{s}.weight_scale_2", .{pre}));
        },
    }
    return p;
}

fn kernels(L: *Loader) !*const kern.Modelopt {
    return if (L.ops.k.modelopt) |*m| m else error.ModeloptNotLoaded;
}

/// Row-stacked parts (one projection, or q, k, v) of one kind and one scale as a lane weight, packed on the GPU.
pub fn dense(L: *Loader, ck: *core.Checkpoint, label: []const u8, prefixes: []const []const u8) !kern.QLinear {
    const m = try kernels(L);
    var parts: [3]Part = undefined;
    var n: usize = 0;
    for (prefixes, 0..) |pre, i| {
        parts[i] = try part(ck, pre);
        n += parts[i].n;
    }
    const first = parts[0];
    for (parts[0..prefixes.len]) |p| if (p.kind != first.kind or p.scale != first.scale or p.k != first.k) return error.MixedQuantization;
    const k = first.k;
    if (k % 64 != 0) return error.UnexpectedTensor;
    var buf: [128]u8 = undefined;
    const s = L.ops.s;
    switch (first.kind) {
        .q8_0, .iq2xxs => return ggmlDense(L, label, parts[0..prefixes.len], n, k),
        .bf16, .fp8 => {
            const size: usize = if (first.kind == .bf16) 2 else 1;
            const base = try L.tmp(n * k * size);
            var at: usize = 0;
            for (parts[0..prefixes.len]) |p| {
                try L.src.upload(base + at, p.w.bytes);
                at += p.w.bytes.len;
            }
            const out = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k * size);
            try L.src.flush();
            if (first.kind == .bf16) {
                try m.pack.bf16Lane(s, base, out, n, k);
                return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
            }
            try m.pack.fp8Lane(s, base, out, n, k);
            return .{ .fp8 = cuda.modelopt.fp8Weight(out, first.scale, n, k) };
        },
        .nvfp4 => {
            const codes = n * k / 2;
            const base = try L.tmp(codes + n * k / 16);
            var at = [2]usize{ 0, codes };
            for (parts[0..prefixes.len]) |p| for ([2]Tensor{ p.w, p.blocks.? }, &at) |t, *o| {
                try L.src.upload(base + o.*, t.bytes);
                o.* += t.bytes.len;
            };
            const words = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k / 2);
            const tiles = try L.alloc(try name(&buf, "{s}.scales", .{label}), cuda.modelopt.padded(n) * k / 16);
            try L.src.flush();
            try m.pack.fp4Lane(s, base, base + codes, words, tiles, n, k);
            return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, first.scale, n, k) };
        },
    }
}

/// Row-stacked ggml parts (q, k, v: rows after rows) uploaded as stored, then tiled on the GPU (a byte permutation).
fn ggmlDense(L: *Loader, label: []const u8, parts: []const Part, n: usize, k: usize) !kern.QLinear {
    const g = try ggmlKernels(L);
    const kind = parts[0].kind;
    const base = try L.tmp(n * k / ggmlBlock(kind)[1] * ggmlBlock(kind)[0]);
    var at: usize = 0;
    for (parts) |p| {
        try L.src.upload(base + at, p.w.bytes);
        at += p.w.bytes.len;
    }
    var buf: [128]u8 = undefined;
    const gk: cuda.ggml.Kind = if (kind == .q8_0) .q8_0 else .iq2xxs;
    const out = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.ggml.bytes(gk, n, k));
    try L.src.flush();
    try g.pack(L.ops.s, gk, base, out, n, k, 1);
    return if (kind == .q8_0) .{ .q8_0 = cuda.ggml.q8Weight(out, n, k) } else .{ .iq2xxs = cuda.ggml.iq2Weight(out, n, k) };
}

fn ggmlKernels(L: *Loader) !*const cuda.ggml.Kernels {
    return if (L.ops.k.ggml) |*g| g else error.GgmlNotLoaded;
}

/// IQ2_XXS grouped experts from `count` tensors named by `names`, [n, k] each, uploaded back to back and tiled.
fn iq2Experts(L: *Loader, ck: *core.Checkpoint, label: []const u8, count: usize, n: usize, k: usize, names: anytype) !u64 {
    const g = try ggmlKernels(L);
    const one = n * k / 256 * 66;
    const base = try L.tmp(count * one);
    var b: [192]u8 = undefined;
    for (0..count) |i| {
        const p = try part(ck, try names.at(&b, i));
        if (p.kind != .iq2xxs or p.n != n or p.k != k) {
            std.log.err("{s}: {t} [{d}, {d}], expected IQ2_XXS [{d}, {d}]", .{ try names.at(&b, i), p.kind, p.n, p.k, n, k });
            return error.UnexpectedTensor;
        }
        try L.src.upload(base + i * one, p.w.bytes);
    }
    const out = try L.alloc(label, cuda.ggml.bytes(.iq2xxs, n, k) * count);
    try L.src.flush();
    try g.pack(L.ops.s, .iq2xxs, base, out, n, k, count);
    return out;
}

/// A LatentMoE layer (Bolt): router and bias, fc1/fc2_latent_proj, latent IQ2_XXS experts, full-width shared expert.
pub fn latent(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config, layer: usize) !W.MoE {
    const lm = c.layerMoe(layer);
    const E = lm.experts;
    const Lw = c.latent;
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = try routerOf(L, try name(&a, "{s}.router", .{label}), try ck.get(try name(&b, "{s}gate.weight", .{pre})), E, c.hidden);
    const bias = try ck.expect(try name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{E});
    var m: W.MoE = .{ .router = r, .bias = try L.raw(try name(&a, "{s}.bias", .{label}), bias), .experts = undefined };
    var ex: kern.Experts = .{ .format = .iq2xxs, .up = 0, .down = 0, .width = lm.width, .dims = Lw, .experts = E, .relu2 = true };
    const Names = struct {
        pre: []const u8,
        leaf: []const u8,
        fn at(x: @This(), buf: []u8, i: usize) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}experts.{d}.{s}", .{ x.pre, i, x.leaf });
        }
    };
    ex.up = try iq2Experts(L, ck, try name(&a, "{s}.experts.up", .{label}), E, lm.width, Lw, Names{ .pre = pre, .leaf = "up_proj" });
    ex.down = try iq2Experts(L, ck, try name(&a, "{s}.experts.down", .{label}), E, Lw, lm.width, Names{ .pre = pre, .leaf = "down_proj" });
    m.experts = ex;
    const Shared = struct {
        pre: []const u8,
        leaf: []const u8,
        fn at(x: @This(), buf: []u8, _: usize) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}shared_experts.{s}", .{ x.pre, x.leaf });
        }
    };
    var sh: kern.Experts = .{ .format = .iq2xxs, .up = 0, .down = 0, .width = c.shared_width, .dims = c.hidden, .experts = 1, .relu2 = true };
    sh.up = try iq2Experts(L, ck, try name(&a, "{s}.shared.up", .{label}), 1, c.shared_width, c.hidden, Shared{ .pre = pre, .leaf = "up_proj" });
    sh.down = try iq2Experts(L, ck, try name(&a, "{s}.shared.down", .{label}), 1, c.hidden, c.shared_width, Shared{ .pre = pre, .leaf = "down_proj" });
    var p1: [192]u8 = undefined;
    var p2: [192]u8 = undefined;
    m.latent = .{
        .fc1 = try dense(L, ck, try name(&a, "{s}.fc1_latent", .{label}), &.{try name(&p1, "{s}fc1_latent_proj", .{pre})}),
        .fc2 = try dense(L, ck, try name(&b, "{s}.fc2_latent", .{label}), &.{try name(&p2, "{s}fc2_latent_proj", .{pre})}),
        .shared = sh,
    };
    return m;
}

/// The router as the kernels read it: bf16 [E, D]; an fp32 router is taken only when every value is a bf16.
fn router(L: *Loader, label: []const u8, t: Tensor, c: Config) !u64 {
    return routerOf(L, label, t, c.experts, c.hidden);
}

fn routerOf(L: *Loader, label: []const u8, t: Tensor, experts_: usize, hidden: usize) !u64 {
    if (t.rank != 2 or t.dim(0) != experts_ or t.dim(1) != hidden) return error.UnexpectedTensor;
    if (t.dtype == .bf16) return L.raw(label, t);
    if (t.dtype != .f32) return error.UnexpectedTensor;
    const raw = try L.gpa.alloc(u8, t.bytes.len);
    defer L.gpa.free(raw);
    try L.src.read(raw, t.bytes);
    const out = try L.keep(u16, t.bytes.len / 4);
    for (out, 0..) |*o, i| {
        const bits = std.mem.readInt(u32, raw[4 * i ..][0..4], .little);
        if (bits & 0xFFFF != 0) return error.RouterNotBf16; // our router kernel reads bf16 weights
        o.* = @intCast(bits >> 16);
    }
    const ptr = try L.alloc(label, out.len * 2);
    try L.src.flush();
    try L.ops.upload(ptr, std.mem.sliceAsBytes(out));
    return ptr;
}

/// Per-expert fp32 global scales on the GPU.
fn scales(L: *Loader, label: []const u8, values: []const f32) !u64 {
    const host = try L.keep(f32, values.len);
    @memcpy(host, values);
    const ptr = try L.alloc(label, values.len * 4);
    try L.src.flush();
    try L.ops.upload(ptr, std.mem.sliceAsBytes(host));
    return ptr;
}

/// A MoE layer: the router, its bias, and NVFP4 experts with the shared expert's halves as experts E and E + 1.
pub fn moe(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = try router(L, try name(&a, "{s}.router", .{label}), try ck.get(try name(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    return .{ .router = r, .bias = try L.raw(try name(&a, "{s}.bias", .{label}), bias), .experts = try experts(L, ck, label, pre, c) };
}

fn experts(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !kern.Experts {
    const m = try kernels(L);
    const E = c.experts;
    const e = E + 2;
    const width = c.expert_width;
    const D = c.hidden;
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = width, .dims = D, .experts = e, .relu2 = true };
    const gs = try L.gpa.alloc(f32, 2 * e);
    defer L.gpa.free(gs);
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) width else D;
        const k = if (up) D else width;
        const codes = n * k / 2;
        const blocks = n * k / 16;
        const base = try L.tmp(e * (codes + blocks));
        for (0..E) |i| {
            const p = try part(ck, try name(&b, "{s}experts.{d}.{s}", .{ pre, i, if (up) "up_proj" else "down_proj" }));
            if (p.kind != .nvfp4 or p.n != n or p.k != k) return error.UnexpectedTensor;
            try L.src.upload(base + i * codes, p.w.bytes);
            try L.src.upload(base + e * codes + i * blocks, p.blocks.?.bytes);
            gs[@intFromBool(!up) * e + i] = p.scale;
        }
        const s = try part(ck, try name(&b, "{s}shared_experts.{s}", .{ pre, if (up) "up_proj" else "down_proj" }));
        if (s.kind != .nvfp4 or s.n != (if (up) 2 * n else n) or s.k != (if (up) k else 2 * k)) return error.SharedExpertWidth;
        try L.src.upload(base + E * codes, s.w.bytes); // up: rows 0..2W are experts E and E + 1; down: [D, 2W] apart
        try L.src.upload(base + e * codes + E * blocks, s.blocks.?.bytes);
        gs[@intFromBool(!up) * e + E] = s.scale;
        gs[@intFromBool(!up) * e + E + 1] = s.scale;
        const out = try L.alloc(try name(&a, "{s}.experts.{s}", .{ label, if (up) "up" else "down" }), Nvfp4Experts.bytes(e, n, k, 1));
        try L.src.flush();
        const st = L.ops.s;
        if (up) {
            try m.pack.fp4Experts(st, base, base + e * codes, out, e, n, k, 0, k, 1, 0);
            layer.up = out;
        } else {
            try m.pack.fp4Experts(st, base, base + e * codes, out, E, n, k, 0, k, 1, 0);
            const one = Nvfp4Experts.bytes(1, n, k, 1);
            for (0..2) |h| try m.pack.fp4Experts(st, base + E * codes, base + e * codes + E * blocks, out + (E + h) * one, 1, n, 2 * k, h * k, k, 1, 0);
            layer.down = out;
        }
    }
    layer.up_scale = try scales(L, try name(&a, "{s}.experts.up_scale", .{label}), gs[0..e]);
    layer.down_scale = try scales(L, try name(&a, "{s}.experts.down_scale", .{label}), gs[e..]);
    return layer;
}

/// The bf16 token table as stored.
pub fn embed(L: *Loader, ck: *core.Checkpoint, c: Config) !W.Embed {
    const t = try ck.expect("backbone.embeddings.weight", .bf16, &.{ c.vocab, c.hidden });
    return .{ .w = try L.raw("embed.weight", t), .n = c.vocab, .k = c.hidden, .bf16 = true };
}

/// A bf16 matrix made MLX affine-4 in groups of 64 on the host: the words, scales and biases the 4-bit loader reads.
fn quantized(L: *Loader, t: Tensor, words: []u32, sc: []u16, bi: []u16) !void {
    if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) % host4.group != 0) return error.UnexpectedTensor;
    const count = t.dim(0) * t.dim(1);
    const raw = try L.gpa.alloc(u8, count * 2);
    defer L.gpa.free(raw);
    try L.src.read(raw, t.bytes);
    const values = try L.gpa.alloc(f32, count);
    defer L.gpa.free(values);
    for (values, 0..) |*v, i| v.* = host4.f32of(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
    host4.quantize(values, words, sc, bi);
}

/// `count` bf16 matrices [n, k] (`fmt` names each by pre and index) made 4-bit, stacked as the MLX loader reads them.
fn stack4(L: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, pre: []const u8, count: usize, n: usize, k: usize) ![3]Tensor {
    const per = n * k;
    const words = try L.keep(u32, count * per / 8);
    const sc = try L.keep(u16, count * per / host4.group);
    const bi = try L.keep(u16, count * per / host4.group);
    var buf: [192]u8 = undefined;
    for (0..count) |i| {
        const t = try ck.get(try name(&buf, fmt, .{ pre, i }));
        if (t.dim(0) != n or t.dim(1) != k) return error.UnexpectedTensor;
        try quantized(L, t, words[i * per / 8 ..][0 .. per / 8], sc[i * per / host4.group ..][0 .. per / host4.group], bi[i * per / host4.group ..][0 .. per / host4.group]);
    }
    return .{ W.made(.u32, count * n, k / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, count * n, k / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, count * n, k / host4.group, std.mem.sliceAsBytes(bi)) };
}

/// One bf16 matrix `{pre}{mid}.weight` made 4-bit.
fn one4(L: *Loader, ck: *core.Checkpoint, pre: []const u8, mid: []const u8) ![3]Tensor {
    var buf: [192]u8 = undefined;
    const t = try ck.get(try name(&buf, "{s}{s}.weight", .{ pre, mid }));
    if (t.rank != 2) return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const words = try L.keep(u32, n * k / 8);
    const sc = try L.keep(u16, n * k / host4.group);
    const bi = try L.keep(u16, n * k / host4.group);
    try quantized(L, t, words, sc, bi);
    return .{ W.made(.u32, n, k / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(bi)) };
}

/// The MTP head (bf16 in these checkpoints) made affine-4 as the MLX one is; the draft head is lm_head's NVFP4 rows.
pub fn mtp(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const p0 = "mtp.layers.0.";
    const p1 = "mtp.layers.1.mixer.";
    const att = p0 ++ "mixer.";
    const qkv = [3][3]Tensor{ try one4(L, ck, att, "q_proj"), try one4(L, ck, att, "k_proj"), try one4(L, ck, att, "v_proj") };
    var moe_: W.MoE = .{ .router = try router(L, "mtp.moe.router", try ck.get(p1 ++ "gate.weight"), c), .bias = try L.raw("mtp.moe.bias", try ck.expect(p1 ++ "gate.e_score_correction_bias", .f32, &.{c.experts})), .experts = undefined };
    const e = c.experts + 2;
    const s_up = try one4(L, ck, p1, "shared_experts.up_proj");
    const s_down = try one4(L, ck, p1, "shared_experts.down_proj");
    moe_.experts = .{
        .format = .affine4,
        .up = try L.packExperts("mtp.moe.experts.up", try stack4(L, ck, "{s}experts.{d}.up_proj.weight", p1, c.experts, c.expert_width, c.hidden), s_up, false, e, c.expert_width, c.hidden),
        .down = try L.packExperts("mtp.moe.experts.down", try stack4(L, ck, "{s}experts.{d}.down_proj.weight", p1, c.experts, c.hidden, c.expert_width), s_down, true, e, c.hidden, c.expert_width),
        .experts = e,
        .width = c.expert_width,
        .dims = c.hidden,
    };
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.get(p0 ++ "enorm.weight")),
        .hnorm = try L.raw("mtp.hnorm", try ck.get(p0 ++ "hnorm.weight")),
        .eh_proj = .{ .affine4 = try L.tile("mtp.eh_proj", &.{try one4(L, ck, p0, "eh_proj")}) },
        .attn_norm = try L.raw("mtp.attn_norm", try ck.get(p0 ++ "norm.weight")),
        .attn = .{ .qkv = .{ .affine4 = try L.tile("mtp.attn.qkv", &qkv) }, .o = .{ .affine4 = try L.tile("mtp.attn.o", &.{try one4(L, ck, att, "o_proj")}) } },
        .moe_norm = try L.raw("mtp.moe_norm", try ck.get("mtp.layers.1.norm.weight")),
        .moe = moe_,
        .final_norm = try L.raw("mtp.final_norm", try ck.get("mtp.layers.1.final_layernorm.weight")),
    };
    try draftHead(L, ck, c);
}

/// head_rows: lm_head's NVFP4 rows for the draft ids, packed as a head of their own (one global scale).
fn draftHead(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const m = try kernels(L);
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    const p = try part(ck, "lm_head");
    if (p.kind != .nvfp4 or p.n != c.vocab) return error.UnexpectedTensor;
    const rows = [2]usize{ p.k / 2, p.k / 16 };
    const host = try L.staging(ids.len * (rows[0] + rows[1]));
    for ([2]Tensor{ p.w, p.blocks.? }, rows, [2]usize{ 0, ids.len * rows[0] }) |t, row, at| {
        const whole = try L.gpa.alloc(u8, t.bytes.len);
        defer L.gpa.free(whole);
        try L.src.read(whole, t.bytes);
        for (ids, 0..) |id, r| @memcpy(host[at + r * row ..][0..row], whole[id * row ..][0..row]);
    }
    const base = try L.tmp(host.len);
    try L.ops.upload(base, host);
    const n = ids.len;
    const words = try L.alloc("draft_head.weight", cuda.modelopt.padded(n) * p.k / 2);
    const tiles = try L.alloc("draft_head.scales", cuda.modelopt.padded(n) * p.k / 16);
    try m.pack.fp4Lane(L.ops.s, base, base + n * rows[0], words, tiles, n, p.k);
    L.w.draft_head = .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, p.scale, n, p.k) };
    try W.draftIds(L, ids);
}
