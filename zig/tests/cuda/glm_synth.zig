//! `glm-synth OUT_DIR [fp8pb|local]`: a 2-layer + MTP GLM-5.3-Flash checkpoint, dc.json; run TF_GLM_LAYERS=2.
const std = @import("std");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");

const st = core.safetensors;
const New = core.shard_edit.New;

const layers = 2;
const experts = 16;
const vocab = 8192;
const hidden = 4096;
/// The ModelOpt block nvidia/GLM-5.3-Flash-NVFP4 writes, which the CUDA family admits (modelopt.admitRecipes).
const modelopt_block = ",\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"NVFP4\", " ++
    "\"config_groups\": {\"group_0\": {\"weights\": {\"num_bits\": 4, \"type\": \"float\", \"group_size\": 16}}}}";
/// fp8pb: FP4-Local's MIXED_PRECISION form, layer 1's MLA and shared expert in block FP8, the NVFP4 ones as before.
const mixed_head = ",\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\", \"quantized_layers\": {";
const layer1 = nv4("layers.1.mlp.experts") ++ "," ++ pb("layers.1.mlp.shared_experts") ++ "," ++ pb("layers.1.self_attn.q_a_proj") ++ "," ++
    pb("layers.1.self_attn.kv_a_proj_with_mqa") ++ "," ++ pb("layers.1.self_attn.q_b_proj") ++ "," ++ pb("layers.1.self_attn.o_proj");
const mixed_block = mixed_head ++ nv4("layers.0.mlp") ++ "," ++ layer1 ++ "}}";
/// local: as fp8pb; layer 0's KDA q k v o, dense MLP, lm_head and the MTP layer (mtp_block) as FP4-Local stores them.
const local_block = mixed_head ++ mx8("lm_head") ++ "," ++ mx8(P ++ "layers.0.self_attn.q_proj") ++ "," ++ mx8(P ++ "layers.0.self_attn.k_proj") ++ "," ++
    mx8(P ++ "layers.0.self_attn.v_proj") ++ "," ++ mx8(P ++ "layers.0.self_attn.o_proj") ++ "," ++ mx8(P ++ "layers.0.mlp.gate_proj") ++ "," ++
    mx8(P ++ "layers.0.mlp.up_proj") ++ "," ++ mx8(P ++ "layers.0.mlp.down_proj") ++ "," ++ layer1 ++ mtp_block ++ "}}";
const mtp_at = std.fmt.comptimePrint("layers.{d}.", .{layers});
/// The MTP layer's recipes as FP4-Local's: attention, kv_b, indexer, shared in MXFP8 (group_size), experts W4A16.
const mtp_block = mtp: {
    var s: []const u8 = ",\"" ++ P ++ mtp_at ++ "mlp.experts\": {\"group_size\": 16, \"quant_algo\": \"W4A16_NVFP4\"}";
    for ([_][]const u8{ "q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "indexer.wk", "indexer.weights_proj", "indexer.wq_b" }) |m| {
        s = s ++ ",\"" ++ P ++ mtp_at ++ "self_attn." ++ m ++ "\": {\"group_size\": 32, \"quant_algo\": \"MXFP8\"}";
    }
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |m| {
        s = s ++ ",\"" ++ P ++ mtp_at ++ "mlp.shared_experts." ++ m ++ "\": {\"group_size\": 32, \"quant_algo\": \"MXFP8\"}";
    }
    break :mtp s;
};
const P = "model.language_model.";

/// The synthetic checkpoint's forms: NVFP4 as nvidia's, FP4-Local's block FP8 added, or its MXFP8 too.
pub const Kind = enum { nvfp4, fp8pb, local };

fn nv4(comptime module: []const u8) []const u8 {
    return "\"" ++ P ++ module ++ "\": {\"quant_algo\": \"NVFP4\", \"group_size\": 16}";
}

fn pb(comptime module: []const u8) []const u8 {
    return "\"" ++ P ++ module ++ "\": {\"quant_algo\": \"FP8_PB_WO\", \"weight_block_size\": [128, 128]}";
}

fn mx8(comptime name: []const u8) []const u8 {
    return "\"" ++ name ++ "\": {\"quant_algo\": \"MXFP8\", \"weight_block_size\": [1, 32]}";
}

const Out = struct {
    a: std.mem.Allocator,
    rng: std.Random,
    list: std.ArrayList(New) = .empty,

    fn put(o: *Out, name: []const u8, dtype: st.DType, shape: []const usize, bytes: []const u8) !void {
        try o.list.append(o.a, .{ .name = name, .dtype = dtype, .shape = try o.a.dupe(usize, shape), .bytes = bytes });
    }

    /// bf16 values uniform in [-scale, scale).
    fn bf16(o: *Out, name: []const u8, shape: []const usize, scale: f32) !void {
        const v = try o.a.alloc(u16, count(shape));
        for (v) |*x| x.* = bf16Of((o.rng.float(f32) * 2 - 1) * scale);
        try o.put(name, .bf16, shape, std.mem.sliceAsBytes(v));
    }

    fn bf16Const(o: *Out, name: []const u8, shape: []const usize, value: f32) !void {
        const v = try o.a.alloc(u16, count(shape));
        @memset(v, bf16Of(value));
        try o.put(name, .bf16, shape, std.mem.sliceAsBytes(v));
    }

    /// A float32 norm weight 1 + u 2^-9 bf16 does not hold (FP4-Local's), from `g`, not the weights' stream.
    fn norm32(o: *Out, name: []const u8, g: std.Random) !void {
        const v = try o.a.alloc(f32, hidden);
        for (v) |*x| x.* = 1 + g.float(f32) * 0x1p-9;
        try o.put(name, .f32, &.{hidden}, std.mem.sliceAsBytes(v));
    }

    /// f32 values uniform in [lo, hi).
    fn f32s(o: *Out, name: []const u8, shape: []const usize, lo: f32, hi: f32) !void {
        const v = try o.a.alloc(f32, count(shape));
        for (v) |*x| x.* = lo + o.rng.float(f32) * (hi - lo);
        try o.put(name, .f32, shape, std.mem.sliceAsBytes(v));
    }

    /// A [k]-wide linear's rows scaled about 1/sqrt(k).
    fn linear(o: *Out, name: []const u8, n: usize, k: usize) !void {
        try o.bf16(name, &.{ n, k }, 1.0 / @sqrt(@as(f32, @floatFromInt(k))));
    }

    /// modelopt NVFP4: random e2m1 codes, e4m3 block scales 0.5 to 1.875, a global scale near 1/(6 sqrt(k)).
    fn nvfp4(o: *Out, base: []const u8, n: usize, k: usize) !void {
        try o.w4a16(base, n, k);
        try o.scalar(try cat(o.a, base, ".input_scale"), 1.0);
    }

    /// W4A16_NVFP4 (FP4-Local's MTP experts): as nvfp4 without an input_scale.
    fn w4a16(o: *Out, base: []const u8, n: usize, k: usize) !void {
        const codes = try o.a.alloc(u8, n * k / 2);
        o.rng.bytes(codes);
        try o.put(try cat(o.a, base, ".weight"), .u8, &.{ n, k / 2 }, codes);
        const scales = try o.a.alloc(u8, n * k / 16);
        for (scales) |*x| x.* = 0x30 + o.rng.uintLessThan(u8, 16);
        try o.put(try cat(o.a, base, ".weight_scale"), .f8_e4m3, &.{ n, k / 16 }, scales);
        try o.scalar(try cat(o.a, base, ".weight_scale_2"), 1.0 / (6.0 * @sqrt(@as(f32, @floatFromInt(k)))));
    }

    /// e4m3 codes [n, k] of magnitude 2^-4 to 1.875: never 0x7f or 0xff (NaN).
    fn e4m3(o: *Out, base: []const u8, n: usize, k: usize) !void {
        const codes = try o.a.alloc(u8, n * k);
        for (codes) |*x| x.* = (o.rng.uintLessThan(u8, 2) << 7) | ((3 + o.rng.uintLessThan(u8, 5)) << 3) | o.rng.uintLessThan(u8, 8);
        try o.put(try cat(o.a, base, ".weight"), .f8_e4m3, &.{ n, k }, codes);
    }

    /// FP8_PB_WO: e4m3 codes, fp32 scales [Rb, 1, Cb, 1] near 1/(2 sqrt(k)).
    fn fp8pb(o: *Out, base: []const u8, n: usize, k: usize) !void {
        try o.e4m3(base, n, k);
        const rb = (n + 127) / 128;
        const cb = (k + 127) / 128;
        const v = try o.a.alloc(f32, rb * cb);
        for (v) |*x| x.* = (0.5 + o.rng.float(f32)) / (2.0 * @sqrt(@as(f32, @floatFromInt(k))));
        try o.put(try cat(o.a, base, ".weight_scale"), .f32, &.{ rb, 1, cb, 1 }, std.mem.sliceAsBytes(v));
    }

    /// MXFP8: e4m3 codes, a UE8M0 scale byte a 32 inputs within a step of linear()'s RMS 1/sqrt(3k) over the codes'.
    fn mxfp8(o: *Out, base: []const u8, n: usize, k: usize) !void {
        try o.e4m3(base, n, k);
        const codes_rms = 0.7566; // e4m3()'s draw: (1 + m/8) 2^(e - 7), m 0 to 7 and e 3 to 7 alike
        const e: u8 = @intFromFloat(127 + @round(-std.math.log2(codes_rms * @sqrt(3.0 * @as(f64, @floatFromInt(k))))));
        const scales = try o.a.alloc(u8, n * k / 32);
        for (scales) |*x| x.* = e - 1 + o.rng.uintLessThan(u8, 3);
        try o.put(try cat(o.a, base, ".weight_scale"), .u8, &.{ n, k / 32 }, scales);
    }

    /// A linear as bf16, or as FP8_PB_WO with `fp8`.
    fn either(o: *Out, base: []const u8, n: usize, k: usize, fp8: bool) !void {
        if (fp8) return o.fp8pb(base, n, k);
        try o.linear(try cat(o.a, base, ".weight"), n, k);
    }

    /// A linear as bf16, or as MXFP8 with `mx`.
    fn bfOrMx(o: *Out, base: []const u8, n: usize, k: usize, mx: bool) !void {
        if (mx) return o.mxfp8(base, n, k);
        try o.linear(try cat(o.a, base, ".weight"), n, k);
    }

    /// A linear as bf16, FP8_PB_WO with `fp8`, or MXFP8 with `mx`.
    fn any(o: *Out, base: []const u8, n: usize, k: usize, fp8: bool, mx: bool) !void {
        if (mx) return o.mxfp8(base, n, k);
        try o.either(base, n, k, fp8);
    }

    fn scalar(o: *Out, name: []const u8, value: f32) !void {
        const v = try o.a.alloc(f32, 1);
        v[0] = value;
        try o.put(name, .f32, &.{}, std.mem.sliceAsBytes(v));
    }
};

fn count(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

fn bf16Of(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn cat(a: std.mem.Allocator, x: []const u8, y: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ x, y });
}

fn tensor(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, P ++ fmt, args);
}

fn hc(o: *Out, i: usize) !void {
    for ([_][]const u8{ "attn", "ffn" }) |w| {
        try o.bf16(try tensor(o.a, "layers.{d}.hc_{s}_fn", .{ i, w }), &.{ 24, 4 * hidden }, 1.0 / 128.0);
        try o.bf16(try tensor(o.a, "layers.{d}.hc_{s}_base", .{ i, w }), &.{24}, 0.1);
        try o.bf16Const(try tensor(o.a, "layers.{d}.hc_{s}_scale", .{ i, w }), &.{3}, 1.0);
    }
}

/// Layer i's norms, ones in bf16; with `g` (local) its post-attention norm float32 as FP4-Local stores it.
fn norms(o: *Out, i: usize, g: ?std.Random) !void {
    try o.bf16Const(try tensor(o.a, "layers.{d}.input_layernorm.weight", .{i}), &.{hidden}, 1.0);
    const post = try tensor(o.a, "layers.{d}.post_attention_layernorm.weight", .{i});
    if (g) |r| try o.norm32(post, r) else try o.bf16Const(post, &.{hidden}, 1.0);
}

fn kda(o: *Out, i: usize, mx: bool) !void {
    const s = try tensor(o.a, "layers.{d}.self_attn.", .{i});
    for ([_][]const u8{ "q", "k", "v" }) |m| {
        try o.bfOrMx(try std.fmt.allocPrint(o.a, "{s}{s}_proj", .{ s, m }), 8192, hidden, mx);
        try o.f32s(try std.fmt.allocPrint(o.a, "{s}{s}_conv1d.weight", .{ s, m }), &.{ 8192, 1, 4 }, -0.5, 0.5);
    }
    try o.linear(try cat(o.a, s, "f_a_proj.weight"), 128, hidden);
    try o.linear(try cat(o.a, s, "g_a_proj.weight"), 128, hidden);
    try o.linear(try cat(o.a, s, "b_proj.weight"), 64, hidden);
    try o.linear(try cat(o.a, s, "f_b_proj.weight"), 8192, 128);
    try o.linear(try cat(o.a, s, "g_b_proj.weight"), 8192, 128);
    try o.f32s(try cat(o.a, s, "A_log"), &.{64}, 0.0, 2.5);
    try o.f32s(try cat(o.a, s, "dt_bias"), &.{8192}, -0.5, 0.5);
    try o.bf16Const(try cat(o.a, s, "o_norm.weight"), &.{128}, 1.0);
    try o.bfOrMx(try cat(o.a, s, "o_proj"), hidden, 8192, mx);
}

/// Latent attention in bf16, its projections in FP8_PB_WO with `fp8`, or with `mx` all of it but norms in MXFP8.
fn mla(o: *Out, i: usize, fp8: bool, mx: bool) !void {
    const s = try tensor(o.a, "layers.{d}.self_attn.", .{i});
    try o.any(try cat(o.a, s, "q_a_proj"), 1536, hidden, fp8, mx);
    try o.any(try cat(o.a, s, "kv_a_proj_with_mqa"), 512, hidden, fp8, mx);
    try o.bf16Const(try cat(o.a, s, "q_a_layernorm.weight"), &.{1536}, 1.0);
    try o.bf16Const(try cat(o.a, s, "kv_a_layernorm.weight"), &.{512}, 1.0);
    try o.any(try cat(o.a, s, "q_b_proj"), 16384, 1536, fp8, mx);
    try o.bfOrMx(try cat(o.a, s, "kv_b_proj"), 32768, 512, mx);
    try o.any(try cat(o.a, s, "o_proj"), hidden, 16384, fp8, mx);
    try o.bfOrMx(try cat(o.a, s, "indexer.wk"), 128, hidden, mx);
    try o.bfOrMx(try cat(o.a, s, "indexer.weights_proj"), 32, hidden, mx);
    try o.bfOrMx(try cat(o.a, s, "indexer.wq_b"), 4096, 1536, mx);
    try o.bf16Const(try cat(o.a, s, "indexer.k_norm.weight"), &.{128}, 1.0);
    try o.bf16Const(try cat(o.a, s, "indexer.k_norm.bias"), &.{128}, 0.0);
    try o.linear(try cat(o.a, s, "indexer.index_kpool_compress_gate"), 128, hidden);
    try o.bf16(try cat(o.a, s, "indexer.index_kpool_compress_ape"), &.{ 4, 128 }, 0.02);
}

fn dense(o: *Out, i: usize, mx: bool) !void {
    const shapes = [_]struct { []const u8, usize, usize }{ .{ "gate", 12288, hidden }, .{ "up", 12288, hidden }, .{ "down", hidden, 12288 } };
    for (shapes) |x| {
        const base = try tensor(o.a, "layers.{d}.mlp.{s}_proj", .{ i, x[0] });
        if (mx) try o.mxfp8(base, x[1], x[2]) else try o.nvfp4(base, x[1], x[2]);
    }
}

/// The routed experts' form: bf16 (the MTP layer's, nvidia's), NVFP4, or W4A16_NVFP4 (FP4-Local's MTP layer).
const Experts = enum { bf16, nvfp4, w4a16 };

/// The router, the shared expert (bf16, block FP8 with `fp8`, MXFP8 with `mx`) and the routed ones.
fn moe(o: *Out, i: usize, routed: Experts, fp8: bool, mx: bool) !void {
    try o.linear(try tensor(o.a, "layers.{d}.mlp.gate.weight", .{i}), experts, hidden);
    try o.f32s(try tensor(o.a, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}), &.{experts}, 0.0, 0.0);
    try o.any(try tensor(o.a, "layers.{d}.mlp.shared_experts.gate_proj", .{i}), 2048, hidden, fp8, mx);
    try o.any(try tensor(o.a, "layers.{d}.mlp.shared_experts.up_proj", .{i}), 2048, hidden, fp8, mx);
    try o.any(try tensor(o.a, "layers.{d}.mlp.shared_experts.down_proj", .{i}), hidden, 2048, fp8, mx);
    for (0..experts) |e| {
        const s = try tensor(o.a, "layers.{d}.mlp.experts.{d}.", .{ i, e });
        const shapes = [_]struct { []const u8, usize, usize }{ .{ "gate_proj", 2048, hidden }, .{ "up_proj", 2048, hidden }, .{ "down_proj", hidden, 2048 } };
        for (shapes) |x| switch (routed) {
            .bf16 => try o.linear(try std.fmt.allocPrint(o.a, "{s}{s}.weight", .{ s, x[0] }), x[1], x[2]),
            .nvfp4 => try o.nvfp4(try cat(o.a, s, x[0]), x[1], x[2]),
            .w4a16 => try o.w4a16(try cat(o.a, s, x[0]), x[1], x[2]),
        };
    }
}

pub fn run(gpu: check.Gpu, out_dir: []const u8, form: Kind) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(53);
    var o: Out = .{ .a = a, .rng = prng.random() };
    var g32 = std.Random.DefaultPrng.init(32); // local's float32 norms: a stream apart from the other tensors'
    const fp8 = form != .nvfp4;
    const mx = form == .local;
    const g: ?std.Random = if (mx) g32.random() else null;
    try o.bf16(P ++ "embed_tokens.weight", &.{ vocab, hidden }, 1.0);
    if (g) |r| try o.norm32(P ++ "norm.weight", r) else try o.bf16Const(P ++ "norm.weight", &.{hidden}, 1.0);
    try o.bfOrMx("lm_head", vocab, hidden, mx);
    try kda(&o, 0, mx);
    try hc(&o, 0);
    try norms(&o, 0, g);
    try dense(&o, 0, mx);
    try mla(&o, 1, fp8, false);
    try hc(&o, 1);
    try norms(&o, 1, g);
    try moe(&o, 1, .nvfp4, fp8, false);
    // the MTP layer: no hyper-connections, the draft head's norms and eh_proj; bf16, or with local FP4-Local's forms
    try mla(&o, layers, false, mx);
    try norms(&o, layers, null);
    try moe(&o, layers, if (mx) .w4a16 else .bf16, false, mx);
    for ([_][]const u8{ "enorm", "hnorm", "shared_head.norm" }) |m| try o.bf16Const(try tensor(a, "layers.{d}.{s}.weight", .{ layers, m }), &.{hidden}, 1.0);
    try o.linear(try tensor(a, "layers.{d}.eh_proj.weight", .{layers}), hidden, 2 * hidden);

    try std.Io.Dir.cwd().createDirPath(gpu.io, out_dir);
    const text = try glm.config.textConfig(a, layers, 1, &.{1}, experts, vocab, hidden);
    const end = std.mem.lastIndexOfScalar(u8, text, '}') orelse return error.BadConfig;
    const block = switch (form) {
        .nvfp4 => modelopt_block,
        .fp8pb => mixed_block,
        .local => local_block,
    };
    const cfg = try std.mem.concat(a, u8, &.{ text[0..end], block, text[end..] });
    _ = try glm.config.parse(a, cfg);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "config.json" }), .data = cfg });
    try core.shard_edit.create(gpu.gpa, try std.fs.path.join(a, &.{ out_dir, "model.safetensors" }), &.{.{ "format", "pt" }}, o.list.items);

    // three prompts of the decode checks' lengths, ids under the eos at vocab - 2; want: 512 zeros (a length only)
    var dc: std.Io.Writer.Allocating = .init(a);
    try dc.writer.writeAll("{\"capacity\": 8192, \"kinds\": [");
    for ([_][]const u8{ "count", "prose", "code" }, [_]usize{ 2066, 2108, 2089 }, 0..) |kind, len, k| {
        try dc.writer.print("{s}{{\"kind\": \"{s}\", \"prompt\": [", .{ if (k > 0) ", " else "", kind });
        for (0..len) |i| try dc.writer.print("{s}{d}", .{ if (i > 0) "," else "", o.rng.uintLessThan(u32, vocab - 128) });
        try dc.writer.writeAll("], \"want\": [");
        for (0..512) |i| try dc.writer.writeAll(if (i > 0) ",0" else "0");
        try dc.writer.writeAll("]}");
    }
    try dc.writer.writeAll("]}\n");
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "dc.json" }), .data = dc.written() });
    var bytes: usize = 0;
    for (o.list.items) |t| bytes += t.bytes.len;
    const forms = switch (form) {
        .nvfp4 => "",
        .fp8pb => ", layer 1's MLA and shared in FP8_PB_WO",
        .local => ", layer 1's MLA and shared in FP8_PB_WO, layer 0's KDA q k v o, dense MLP, lm_head and the MTP layer's " ++
            "attention, indexer and shared in MXFP8, its experts W4A16_NVFP4, the post-attention and final norms float32",
    };
    check.pass("glm-synth: {d} tensors, {d} MiB, config {d} layers {d} experts vocabulary {d}{s}, dc.json in {s}", .{ o.list.items.len, bytes >> 20, layers, experts, vocab, forms, out_dir });
}
