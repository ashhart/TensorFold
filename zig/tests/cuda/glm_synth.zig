//! `glm-synth OUT_DIR [fp8pb]`: a 2-layer + MTP GLM-5.3-Flash checkpoint, dc.json; run TF_GLM_LAYERS=2.
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
const mixed_block = ",\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\", " ++
    "\"quantized_layers\": {" ++ nv4("layers.0.mlp") ++ "," ++ nv4("layers.1.mlp.experts") ++ "," ++ pb("layers.1.mlp.shared_experts") ++ "," ++
    pb("layers.1.self_attn.q_a_proj") ++ "," ++ pb("layers.1.self_attn.kv_a_proj_with_mqa") ++ "," ++ pb("layers.1.self_attn.q_b_proj") ++ "," ++
    pb("layers.1.self_attn.o_proj") ++ "}}";
const P = "model.language_model.";

fn nv4(comptime module: []const u8) []const u8 {
    return "\"" ++ P ++ module ++ "\": {\"quant_algo\": \"NVFP4\", \"group_size\": 16}";
}

fn pb(comptime module: []const u8) []const u8 {
    return "\"" ++ P ++ module ++ "\": {\"quant_algo\": \"FP8_PB_WO\", \"weight_block_size\": [128, 128]}";
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
        const codes = try o.a.alloc(u8, n * k / 2);
        o.rng.bytes(codes);
        try o.put(try cat(o.a, base, ".weight"), .u8, &.{ n, k / 2 }, codes);
        const scales = try o.a.alloc(u8, n * k / 16);
        for (scales) |*x| x.* = 0x30 + o.rng.uintLessThan(u8, 16);
        try o.put(try cat(o.a, base, ".weight_scale"), .f8_e4m3, &.{ n, k / 16 }, scales);
        try o.scalar(try cat(o.a, base, ".weight_scale_2"), 1.0 / (6.0 * @sqrt(@as(f32, @floatFromInt(k)))));
        try o.scalar(try cat(o.a, base, ".input_scale"), 1.0);
    }

    /// FP8_PB_WO: e4m3 codes of magnitude 2^-4 to 1.875 (no NaN), fp32 scales [Rb, 1, Cb, 1] near 1/(2 sqrt(k)).
    fn fp8pb(o: *Out, base: []const u8, n: usize, k: usize) !void {
        const codes = try o.a.alloc(u8, n * k);
        for (codes) |*x| x.* = (o.rng.uintLessThan(u8, 2) << 7) | ((3 + o.rng.uintLessThan(u8, 5)) << 3) | o.rng.uintLessThan(u8, 8);
        try o.put(try cat(o.a, base, ".weight"), .f8_e4m3, &.{ n, k }, codes);
        const rb = (n + 127) / 128;
        const cb = (k + 127) / 128;
        const v = try o.a.alloc(f32, rb * cb);
        for (v) |*x| x.* = (0.5 + o.rng.float(f32)) / (2.0 * @sqrt(@as(f32, @floatFromInt(k))));
        try o.put(try cat(o.a, base, ".weight_scale"), .f32, &.{ rb, 1, cb, 1 }, std.mem.sliceAsBytes(v));
    }

    /// A linear as bf16, or as FP8_PB_WO with `fp8`.
    fn either(o: *Out, base: []const u8, n: usize, k: usize, fp8: bool) !void {
        if (fp8) return o.fp8pb(base, n, k);
        try o.linear(try cat(o.a, base, ".weight"), n, k);
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

fn norms(o: *Out, i: usize) !void {
    try o.bf16Const(try tensor(o.a, "layers.{d}.input_layernorm.weight", .{i}), &.{hidden}, 1.0);
    try o.bf16Const(try tensor(o.a, "layers.{d}.post_attention_layernorm.weight", .{i}), &.{hidden}, 1.0);
}

fn kda(o: *Out, i: usize) !void {
    const s = try tensor(o.a, "layers.{d}.self_attn.", .{i});
    for ([_][]const u8{ "q", "k", "v" }) |m| {
        try o.linear(try std.fmt.allocPrint(o.a, "{s}{s}_proj.weight", .{ s, m }), 8192, hidden);
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
    try o.linear(try cat(o.a, s, "o_proj.weight"), hidden, 8192);
}

fn mla(o: *Out, i: usize, fp8: bool) !void {
    const s = try tensor(o.a, "layers.{d}.self_attn.", .{i});
    try o.either(try cat(o.a, s, "q_a_proj"), 1536, hidden, fp8);
    try o.either(try cat(o.a, s, "kv_a_proj_with_mqa"), 512, hidden, fp8);
    try o.bf16Const(try cat(o.a, s, "q_a_layernorm.weight"), &.{1536}, 1.0);
    try o.bf16Const(try cat(o.a, s, "kv_a_layernorm.weight"), &.{512}, 1.0);
    try o.either(try cat(o.a, s, "q_b_proj"), 16384, 1536, fp8);
    try o.linear(try cat(o.a, s, "kv_b_proj.weight"), 32768, 512);
    try o.either(try cat(o.a, s, "o_proj"), hidden, 16384, fp8);
    try o.linear(try cat(o.a, s, "indexer.wk.weight"), 128, hidden);
    try o.linear(try cat(o.a, s, "indexer.weights_proj.weight"), 32, hidden);
    try o.linear(try cat(o.a, s, "indexer.wq_b.weight"), 4096, 1536);
    try o.bf16Const(try cat(o.a, s, "indexer.k_norm.weight"), &.{128}, 1.0);
    try o.bf16Const(try cat(o.a, s, "indexer.k_norm.bias"), &.{128}, 0.0);
    try o.linear(try cat(o.a, s, "indexer.index_kpool_compress_gate"), 128, hidden);
    try o.bf16(try cat(o.a, s, "indexer.index_kpool_compress_ape"), &.{ 4, 128 }, 0.02);
}

fn dense(o: *Out, i: usize) !void {
    for ([_][]const u8{ "gate", "up" }) |m| try o.nvfp4(try tensor(o.a, "layers.{d}.mlp.{s}_proj", .{ i, m }), 12288, hidden);
    try o.nvfp4(try tensor(o.a, "layers.{d}.mlp.down_proj", .{i}), hidden, 12288);
}

/// The router, the shared expert (bf16, or block FP8 with `fp8`), the routed ones in NVFP4 or bf16 (the MTP layer).
fn moe(o: *Out, i: usize, bf: bool, fp8: bool) !void {
    try o.linear(try tensor(o.a, "layers.{d}.mlp.gate.weight", .{i}), experts, hidden);
    try o.f32s(try tensor(o.a, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}), &.{experts}, 0.0, 0.0);
    try o.either(try tensor(o.a, "layers.{d}.mlp.shared_experts.gate_proj", .{i}), 2048, hidden, fp8);
    try o.either(try tensor(o.a, "layers.{d}.mlp.shared_experts.up_proj", .{i}), 2048, hidden, fp8);
    try o.either(try tensor(o.a, "layers.{d}.mlp.shared_experts.down_proj", .{i}), hidden, 2048, fp8);
    for (0..experts) |e| {
        const s = try tensor(o.a, "layers.{d}.mlp.experts.{d}.", .{ i, e });
        if (bf) {
            try o.linear(try cat(o.a, s, "gate_proj.weight"), 2048, hidden);
            try o.linear(try cat(o.a, s, "up_proj.weight"), 2048, hidden);
            try o.linear(try cat(o.a, s, "down_proj.weight"), hidden, 2048);
        } else {
            try o.nvfp4(try cat(o.a, s, "gate_proj"), 2048, hidden);
            try o.nvfp4(try cat(o.a, s, "up_proj"), 2048, hidden);
            try o.nvfp4(try cat(o.a, s, "down_proj"), hidden, 2048);
        }
    }
}

pub fn run(gpu: check.Gpu, out_dir: []const u8, fp8: bool) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(53);
    var o: Out = .{ .a = a, .rng = prng.random() };
    try o.bf16(P ++ "embed_tokens.weight", &.{ vocab, hidden }, 1.0);
    try o.bf16Const(P ++ "norm.weight", &.{hidden}, 1.0);
    try o.linear("lm_head.weight", vocab, hidden);
    try kda(&o, 0);
    try hc(&o, 0);
    try norms(&o, 0);
    try dense(&o, 0);
    try mla(&o, 1, fp8);
    try hc(&o, 1);
    try norms(&o, 1);
    try moe(&o, 1, false, fp8);
    // the MTP layer: no hyper-connections, bf16 attention and routed experts, the draft head's norms and eh_proj
    try mla(&o, layers, false);
    try norms(&o, layers);
    try moe(&o, layers, true, false);
    for ([_][]const u8{ "enorm", "hnorm", "shared_head.norm" }) |m| try o.bf16Const(try tensor(a, "layers.{d}.{s}.weight", .{ layers, m }), &.{hidden}, 1.0);
    try o.linear(try tensor(a, "layers.{d}.eh_proj.weight", .{layers}), hidden, 2 * hidden);

    try std.Io.Dir.cwd().createDirPath(gpu.io, out_dir);
    const text = try glm.config.textConfig(a, layers, 1, &.{1}, experts, vocab, hidden);
    const end = std.mem.lastIndexOfScalar(u8, text, '}') orelse return error.BadConfig;
    const cfg = try std.mem.concat(a, u8, &.{ text[0..end], if (fp8) mixed_block else modelopt_block, text[end..] });
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
    check.pass("glm-synth: {d} tensors, {d} MiB, config {d} layers {d} experts vocabulary {d}{s}, dc.json in {s}", .{ o.list.items.len, bytes >> 20, layers, experts, vocab, if (fp8) ", layer 1's MLA and shared in FP8_PB_WO" else "", out_dir });
}
