//! `glm-synth OUT_DIR`: a 2-layer + MTP GLM-5.3-Flash checkpoint, dc.json; run TF_GLM_LAYERS=2.
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
const P = "model.language_model.";

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

fn mla(o: *Out, i: usize) !void {
    const s = try tensor(o.a, "layers.{d}.self_attn.", .{i});
    try o.linear(try cat(o.a, s, "q_a_proj.weight"), 1536, hidden);
    try o.linear(try cat(o.a, s, "kv_a_proj_with_mqa.weight"), 512, hidden);
    try o.bf16Const(try cat(o.a, s, "q_a_layernorm.weight"), &.{1536}, 1.0);
    try o.bf16Const(try cat(o.a, s, "kv_a_layernorm.weight"), &.{512}, 1.0);
    try o.linear(try cat(o.a, s, "q_b_proj.weight"), 16384, 1536);
    try o.linear(try cat(o.a, s, "kv_b_proj.weight"), 32768, 512);
    try o.linear(try cat(o.a, s, "o_proj.weight"), hidden, 16384);
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

/// The router, the shared expert, and the routed ones in NVFP4 (backbone) or bf16 (the MTP layer).
fn moe(o: *Out, i: usize, bf: bool) !void {
    try o.linear(try tensor(o.a, "layers.{d}.mlp.gate.weight", .{i}), experts, hidden);
    try o.f32s(try tensor(o.a, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}), &.{experts}, 0.0, 0.0);
    try o.linear(try tensor(o.a, "layers.{d}.mlp.shared_experts.gate_proj.weight", .{i}), 2048, hidden);
    try o.linear(try tensor(o.a, "layers.{d}.mlp.shared_experts.up_proj.weight", .{i}), 2048, hidden);
    try o.linear(try tensor(o.a, "layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), hidden, 2048);
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

pub fn run(gpu: check.Gpu, out_dir: []const u8) !void {
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
    try mla(&o, 1);
    try hc(&o, 1);
    try norms(&o, 1);
    try moe(&o, 1, false);
    try mla(&o, layers); // the MTP layer: no hyper-connections, bf16 routed experts, the draft head's norms and eh_proj
    try norms(&o, layers);
    try moe(&o, layers, true);
    for ([_][]const u8{ "enorm", "hnorm", "shared_head.norm" }) |m| try o.bf16Const(try tensor(a, "layers.{d}.{s}.weight", .{ layers, m }), &.{hidden}, 1.0);
    try o.linear(try tensor(a, "layers.{d}.eh_proj.weight", .{layers}), hidden, 2 * hidden);

    try std.Io.Dir.cwd().createDirPath(gpu.io, out_dir);
    const cfg = try glm.config.textConfig(a, layers, 1, &.{1}, experts, vocab, hidden);
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
    check.pass("glm-synth: {d} tensors, {d} MiB, config {d} layers {d} experts vocabulary {d}, dc.json in {s}", .{ o.list.items.len, bytes >> 20, layers, experts, vocab, out_dir });
}
