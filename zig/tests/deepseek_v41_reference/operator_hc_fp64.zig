//! FP64 ideal-reference isolation for the mHC stage: rounding level, Metal deltas and 1-ULP
//! input amplification. Receipt lines print with an "HC64" prefix. No tolerance is relaxed and
//! no source coefficient is injected: retained rows and real checkpoint scale/base are inputs.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");

const mhc = tf.deepseek_v41.mhc;
const ops = tf.deepseek_v41.ops;
const View = tf.deepseek_v41.attention.View;
const Tensor = tf.checkpoint_metal.Tensor;
const Hc = tf.deepseek_v41_weights.Hc;
const Config = tf.deepseek_v41_config.Config;

const width: usize = 5120;
const copies: usize = 4;
const stream_width: usize = width * copies;
const mix_count: usize = 24;

const released_config =
    \\{"model_type":"deepseek_v41","quantization_config":{"quant_method":"exl3","version":"1"},
    \\"text_config":{"vocab_size":129280,"hidden_size":5120,"moe_intermediate_size":2304,
    \\"num_hidden_layers":40,"num_attention_heads":64,"num_key_value_heads":1,"head_dim":512,
    \\"qk_rope_head_dim":64,"q_lora_rank":1280,"o_lora_rank":1024,"o_groups":8,
    \\"swiglu_limit":10,"rms_norm_eps":0.000001,"max_position_embeddings":1048576,
    \\"rope_theta":10000,"rope_scaling":{"rope_type":"yarn","factor":16,"beta_fast":32,
    \\"beta_slow":1,"original_max_position_embeddings":65536},"n_routed_experts":384,
    \\"n_shared_experts":1,"num_experts_per_tok":6,"scoring_func":"sqrtsoftplus",
    \\"topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.5,
    \\"sliding_window":128,"compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    \\1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],"compress_rope_theta":10000,
    \\"kv_source_layer_ids":[2,8,14,20],"index_source_layer_ids":[2,8,14,20,24,28,32,36],
    \\"index_n_heads":32,"index_head_dim":128,"index_topk":512,"candidate_source_layer_id":20,
    \\"candidate_topk_blocks":2048,"candidate_block_size":8,"hc_mult":4,"hc_sinkhorn_iters":20,
    \\"hc_eps":0.000001,"engram_layer_ids":[1,14],"engram_num_embeddings":[384006168,384016682],
    \\"engram_max_ngram_size":4,"engram_vocab_size":16000000,"engram_n_heads":8,"engram_head_dim":256,
    \\"engram_pad_token_id":2,"engram_compressed_vocab_size":99092,"num_nextn_predict_layers":3}}
;

/// The 16 retained HC unary arguments (hc-exp-variants-input.json), including 1.4393715858459473.
const retained_exp_argument_bits = [_]u32{
    3217026610, 1081519143, 1049174916, 1078264130, 1073892416, 1078760804, 1083574034, 1069038932,
    1043563522, 1083490981, 1087210444, 1094911558, 3195364771, 1082252091, 1096635047, 1091866698,
};

/// The documented controlling exponent input: Metal FP32 0x4086fa39, ideal 0x4086fa38 (1 ULP).
const controlling_exponent_input: f32 = 1.4393715858459473;

const fixture_path = "../../src/families/deepseek_v41/fixtures/hc_fp64_rows.json";

const Fixture = struct {
    eight_rows_bf16: []const u8,
    ffn_stream_bf16: []const u8,
    ffn_branch_bf16: []const u8,
    ffn_scale_f32: []const u8,
    ffn_base_f32: []const u8,
    provenance: []const u8 = "",
};

fn decodeBase64(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    const size = try std.base64.standard.Decoder.calcSizeForSlice(text);
    const out = try gpa.alloc(u8, size);
    errdefer gpa.free(out);
    try std.base64.standard.Decoder.decode(out, text);
    return out;
}

fn decodeU16(gpa: std.mem.Allocator, text: []const u8) ![]u16 {
    const bytes = try decodeBase64(gpa, text);
    defer gpa.free(bytes);
    const out = try gpa.alloc(u16, bytes.len / 2);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return out;
}

fn decodeF32(gpa: std.mem.Allocator, text: []const u8) ![]f32 {
    const bytes = try decodeBase64(gpa, text);
    defer gpa.free(bytes);
    const out = try gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    fn next(s: *Lcg) u64 {
        s.state = s.state *% 6364136223846793005 +% 1442695040888963407;
        return s.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    fn uniform(s: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(s.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
};

fn nextUpF32(x: f32) f32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    const bits: u32 = @bitCast(x);
    if (bits == 0 or bits == 0x80000000) return @bitCast(@as(u32, 1));
    return @bitCast(if (bits >> 31 == 0) bits + 1 else bits - 1);
}

/// The local FP32 spacing at the FP64 reference value: the ULP unit for kernel deltas.
fn ulp32At(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f) or std.math.isInf(f)) return @abs(v);
    const a = @abs(f);
    if (a == 0) return 0x1p-149;
    const up = nextUpF32(a);
    if (std.math.isInf(up)) return @as(f64, a) - @as(f64, -nextUpF32(-a));
    return @as(f64, up) - @as(f64, a);
}

const Delta = struct {
    nrms: f64 = 0,
    max_abs: f64 = 0,
    max_ulp: f64 = 0,
    worst: usize = 0,
    count: usize = 0,
};

fn deltaF32VsF64(actual: []const f32, ref: []const f64) Delta {
    var d: Delta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ref, 0..) |a, r, i| {
        const diff = @as(f64, a) - r;
        squared += diff * diff;
        magnitude += r * r;
        const ad = @abs(diff);
        if (ad > d.max_abs) {
            d.max_abs = ad;
            d.worst = i;
        }
        const u = ad / ulp32At(r);
        if (u > d.max_ulp) d.max_ulp = u;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

const BfDelta = struct {
    nrms: f64 = 0,
    unequal_ideal: usize = 0,
    unequal_f32_ref: usize = 0,
    count: usize = 0,
    worst: usize = 0,
    worst_gap: f64 = 0,
};

/// BF16 outputs: decoded NRMS against the FP64 ideal cells plus ideal/F32-reference cell counts.
fn deltaBf16(actual: []const u16, ideal: []const u16, f32_ref: []const u16) BfDelta {
    var d: BfDelta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, f32_ref, 0..) |a, id, fr, i| {
        const av: f64 = @floatCast(math.f(a));
        const iv: f64 = @floatCast(math.f(id));
        const diff = av - iv;
        squared += diff * diff;
        magnitude += iv * iv;
        if (a != id) {
            d.unequal_ideal += 1;
            if (@abs(diff) > d.worst_gap) {
                d.worst_gap = @abs(diff);
                d.worst = i;
            }
        }
        if (a != fr) d.unequal_f32_ref += 1;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

fn StageRefOf(comptime T: type) type {
    return struct {
        inv: T,
        proj: [mix_count]T,
        pre: [4]T,
        post: [4]T,
        comb: [16]T,
        collapsed: [width]u16,
        y: [stream_width]u16,
    };
}

fn referenceRow(comptime T: type, stream: []const u16, hc: Hc, cfg: Config, branch: []const u16, collapsed: *[width]u16, y: *[stream_width]u16) StageRefOf(T) {
    var out: StageRefOf(T) = undefined;
    out.inv = math.inverseOf(T, stream, cfg.text_config.rms_norm_eps);
    math.denseOf(T, hc.projection, stream, &out.proj);
    const mix = math.splitOf(T, &out.proj, out.inv, hc, cfg.text_config);
    out.pre = mix.pre;
    out.post = mix.post;
    out.comb = mix.comb;
    math.collapseOf(T, stream, mix.pre, collapsed);
    var h: [stream_width]u16 = undefined;
    @memcpy(&h, stream);
    math.expandOf(T, &h, branch, mix);
    @memcpy(y, &h);
    return out;
}

const Buffers = struct {
    device: mtl.Device,
    items: [24]mtl.Buffer = undefined,
    count: usize = 0,

    fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), n), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};

fn upload(comptime T: type, buffer: mtl.Buffer, data: []const T) void {
    @memcpy(buffer.slice(T, data.len), data);
}

fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

fn runStage(
    queue: mtl.Queue,
    p: *const mhc.Pipelines,
    o: *const ops.Pipelines,
    rows: u32,
    cfg: Config,
    stream: View,
    weight: View,
    inv: View,
    mixes: View,
    scale: View,
    base: View,
    pre: View,
    post: View,
    comb: View,
    collapsed: View,
    branch: View,
    y: View,
) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeRms(e, p, stream, inv, .{ .rows = rows, .width = width, .stream_stride = stream_width, .eps = cfg.text_config.rms_norm_eps });
    ops.encodeDense(e, o, stream, weight, mixes, .{ .rows = rows, .k = stream_width, .n = mix_count, .x_stride = stream_width, .w_stride = stream_width, .y_stride = mix_count, .w_dtype = .f32, .y_dtype = .f32 });
    e.barrier();
    mhc.encodeSplit(e, p, mixes, inv, scale, base, pre, post, comb, .{ .rows = rows, .eps = cfg.text_config.hc_eps });
    e.barrier();
    mhc.encodeCollapse(e, p, stream, pre, collapsed, .{ .rows = rows, .width = width, .stream_stride = stream_width, .y_stride = width });
    e.barrier();
    mhc.encodePost(e, p, branch, stream, post, comb, y, .{ .rows = rows, .width = width, .branch_stride = width, .residual_stride = stream_width, .y_stride = stream_width });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;
}

const Case = struct {
    name: []const u8,
    rows: usize,
    stream: []u16,
    weight: []f32,
    branch: []u16,
    scale: []f32,
    base: []f32,
};

const StageMetric = struct {
    name: []const u8,
    metal: Delta = .{},
    f32ref: Delta = .{},
    bf16: ?BfDelta = null,
};

fn printMetric(case_name: []const u8, m: StageMetric) void {
    if (m.bf16) |bf| {
        std.debug.print("HC64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"elements\":{d},\"nrms_metal_vs_ideal\":{d},\"bf16_unequal_metal_vs_ideal\":{d},\"bf16_unequal_f32ref_vs_ideal\":{d},\"worst_gap\":{d},\"worst_index\":{d}}}\n", .{ case_name, m.name, bf.count, bf.nrms, m.bf16.?.unequal_ideal, m.bf16.?.unequal_f32_ref, m.bf16.?.worst_gap, m.bf16.?.worst });
    } else {
        std.debug.print("HC64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"elements\":{d},\"nrms_metal_vs_f64\":{d},\"max_ulp_metal_vs_f64\":{d:.4},\"max_abs_metal_vs_f64\":{d},\"worst_metal\":{d},\"nrms_f32_vs_f64\":{d},\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ case_name, m.name, m.metal.count, m.metal.nrms, m.metal.max_ulp, m.metal.max_abs, m.metal.worst, m.f32ref.nrms, m.f32ref.max_ulp });
    }
}

/// One full HC stage over `rows` streams: statistic, dense, split, collapse and post-expansion,
/// each side (Metal, FP32 source-contract reference, FP64 ideal reference) on identical inputs.
fn runCase(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const mhc.Pipelines,
    o: *const ops.Pipelines,
    cfg: Config,
    case: Case,
) !void {
    var b = Buffers{ .device = device };
    defer b.deinit();
    const rows: u32 = @intCast(case.rows);
    const n = case.rows;
    const stream = try b.get(u16, n * stream_width);
    upload(u16, stream, case.stream);
    const weight = try b.get(f32, mix_count * stream_width);
    upload(f32, weight, case.weight);
    const inv = try b.get(f32, n);
    const mixes = try b.get(f32, n * mix_count);
    const scale = try b.get(f32, 3);
    upload(f32, scale, case.scale);
    const base = try b.get(f32, mix_count);
    upload(f32, base, case.base);
    const pre = try b.get(f32, n * 4);
    const post = try b.get(f32, n * 4);
    const comb = try b.get(f32, n * 16);
    const collapsed = try b.get(u16, n * width);
    const branch = try b.get(u16, n * width);
    upload(u16, branch, case.branch);
    const y = try b.get(u16, n * stream_width);
    try runStage(queue, p, o, rows, cfg, view(stream), view(weight), view(inv), view(mixes), view(scale), view(base), view(pre), view(post), view(comb), view(collapsed), view(branch), view(y));

    const weight_tensor: Tensor = .{ .buffer = weight, .offset = 0, .bytes = mix_count * stream_width * 4, .dtype = .f32 };
    const scale_tensor: Tensor = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 };
    const base_tensor: Tensor = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 };
    const hc = Hc{ .projection = weight_tensor, .scale = scale_tensor, .base = base_tensor };

    // Gathered arrays: Metal (m), FP32 source-contract reference (32), FP64 ideal (64).
    const inv_m = try gpa.alloc(f32, n);
    defer gpa.free(inv_m);
    @memcpy(inv_m, inv.slice(f32, n));
    const proj_m = try gpa.alloc(f32, n * mix_count);
    defer gpa.free(proj_m);
    @memcpy(proj_m, mixes.slice(f32, n * mix_count));
    const pre_m = try gpa.alloc(f32, n * 4);
    defer gpa.free(pre_m);
    @memcpy(pre_m, pre.slice(f32, n * 4));
    const post_m = try gpa.alloc(f32, n * 4);
    defer gpa.free(post_m);
    @memcpy(post_m, post.slice(f32, n * 4));
    const comb_m = try gpa.alloc(f32, n * 16);
    defer gpa.free(comb_m);
    @memcpy(comb_m, comb.slice(f32, n * 16));
    const collapsed_m = try gpa.alloc(u16, n * width);
    defer gpa.free(collapsed_m);
    @memcpy(collapsed_m, collapsed.slice(u16, n * width));
    const y_m = try gpa.alloc(u16, n * stream_width);
    defer gpa.free(y_m);
    @memcpy(y_m, y.slice(u16, n * stream_width));

    const inv_64 = try gpa.alloc(f64, n);
    defer gpa.free(inv_64);
    const proj_64 = try gpa.alloc(f64, n * mix_count);
    defer gpa.free(proj_64);
    const pre_64 = try gpa.alloc(f64, n * 4);
    defer gpa.free(pre_64);
    const post_64 = try gpa.alloc(f64, n * 4);
    defer gpa.free(post_64);
    const comb_64 = try gpa.alloc(f64, n * 16);
    defer gpa.free(comb_64);
    const collapsed_64 = try gpa.alloc(u16, n * width);
    defer gpa.free(collapsed_64);
    const y_64 = try gpa.alloc(u16, n * stream_width);
    defer gpa.free(y_64);

    const inv_32 = try gpa.alloc(f32, n);
    defer gpa.free(inv_32);
    const proj_32 = try gpa.alloc(f32, n * mix_count);
    defer gpa.free(proj_32);
    const pre_32 = try gpa.alloc(f32, n * 4);
    defer gpa.free(pre_32);
    const post_32 = try gpa.alloc(f32, n * 4);
    defer gpa.free(post_32);
    const comb_32 = try gpa.alloc(f32, n * 16);
    defer gpa.free(comb_32);
    const collapsed_32 = try gpa.alloc(u16, n * width);
    defer gpa.free(collapsed_32);
    const y_32 = try gpa.alloc(u16, n * stream_width);
    defer gpa.free(y_32);

    for (0..n) |r| {
        const stream_row = case.stream[r * stream_width ..][0..stream_width];
        const branch_row = case.branch[r * width ..][0..width];
        var collapsed64: [width]u16 = undefined;
        var y64: [stream_width]u16 = undefined;
        const ref64 = referenceRow(f64, stream_row, hc, cfg, branch_row, &collapsed64, &y64);
        var collapsed32: [width]u16 = undefined;
        var y32: [stream_width]u16 = undefined;
        const ref32 = referenceRow(f32, stream_row, hc, cfg, branch_row, &collapsed32, &y32);
        inv_64[r] = ref64.inv;
        inv_32[r] = ref32.inv;
        @memcpy(proj_64[r * mix_count ..][0..mix_count], &ref64.proj);
        @memcpy(proj_32[r * mix_count ..][0..mix_count], &ref32.proj);
        @memcpy(pre_64[r * 4 ..][0..4], &ref64.pre);
        @memcpy(pre_32[r * 4 ..][0..4], &ref32.pre);
        @memcpy(post_64[r * 4 ..][0..4], &ref64.post);
        @memcpy(post_32[r * 4 ..][0..4], &ref32.post);
        @memcpy(comb_64[r * 16 ..][0..16], &ref64.comb);
        @memcpy(comb_32[r * 16 ..][0..16], &ref32.comb);
        @memcpy(collapsed_64[r * width ..][0..width], &collapsed64);
        @memcpy(collapsed_32[r * width ..][0..width], &collapsed32);
        @memcpy(y_64[r * stream_width ..][0..stream_width], &y64);
        @memcpy(y_32[r * stream_width ..][0..stream_width], &y32);
    }

    const metrics = [_]StageMetric{
        .{ .name = "inverse", .metal = deltaF32VsF64(inv_m, inv_64), .f32ref = deltaF32VsF64(inv_32, inv_64) },
        .{ .name = "projection", .metal = deltaF32VsF64(proj_m, proj_64), .f32ref = deltaF32VsF64(proj_32, proj_64) },
        .{ .name = "pre", .metal = deltaF32VsF64(pre_m, pre_64), .f32ref = deltaF32VsF64(pre_32, pre_64) },
        .{ .name = "post", .metal = deltaF32VsF64(post_m, post_64), .f32ref = deltaF32VsF64(post_32, post_64) },
        .{ .name = "comb", .metal = deltaF32VsF64(comb_m, comb_64), .f32ref = deltaF32VsF64(comb_32, comb_64) },
        .{ .name = "collapse", .bf16 = deltaBf16(collapsed_m, collapsed_64, collapsed_32) },
        .{ .name = "post_expand", .bf16 = deltaBf16(y_m, y_64, y_32) },
    };
    for (metrics) |m| printMetric(case.name, m);

    // Rounding-level gate: Metal must sit at the FP32 source-contract reference's own distance
    // from the FP64 ideal (measured bounds; a real operator defect breaks these by orders).
    for (metrics, 0..) |m, i| {
        if (i < 5) {
            try std.testing.expect(m.metal.nrms < 2e-7);
            try std.testing.expect(m.f32ref.nrms < 2e-7);
        }
    }
    try std.testing.expect(metrics[0].metal.max_ulp < 1.0); // statistic tree matches the wide sum here
    try std.testing.expect(metrics[1].metal.max_ulp < 1.0); // compensated dense dot
    try std.testing.expect(metrics[2].metal.max_ulp < 4.0); // sigmoid affine + exp boundary
    try std.testing.expect(metrics[3].metal.max_ulp < 4.0);
    // comb ULPs inflate on near-cancelled Sinkhorn cells; the absolute error stays at FP32 level.
    try std.testing.expect(metrics[4].metal.max_abs < 5e-7);
    try std.testing.expect(metrics[4].f32ref.nrms < 2e-7);
    try std.testing.expect(metrics[5].bf16.?.unequal_ideal * 4096 <= metrics[5].bf16.?.count + 4096);
    try std.testing.expect(metrics[6].bf16.?.unequal_ideal * 4096 <= metrics[6].bf16.?.count + 4096);
    try std.testing.expect(metrics[5].bf16.?.unequal_f32_ref * 4096 <= metrics[5].bf16.?.count + 4096);
    try std.testing.expect(metrics[6].bf16.?.unequal_f32_ref * 4096 <= metrics[6].bf16.?.count + 4096);
}

fn syntheticWeight(gpa: std.mem.Allocator) ![]f32 {
    var lcg = Lcg{};
    const w = try gpa.alloc(f32, mix_count * stream_width);
    for (w) |*v| v.* = lcg.uniform() * 0.03;
    return w;
}

fn fourCopies(gpa: std.mem.Allocator, row: []const u16) ![]u16 {
    var stream = try gpa.alloc(u16, stream_width);
    for (0..copies) |c| @memcpy(stream[c * width ..][0..width], row);
    return stream;
}

test "mHC fp64 rounding level and Metal deltas on retained streams" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var o = try ops.Pipelines.load(device);
    defer o.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_stream = try decodeU16(gpa, fixture.value.ffn_stream_bf16);
    const ffn_branch = try decodeU16(gpa, fixture.value.ffn_branch_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const token671 = eight_rows[1 * width ..][0..width]; // hc_attention_rms.json provenance: token 671.

    const weight = try syntheticWeight(gpa);

    // Token 671 embedding expanded to four copies; real FFN scale/base; synthetic branch-free case
    // uses the retained branch row. All three sides consume identical inputs.
    {
        const stream = try fourCopies(gpa, token671);
        const branch = try gpa.alloc(u16, width);
        @memcpy(branch, ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "token671_attention", .rows = 1, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    {
        const branch = try gpa.alloc(u16, width);
        @memcpy(branch, ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "token1_ffn", .rows = 1, .stream = ffn_stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    {
        const rows = 8;
        const stream = try gpa.alloc(u16, rows * stream_width);
        for (0..rows) |r| {
            const row = eight_rows[r * width ..][0..width];
            for (0..copies) |c| @memcpy(stream[r * stream_width + c * width ..][0..width], row);
        }
        const branch = try gpa.alloc(u16, rows * width);
        for (0..rows) |r| @memcpy(branch[r * width ..][0..width], ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "eight_teacher_rows", .rows = rows, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    // Wide dynamic range: copy magnitudes 2^24..2^-12 with rotating signs, branch 2^8..2^-4.
    {
        const stream = try gpa.alloc(u16, stream_width);
        const exponents = [_]i32{ 24, 12, 0, -12 };
        for (0..copies) |c| {
            for (0..width) |d| {
                const sign: f32 = if ((d + c) % 2 == 0) 1 else -1;
                stream[c * width + d] = math.b(sign * @as(f32, @floatCast(std.math.pow(f64, 2, @floatFromInt(exponents[c])))));
            }
        }
        const branch = try gpa.alloc(u16, width);
        const branch_exponents = [_]i32{ 8, 4, 0, -4 };
        for (0..width) |d| {
            const sign: f32 = if (d % 2 == 0) 1 else -1;
            branch[d] = math.b(sign * @as(f32, @floatCast(std.math.pow(f64, 2, @floatFromInt(branch_exponents[d % 4])))));
        }
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "wide_dynamic_range", .rows = 1, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
}

// The comptime parameterization must keep the FP32 source contract bitwise; this compares
// math.mixesOf(f32) against the pre-parameterization algorithm verbatim on a retained row.
test "mHC fp32 reference path stays bit-identical under the comptime parameterization" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);
    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const scale = try device.buffer(12, mtl.ResourceOptions.shared);
    defer scale.deinit();
    upload(f32, scale, ffn_scale);
    const base = try device.buffer(mix_count * 4, mtl.ResourceOptions.shared);
    defer base.deinit();
    upload(f32, base, ffn_base);
    var lcg = Lcg{};
    const weight = try device.buffer(mix_count * stream_width * 4, mtl.ResourceOptions.shared);
    defer weight.deinit();
    const weight_host = try gpa.alloc(f32, mix_count * stream_width);
    for (weight_host) |*v| v.* = lcg.uniform() * 0.03;
    upload(f32, weight, weight_host);
    const hc = Hc{
        .projection = .{ .buffer = weight, .offset = 0, .bytes = mix_count * stream_width * 4, .dtype = .f32 },
        .scale = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 },
        .base = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 },
    };
    for (0..8) |r| {
        const row = eight_rows[r * width ..][0..width];
        var stream: [stream_width]u16 = undefined;
        for (0..copies) |c| @memcpy(stream[c * width ..][0..width], row);
        const parameterized = math.mixesOf(f32, &stream, hc, cfg.text_config);
        const legacy = mixesLegacy(&stream, hc, cfg);
        try std.testing.expect(std.meta.eql(parameterized, legacy));
    }
}

fn mixesLegacy(x: []const u16, hc: Hc, c: Config) math.Mix {
    @setFloatMode(.strict);
    var projected: [24]f32 = undefined;
    math.denseOf(f32, hc.projection, x, &projected);
    var squares: f64 = 0;
    for (x) |v| squares += @as(f64, math.f(v)) * math.f(v);
    const mean = math.meanFromWide(squares, x.len);
    const r = math.idealRsqrt(mean + c.text_config.rms_norm_eps);
    var m: math.Mix = undefined;
    for (0..4) |i| {
        m.pre[i] = math.sigmoid(@mulAdd(f32, projected[i] * r, math.value(hc.scale, 0), math.value(hc.base, i))) + c.text_config.hc_eps;
        m.post[i] = 2 * math.sigmoid(@mulAdd(f32, projected[i + 4] * r, math.value(hc.scale, 1), math.value(hc.base, i + 4)));
    }
    for (0..4) |i| {
        var high: f32 = -std.math.inf(f32);
        for (0..4) |j| {
            const at = i * 4 + j;
            m.comb[at] = @mulAdd(f32, projected[at + 8] * r, math.value(hc.scale, 2), math.value(hc.base, at + 8));
            high = @max(high, m.comb[at]);
        }
        for (0..4) |j| {
            m.comb[i * 4 + j] = @exp(m.comb[i * 4 + j] - high);
        }
        const denominator = (m.comb[i * 4] + m.comb[i * 4 + 2]) +
            (m.comb[i * 4 + 1] + m.comb[i * 4 + 3]);
        for (0..4) |j| m.comb[i * 4 + j] = m.comb[i * 4 + j] / denominator + c.text_config.hc_eps;
    }
    for (0..c.text_config.hc_sinkhorn_iters) |iteration| {
        if (iteration != 0) for (0..4) |i| {
            const denominator = ((m.comb[i * 4] + m.comb[i * 4 + 2]) +
                (m.comb[i * 4 + 1] + m.comb[i * 4 + 3])) + c.text_config.hc_eps;
            for (0..4) |j| m.comb[i * 4 + j] /= denominator;
        };
        for (0..4) |j| {
            const denominator = ((m.comb[j] + m.comb[8 + j]) +
                (m.comb[4 + j] + m.comb[12 + j])) + c.text_config.hc_eps;
            for (0..4) |i| m.comb[i * 4 + j] /= denominator;
        }
    }
    return m;
}

// Direct split-stage isolation: identical projected mixes and inverse on both sides, so every
// delta is a unary (sigmoid/exp) or reduction boundary, not a projection difference.
test "mHC split unary boundaries against the FP64 ideal on retained arguments" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const rows = 7 + 65; // retained arguments plus every integer exponent input -32..32
    var b = Buffers{ .device = device };
    defer b.deinit();
    const mixes_host = try gpa.alloc(f32, rows * mix_count);
    @memset(mixes_host, 0);
    // Rows 0..3: the 16 retained unary arguments rotated through pre, post and comb positions.
    for (0..4) |i| {
        for (0..4) |j| {
            mixes_host[i * mix_count + j] = @bitCast(retained_exp_argument_bits[(4 * i + j) % 16]);
            mixes_host[i * mix_count + 4 + j] = @bitCast(retained_exp_argument_bits[(4 * i + j) % 16]);
        }
        for (0..16) |q| mixes_host[i * mix_count + 8 + q] = @bitCast(retained_exp_argument_bits[(i + q) % 16]);
    }
    // Row 4: the documented controlling exponent input as a post argument (exp(+1.439...) path).
    mixes_host[4 * mix_count + 5] = -controlling_exponent_input;
    // Row 5: comb affine arguments spanning the exponent range.
    mixes_host[5 * mix_count + 8] = -32;
    mixes_host[5 * mix_count + 9] = -16;
    mixes_host[5 * mix_count + 10] = 16;
    mixes_host[5 * mix_count + 11] = 32;
    // Row 6: the same span through the sigmoid pre/post arguments.
    const span = [_]f32{ -32, -16, 16, 32 };
    for (span, 0..) |v, j| {
        mixes_host[6 * mix_count + j] = v;
        mixes_host[6 * mix_count + 4 + j] = v;
    }

    // Rows 7..69: one integer exponent input per row at a pre, post and comb position each.
    for (0..65) |i| {
        const r = 7 + i;
        const v: f32 = @floatFromInt(@as(i32, @intCast(i)) - 32);
        mixes_host[r * mix_count] = v;
        mixes_host[r * mix_count + 4] = v;
        mixes_host[r * mix_count + 8] = v;
    }

    const mixes = try b.get(f32, rows * mix_count);
    upload(f32, mixes, mixes_host);
    const inv = try b.get(f32, rows);
    @memset(inv.slice(f32, rows), 1.0);
    const scale = try b.get(f32, 3);
    @memset(scale.slice(f32, 3), 1.0);
    const base = try b.get(f32, mix_count);
    const pre = try b.get(f32, rows * 4);
    const post = try b.get(f32, rows * 4);
    const comb = try b.get(f32, rows * 16);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeSplit(e, &p, view(mixes), view(inv), view(scale), view(base), view(pre), view(post), view(comb), .{ .rows = rows, .eps = cfg.text_config.hc_eps });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    const scale_tensor: Tensor = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 };
    const base_tensor: Tensor = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 };
    const hc = Hc{ .projection = base_tensor, .scale = scale_tensor, .base = base_tensor };

    var max_ulp_metal: f64 = 0;
    var max_ulp_f32: f64 = 0;
    var worst_name: []const u8 = "";
    const stages = [_]struct { name: []const u8, offset: usize, len: usize }{
        .{ .name = "pre", .offset = 0, .len = 4 },
        .{ .name = "post", .offset = 4, .len = 4 },
        .{ .name = "comb", .offset = 8, .len = 16 },
    };
    for (stages) |stage| {
        for (0..rows) |r| {
            const m: []const f32 = mixes_host[r * mix_count ..][0..mix_count];
            var m64: [mix_count]f64 = undefined;
            for (m, 0..) |v, i| m64[i] = v;
            const mix64 = math.splitOf(f64, &m64, 1.0, hc, cfg.text_config);
            const mix32 = math.splitOf(f32, m, 1.0, hc, cfg.text_config);
            const src64: []const f64 = if (std.mem.eql(u8, stage.name, "pre")) &mix64.pre else if (std.mem.eql(u8, stage.name, "post")) &mix64.post else &mix64.comb;
            const src32: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) &mix32.pre else if (std.mem.eql(u8, stage.name, "post")) &mix32.post else &mix32.comb;
            const metal: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) pre.slice(f32, rows * 4)[r * 4 ..][0..4] else if (std.mem.eql(u8, stage.name, "post")) post.slice(f32, rows * 4)[r * 4 ..][0..4] else comb.slice(f32, rows * 16)[r * 16 ..][0..16];
            const d = deltaF32VsF64(metal[0..stage.len], src64[0..stage.len]);
            const d32 = deltaF32VsF64(src32[0..stage.len], src64[0..stage.len]);
            if (d.max_ulp > max_ulp_metal) {
                max_ulp_metal = d.max_ulp;
                worst_name = stage.name;
            }
            if (d32.max_ulp > max_ulp_f32) max_ulp_f32 = d32.max_ulp;
            std.debug.print("HC64 {{\"case\":\"split_args\",\"stage\":\"{s}\",\"row\":{d},\"nrms_metal_vs_f64\":{d},\"max_ulp_metal_vs_f64\":{d:.4},\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ stage.name, r, d.nrms, d.max_ulp, d32.max_ulp });
        }
    }
    std.debug.print("HC64 {{\"case\":\"split_args\",\"summary\":true,\"max_ulp_metal_vs_f64\":{d:.4},\"worst_stage\":\"{s}\",\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ max_ulp_metal, worst_name, max_ulp_f32 });
    // The kernel may exceed the FP32 reference by at most the documented Metal exp slack (1 ULP).
    // Absolute ULPs inflate on near-cancelled Sinkhorn cells; NRMS stays below 1e-7 everywhere.
    try std.testing.expect(max_ulp_metal <= max_ulp_f32 + 2.0);
    try std.testing.expect(max_ulp_metal < 48.0);
}

// The retained cancellation fixtures: the kernel must keep the source FP32 accumulation
// contract exactly; the FP64 ideal column documents the by-design half-ULP divergence.
test "mHC collapse and post keep source FP32 accumulation under cancellation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    var b = Buffers{ .device = device };
    defer b.deinit();
    // Collapse fixture: stream {3,3,0,0} (width 1), previous_pre {-1, 1+2^-23, 0, 0}.
    const stream_a = try b.get(u16, 4);
    const values_a = [_]u16{ math.b(3), math.b(3), math.b(0), math.b(0) };
    upload(u16, stream_a, &values_a);
    const previous_a = try b.get(f32, 4);
    const coeff = 1 + 0x1p-23;
    const previous_values = [_]f32{ -1, coeff, 0, 0 };
    upload(f32, previous_a, &previous_values);
    const collapsed = try b.get(u16, 1);

    // Post fixture: stream {1,1,1,1}, branch {1}, post {-1,0,0,0},
    // comb column 0 = {1, 2^-24, 2^-24, 0} (two half-ULP terms after 1.0).
    const stream_b = try b.get(u16, 4);
    const values_b = [_]u16{ math.b(1), math.b(1), math.b(1), math.b(1) };
    upload(u16, stream_b, &values_b);
    const branch_b = try b.get(u16, 1);
    const branch_values = [_]u16{math.b(1)};
    upload(u16, branch_b, &branch_values);
    const post_b = try b.get(f32, 4);
    const post_values = [_]f32{ -1, 0, 0, 0 };
    upload(f32, post_b, &post_values);
    const comb_b = try b.get(f32, 16);
    comb_b.slice(f32, 16)[0] = 1;
    comb_b.slice(f32, 16)[4] = 0x1p-24;
    comb_b.slice(f32, 16)[8] = 0x1p-24;
    const y_b = try b.get(u16, 4);

    // Branch-product fixture: stream {3,0,0,0}, post {1+2^-23,...}, comb[0] = -1, branch {3}.
    const stream_c = try b.get(u16, 4);
    const values_c = [_]u16{ math.b(3), math.b(0), math.b(0), math.b(0) };
    upload(u16, stream_c, &values_c);
    const branch_c = try b.get(u16, 1);
    const branch_values_c = [_]u16{math.b(3)};
    upload(u16, branch_c, &branch_values_c);
    const post_c = try b.get(f32, 4);
    const post_values_c = [_]f32{ coeff, 0, 0, 0 };
    upload(f32, post_c, &post_values_c);
    const comb_c = try b.get(f32, 16);
    comb_c.slice(f32, 16)[0] = -1;
    const y_c = try b.get(u16, 4);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeCollapse(e, &p, view(stream_a), view(previous_a), view(collapsed), .{ .rows = 1, .width = 1, .stream_stride = 4, .y_stride = 1 });
    mhc.encodePost(e, &p, view(branch_b), view(stream_b), view(post_b), view(comb_b), view(y_b), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    mhc.encodePost(e, &p, view(branch_c), view(stream_c), view(post_c), view(comb_c), view(y_c), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    // Source-contract expectations (math.zig retained boundaries).
    try std.testing.expectEqual(@as(u16, 0x3500), collapsed.slice(u16, 1)[0]);
    try std.testing.expectEqual(@as(u16, 0), y_b.slice(u16, 4)[0]);
    try std.testing.expectEqual(@as(u16, 0x3500), y_c.slice(u16, 4)[0]);

    // Reference columns for the same fixtures.
    var collapsed32: [1]u16 = undefined;
    var collapsed64: [1]u16 = undefined;
    math.collapseOf(f32, &values_a, .{ -1, coeff, 0, 0 }, &collapsed32);
    math.collapseOf(f64, &values_a, .{ -1, @as(f64, coeff), 0, 0 }, &collapsed64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"collapse_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ collapsed.slice(u16, 1)[0], collapsed32[0], collapsed64[0] });
    try std.testing.expectEqual(collapsed32[0], collapsed.slice(u16, 1)[0]);

    const mix_b = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var y32: [4]u16 = undefined;
    var h32 = values_b;
    math.expandOf(f32, &h32, &branch_values, mix_b);
    @memcpy(&y32, &h32);
    const mix_b64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var h64 = values_b;
    math.expandOf(f64, &h64, &branch_values, mix_b64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"post_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_b.slice(u16, 4)[0], y32[0], h64[0] });
    try std.testing.expectEqual(y32[0], y_b.slice(u16, 4)[0]);

    const mix_c = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ coeff, 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc32 = values_c;
    math.expandOf(f32, &hc32, &branch_values_c, mix_c);
    const mix_c64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ @as(f64, coeff), 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc64 = values_c;
    math.expandOf(f64, &hc64, &branch_values_c, mix_c64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"branch_product\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_c.slice(u16, 4)[0], hc32[0], hc64[0] });
    try std.testing.expectEqual(hc32[0], y_c.slice(u16, 4)[0]);
    _ = cfg;
}

// Ideal FP64 stage propagation used by the amplification measurement: statistic, dense, split,
// collapse and post-expansion, with the pre-rounding FP64 outputs kept for analog-gain figures.
fn stageF64(stream: []const u16, hc: Hc, cfg: Config, branch: []const u16, y_pre: []f64, y_bf: *[stream_width]u16, coll_pre: []f64, inv_out: *f64, comb_out: *[16]f64) void {
    var proj: [mix_count]f64 = undefined;
    math.denseOf(f64, hc.projection, stream, &proj);
    const inv = math.inverseOf(f64, stream, cfg.text_config.rms_norm_eps);
    const mix = math.splitOf(f64, &proj, inv, hc, cfg.text_config);
    inv_out.* = inv;
    comb_out.* = mix.comb;
    for (0..width) |d| {
        var s: f64 = 0;
        for (0..copies) |c| s += mix.pre[c] * @as(f64, math.f(stream[c * width + d]));
        coll_pre[d] = s;
    }
    for (0..width) |d| {
        for (0..copies) |k| {
            var acc: f64 = mix.post[k] * @as(f64, math.f(branch[d]));
            for (0..copies) |j| acc += mix.comb[j * 4 + k] * @as(f64, math.f(stream[j * width + d]));
            y_pre[k * width + d] = acc;
        }
    }
    var h: [stream_width]u16 = undefined;
    @memcpy(&h, stream);
    math.expandOf(f64, &h, branch, mix);
    @memcpy(y_bf, &h);
}

fn perturbOneUlp(x: u16) u16 {
    return if ((x & 0x8000) == 0) x + 1 else x - 1;
}

test "mHC 1-ULP input amplification through the hyperconnection mix" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var o = try ops.Pipelines.load(device);
    defer o.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_stream = try decodeU16(gpa, fixture.value.ffn_stream_bf16);
    const ffn_branch = try decodeU16(gpa, fixture.value.ffn_branch_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const weight = try syntheticWeight(gpa);

    const scale = try device.buffer(12, mtl.ResourceOptions.shared);
    defer scale.deinit();
    upload(f32, scale, ffn_scale);
    const base = try device.buffer(mix_count * 4, mtl.ResourceOptions.shared);
    defer base.deinit();
    upload(f32, base, ffn_base);
    const weight_buffer = try device.buffer(mix_count * stream_width * 4, mtl.ResourceOptions.shared);
    defer weight_buffer.deinit();
    upload(f32, weight_buffer, weight);
    const hc = Hc{
        .projection = .{ .buffer = weight_buffer, .offset = 0, .bytes = mix_count * stream_width * 4, .dtype = .f32 },
        .scale = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 },
        .base = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 },
    };

    const columns = 64;
    const column_stride = width / columns;
    const Row = struct { name: []const u8, stream: []u16 };
    var rows: [9]Row = undefined;
    for (0..8) |r| {
        rows[r] = .{ .name = "teacher", .stream = try fourCopies(gpa, eight_rows[r * width ..][0..width]) };
    }
    rows[8] = .{ .name = "ffn_token1", .stream = try gpa.dupe(u16, ffn_stream) };

    const y0 = try gpa.alloc(f64, stream_width);
    const y1 = try gpa.alloc(f64, stream_width);
    var y0_bf: [stream_width]u16 = undefined;
    var y1_bf: [stream_width]u16 = undefined;
    const coll0 = try gpa.alloc(f64, width);
    const coll1 = try gpa.alloc(f64, width);
    var inv0: f64 = undefined;
    var inv1: f64 = undefined;
    var comb0: [16]f64 = undefined;
    var comb1: [16]f64 = undefined;

    for (rows, 0..) |row, row_index| {
        stageF64(row.stream, hc, cfg, ffn_branch, y0, &y0_bf, coll0, &inv0, &comb0);
        var sum_y2: f64 = 0;
        for (y0) |v| sum_y2 += v * v;
        var sum_x2: f64 = 0;
        for (row.stream) |v| sum_x2 += blk: {
            const f: f64 = @floatCast(math.f(v));
            break :blk f * f;
        };
        var sum_c2: f64 = 0;
        for (coll0) |v| sum_c2 += v * v;
        var sum_b2: f64 = 0;
        for (y0_bf) |v| {
            const f: f64 = @floatCast(math.f(v));
            sum_b2 += f * f;
        }

        var amp_sum: f64 = 0;
        var amp_max: f64 = 0;
        var amp_n: usize = 0;
        var amp_c_sum: f64 = 0;
        var amp_c_max: f64 = 0;
        var amp_b_sum: f64 = 0;
        var amp_b_max: f64 = 0;
        var changed_any: usize = 0;
        var changed_cells: usize = 0;
        var inv_rel_sum: f64 = 0;
        var comb_sq: f64 = 0;
        var comb_n: usize = 0;

        for (0..copies) |j| {
            for (0..columns) |k| {
                const col = k * column_stride;
                const idx = j * width + col;
                const orig = row.stream[idx];
                if (math.f(orig) == 0) continue;
                const perturbed = perturbOneUlp(orig);
                const dx = @as(f64, @floatCast(math.f(perturbed))) - @as(f64, @floatCast(math.f(orig)));
                row.stream[idx] = perturbed;
                stageF64(row.stream, hc, cfg, ffn_branch, y1, &y1_bf, coll1, &inv1, &comb1);
                row.stream[idx] = orig;

                var dy2: f64 = 0;
                for (y0, y1) |a, b| {
                    const d = a - b;
                    dy2 += d * d;
                }
                const amp = @sqrt(dy2 / sum_y2) * @sqrt(sum_x2) / @abs(dx);
                amp_sum += amp;
                if (amp > amp_max) amp_max = amp;
                amp_n += 1;

                var dc2: f64 = 0;
                for (coll0, coll1) |a, b| {
                    const d = a - b;
                    dc2 += d * d;
                }
                const amp_c = @sqrt(dc2 / sum_c2) * @sqrt(sum_x2) / @abs(dx);
                amp_c_sum += amp_c;
                if (amp_c > amp_c_max) amp_c_max = amp_c;

                var changed: usize = 0;
                var dyb2: f64 = 0;
                for (y0_bf, y1_bf) |a, b| {
                    if (a != b) changed += 1;
                    const av: f64 = @floatCast(math.f(a));
                    const bv: f64 = @floatCast(math.f(b));
                    const d = av - bv;
                    dyb2 += d * d;
                }
                const amp_b = @sqrt(dyb2 / sum_b2) * @sqrt(sum_x2) / @abs(dx);
                amp_b_sum += amp_b;
                if (amp_b > amp_b_max) amp_b_max = amp_b;
                if (changed > 0) {
                    changed_any += 1;
                    changed_cells += changed;
                }
                inv_rel_sum += @abs(inv1 - inv0) / inv0;
                var comb_d2: f64 = 0;
                var comb_m2: f64 = 0;
                for (comb0, comb1) |a, b| {
                    const d = a - b;
                    comb_d2 += d * d;
                    comb_m2 += b * b;
                }
                comb_sq += @sqrt(comb_d2 / comb_m2);
                comb_n += 1;
            }
        }
        std.debug.print("HC64 {{\"amplification\":{{\"row\":{d},\"row_name\":\"{s}\",\"perturbations\":{d},\"amp_mean\":{d},\"amp_max\":{d},\"amp_collapse_mean\":{d},\"amp_collapse_max\":{d},\"amp_bf16_mean\":{d},\"amp_bf16_max\":{d},\"bf16_changed_fraction\":{d},\"mean_changed_cells\":{d},\"inv_rel_mean\":{d},\"comb_nrms_mean\":{d}}}}}\n", .{ row_index, row.name, amp_n, amp_sum / @as(f64, @floatFromInt(amp_n)), amp_max, amp_c_sum / @as(f64, @floatFromInt(amp_n)), amp_c_max, amp_b_sum / @as(f64, @floatFromInt(amp_n)), amp_b_max, changed_any, @as(f64, @floatFromInt(changed_cells)) / @as(f64, @floatFromInt(amp_n)), inv_rel_sum / @as(f64, @floatFromInt(amp_n)), comb_sq / @as(f64, @floatFromInt(comb_n)) });
        // The mix attenuates a single-ULP input in FP64 (analog gain < 1) and stays near one
        // BF16-visible ULP of growth; the collapse gain is pinned near one half.
        try std.testing.expect(amp_sum / @as(f64, @floatFromInt(amp_n)) < 1.0);
        try std.testing.expect(amp_max < 1.0);
        try std.testing.expect(amp_c_max < 2.5);
        try std.testing.expect(amp_b_sum / @as(f64, @floatFromInt(amp_n)) < 3.0);
        try std.testing.expect(inv_rel_sum / @as(f64, @floatFromInt(amp_n)) < 1e-5);
    }

    // Metal confirmation on the first teacher row: batch the base row plus 16 stream-cell
    // perturbations through the real kernels and compare against the FP64 analog gain.
    {
        const metal_perturbations = 16;
        const batch = metal_perturbations + 1;
        var batch_stream = try gpa.alloc(u16, batch * stream_width);
        var cells: [metal_perturbations]usize = undefined;
        var dxs: [metal_perturbations]f64 = undefined;
        var filled: usize = 0;
        @memcpy(batch_stream[0..stream_width], rows[0].stream);
        outer: for (0..copies) |j| {
            for (0..columns) |k| {
                const idx = j * width + k * column_stride;
                const orig = rows[0].stream[idx];
                if (math.f(orig) == 0) continue;
                cells[filled] = idx;
                dxs[filled] = @as(f64, @floatCast(math.f(perturbOneUlp(orig)))) - @as(f64, @floatCast(math.f(orig)));
                filled += 1;
                if (filled == metal_perturbations) break :outer;
            }
        }
        for (0..filled) |i| {
            @memcpy(batch_stream[(i + 1) * stream_width ..][0..stream_width], rows[0].stream);
            batch_stream[(i + 1) * stream_width + cells[i]] = perturbOneUlp(rows[0].stream[cells[i]]);
        }
        const stream_buf = try device.buffer(batch * stream_width * 2, mtl.ResourceOptions.shared);
        defer stream_buf.deinit();
        upload(u16, stream_buf, batch_stream);
        const inv_buf = try device.buffer(batch * 4, mtl.ResourceOptions.shared);
        defer inv_buf.deinit();
        const mixes_buf = try device.buffer(batch * mix_count * 4, mtl.ResourceOptions.shared);
        defer mixes_buf.deinit();
        const pre_buf = try device.buffer(batch * 4 * 4, mtl.ResourceOptions.shared);
        defer pre_buf.deinit();
        const post_buf = try device.buffer(batch * 4 * 4, mtl.ResourceOptions.shared);
        defer post_buf.deinit();
        const comb_buf = try device.buffer(batch * 16 * 4, mtl.ResourceOptions.shared);
        defer comb_buf.deinit();
        const collapsed_buf = try device.buffer(batch * width * 2, mtl.ResourceOptions.shared);
        defer collapsed_buf.deinit();
        const branch_buf = try device.buffer(batch * width * 2, mtl.ResourceOptions.shared);
        defer branch_buf.deinit();
        for (0..batch) |r| @memcpy(branch_buf.slice(u16, batch * width)[r * width ..][0..width], ffn_branch);
        const y_buf = try device.buffer(batch * stream_width * 2, mtl.ResourceOptions.shared);
        defer y_buf.deinit();
        try runStage(queue, &p, &o, @intCast(batch), cfg, view(stream_buf), view(weight_buffer), view(inv_buf), view(mixes_buf), view(scale), view(base), view(pre_buf), view(post_buf), view(comb_buf), view(collapsed_buf), view(branch_buf), view(y_buf));

        const y_metal = y_buf.slice(u16, batch * stream_width);
        var nan_count: usize = 0;
        var first_nan: usize = 0;
        for (y_metal[0..stream_width], 0..) |v, i| {
            const f = math.f(v);
            if (std.math.isNan(f)) {
                nan_count += 1;
                if (nan_count == 1) first_nan = i;
            }
        }
        var sum_x2_row0: f64 = 0;
        for (rows[0].stream) |v| {
            const f: f64 = @floatCast(math.f(v));
            sum_x2_row0 += f * f;
        }
        var sum_base2: f64 = 0;
        for (0..stream_width) |i| {
            const f: f64 = @floatCast(math.f(y_metal[i]));
            sum_base2 += f * f;
        }
        var amp_sum: f64 = 0;
        var changed_total: usize = 0;
        for (0..filled) |i| {
            var dy2: f64 = 0;
            var changed: usize = 0;
            for (0..stream_width) |c| {
                const a: f64 = @floatCast(math.f(y_metal[(i + 1) * stream_width + c]));
                const b: f64 = @floatCast(math.f(y_metal[c]));
                const d = a - b;
                if (d != 0) changed += 1;
                dy2 += d * d;
            }
            changed_total += changed;
            amp_sum += @sqrt(dy2 / sum_base2) * @sqrt(sum_x2_row0) / @abs(dxs[i]);
        }
        std.debug.print("HC64 {{\"amplification\":{{\"row\":0,\"metal\":true,\"perturbations\":{d},\"amp_mean\":{d},\"mean_changed_cells\":{d}}}}}\n", .{ filled, amp_sum / @as(f64, @floatFromInt(filled)), @as(f64, @floatFromInt(changed_total)) / @as(f64, @floatFromInt(filled)) });
        const metal_amp = amp_sum / @as(f64, @floatFromInt(filled));
        try std.testing.expect(metal_amp > 0.5 and metal_amp < 3.0);
    }
}
