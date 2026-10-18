//! HC64 compares mHC rounding and 1-ULP amplification using retained rows; see docs/reference-precision.md.
pub const std = @import("std");
pub const mtl = @import("metal");
pub const tf = @import("tensorfold");
pub const math = @import("math.zig");

pub const mhc = tf.deepseek_v41.mhc;
pub const ops = tf.deepseek_v41.ops;
pub const View = tf.deepseek_v41.attention.View;
pub const Tensor = tf.checkpoint_metal.Tensor;
pub const Hc = tf.deepseek_v41_weights.Hc;
pub const Config = tf.deepseek_v41_config.Config;

pub const width: usize = 5120;
pub const copies: usize = 4;
pub const stream_width: usize = width * copies;
pub const mix_count: usize = 24;

pub const released_config =
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
pub const retained_exp_argument_bits = [_]u32{
    3217026610, 1081519143, 1049174916, 1078264130, 1073892416, 1078760804, 1083574034, 1069038932,
    1043563522, 1083490981, 1087210444, 1094911558, 3195364771, 1082252091, 1096635047, 1091866698,
};

/// The documented controlling exponent input: Metal FP32 0x4086fa39, ideal 0x4086fa38 (1 ULP).
pub const controlling_exponent_input: f32 = 1.4393715858459473;

pub const fixture_path = "../../src/families/deepseek_v41/fixtures/hc_fp64_rows.json";

pub const Fixture = struct {
    eight_rows_bf16: []const u8,
    ffn_stream_bf16: []const u8,
    ffn_branch_bf16: []const u8,
    ffn_scale_f32: []const u8,
    ffn_base_f32: []const u8,
    provenance: []const u8 = "",
};

pub fn decodeBase64(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    const size = try std.base64.standard.Decoder.calcSizeForSlice(text);
    const out = try gpa.alloc(u8, size);
    errdefer gpa.free(out);
    try std.base64.standard.Decoder.decode(out, text);
    return out;
}

pub fn decodeU16(gpa: std.mem.Allocator, text: []const u8) ![]u16 {
    const bytes = try decodeBase64(gpa, text);
    defer gpa.free(bytes);
    const out = try gpa.alloc(u16, bytes.len / 2);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return out;
}

pub fn decodeF32(gpa: std.mem.Allocator, text: []const u8) ![]f32 {
    const bytes = try decodeBase64(gpa, text);
    defer gpa.free(bytes);
    const out = try gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

pub const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    pub fn next(s: *Lcg) u64 {
        s.state = s.state *% 6364136223846793005 +% 1442695040888963407;
        return s.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    pub fn uniform(s: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(s.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
};

pub fn nextUpF32(x: f32) f32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    const bits: u32 = @bitCast(x);
    if (bits == 0 or bits == 0x80000000) return @bitCast(@as(u32, 1));
    return @bitCast(if (bits >> 31 == 0) bits + 1 else bits - 1);
}

/// The local FP32 spacing at the FP64 reference value: the ULP unit for kernel deltas.
pub fn ulp32At(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f) or std.math.isInf(f)) return @abs(v);
    const a = @abs(f);
    if (a == 0) return 0x1p-149;
    const up = nextUpF32(a);
    if (std.math.isInf(up)) return @as(f64, a) - @as(f64, -nextUpF32(-a));
    return @as(f64, up) - @as(f64, a);
}

pub const Delta = struct {
    nrms: f64 = 0,
    max_abs: f64 = 0,
    max_ulp: f64 = 0,
    worst: usize = 0,
    count: usize = 0,
};

pub fn deltaF32VsF64(actual: []const f32, ref: []const f64) Delta {
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

pub const BfDelta = struct {
    nrms: f64 = 0,
    unequal_ideal: usize = 0,
    unequal_f32_ref: usize = 0,
    count: usize = 0,
    worst: usize = 0,
    worst_gap: f64 = 0,
};

/// BF16 outputs: decoded NRMS against the FP64 ideal cells plus ideal/F32-reference cell counts.
pub fn deltaBf16(actual: []const u16, ideal: []const u16, f32_ref: []const u16) BfDelta {
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

pub fn StageRefOf(comptime T: type) type {
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

pub fn referenceRow(comptime T: type, stream: []const u16, hc: Hc, cfg: Config, branch: []const u16, collapsed: *[width]u16, y: *[stream_width]u16) StageRefOf(T) {
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

pub const Buffers = struct {
    device: mtl.Device,
    items: [24]mtl.Buffer = undefined,
    count: usize = 0,

    pub fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), n), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    pub fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};

pub fn upload(comptime T: type, buffer: mtl.Buffer, data: []const T) void {
    @memcpy(buffer.slice(T, data.len), data);
}

pub fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

pub fn runStage(
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

pub const Case = struct {
    name: []const u8,
    rows: usize,
    stream: []u16,
    weight: []f32,
    branch: []u16,
    scale: []f32,
    base: []f32,
};

pub const StageMetric = struct {
    name: []const u8,
    metal: Delta = .{},
    f32ref: Delta = .{},
    bf16: ?BfDelta = null,
};

pub fn printMetric(case_name: []const u8, m: StageMetric) void {
    if (m.bf16) |bf| {
        std.debug.print("HC64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"elements\":{d},\"nrms_metal_vs_ideal\":{d},\"bf16_unequal_metal_vs_ideal\":{d},\"bf16_unequal_f32ref_vs_ideal\":{d},\"worst_gap\":{d},\"worst_index\":{d}}}\n", .{ case_name, m.name, bf.count, bf.nrms, m.bf16.?.unequal_ideal, m.bf16.?.unequal_f32_ref, m.bf16.?.worst_gap, m.bf16.?.worst });
    } else {
        std.debug.print("HC64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"elements\":{d},\"nrms_metal_vs_f64\":{d},\"max_ulp_metal_vs_f64\":{d:.4},\"max_abs_metal_vs_f64\":{d},\"worst_metal\":{d},\"nrms_f32_vs_f64\":{d},\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ case_name, m.name, m.metal.count, m.metal.nrms, m.metal.max_ulp, m.metal.max_abs, m.metal.worst, m.f32ref.nrms, m.f32ref.max_ulp });
    }
}

/// Full HC stage on identical Metal/FP32/FP64 inputs: statistic, dense, split, collapse and post-expansion.
pub fn runCase(
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

    // Bound Metal error against the FP32 contract's distance from the FP64 ideal; retain existing tolerances.
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

pub fn syntheticWeight(gpa: std.mem.Allocator) ![]f32 {
    var lcg = Lcg{};
    const w = try gpa.alloc(f32, mix_count * stream_width);
    for (w) |*v| v.* = lcg.uniform() * 0.03;
    return w;
}

pub fn fourCopies(gpa: std.mem.Allocator, row: []const u16) ![]u16 {
    var stream = try gpa.alloc(u16, stream_width);
    for (0..copies) |c| @memcpy(stream[c * width ..][0..width], row);
    return stream;
}

pub fn mixesLegacy(x: []const u16, hc: Hc, c: Config) math.Mix {
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

// Propagate the ideal HC stages in FP64, retaining pre-rounding outputs for analog-gain measurements.
pub fn stageF64(stream: []const u16, hc: Hc, cfg: Config, branch: []const u16, y_pre: []f64, y_bf: *[stream_width]u16, coll_pre: []f64, inv_out: *f64, comb_out: *[16]f64) void {
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

pub fn perturbOneUlp(x: u16) u16 {
    return if ((x & 0x8000) == 0) x + 1 else x - 1;
}
