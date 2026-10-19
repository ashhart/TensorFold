//! Retained-input hyperconnection isolation, separate from qualification.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const attention = tf.deepseek_v41.attention;
const Model = tf.deepseek_v41.Model;
const diagnostic = @import("operator_diagnostic.zig");
const diagnosticRow = diagnostic.diagnosticRow;
const attentionMetric = diagnostic.attentionMetric;

pub fn hyperconnections(gpa: std.mem.Allocator, model: *Model, stream_bytes: []const u8, branch: attention.View, report: *std.ArrayList(u8), count: *usize, layer: usize) !void {
    const ws = &model.workspace;
    const hc = model.weights.layers[layer].hc_ffn;
    const cfg = model.cfg.text_config;
    const stream = ws.view(.stream);
    @memcpy(stream.buffer.contents()[stream.offset..][0..40960], stream_bytes[0..40960]);
    var input: [20480]u16 = undefined;
    @memcpy(&input, diagnosticRow(u16, stream, input.len));
    var projected: [24]f32 = undefined;
    math.dense(hc.projection, &input, &projected);
    const expected_mix = math.mixes(&input, hc, cfg);
    var squares: f64 = 0;
    for (input) |v| squares += @as(f64, math.f(v)) * math.f(v);
    const mean = math.meanFromWide(squares, input.len);
    const expected_inv = [_]f32{math.idealRsqrt(mean + cfg.rms_norm_eps)};
    const cb = model.queue.commandBufferUnretained();
    const enc = cb.compute(.serial);
    for (model.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
    tf.deepseek_v41.mhc.encodeRms(enc, &model.pipelines.mhc, stream, ws.view(.inv), .{ .rows = 1, .width = 5120, .stream_stride = 20480, .eps = cfg.rms_norm_eps });
    tf.deepseek_v41.ops.encodeDense(enc, &model.pipelines.ops, stream, .{ .buffer = hc.projection.buffer, .offset = hc.projection.offset }, ws.view(.mixes), .{ .rows = 1, .k = 20480, .n = 24, .x_stride = 20480, .w_stride = 20480, .y_stride = 24, .w_dtype = .f32, .y_dtype = .f32 });
    enc.barrier();
    tf.deepseek_v41.mhc.encodeSplit(enc, &model.pipelines.mhc, ws.view(.mixes), ws.view(.inv), .{ .buffer = hc.scale.buffer, .offset = hc.scale.offset }, .{ .buffer = hc.base.buffer, .offset = hc.base.offset }, ws.view(.attention_pre), ws.view(.post), ws.view(.comb), .{ .rows = 1, .eps = cfg.hc_eps });
    enc.barrier();
    tf.deepseek_v41.mhc.encodeCollapse(enc, &model.pipelines.mhc, stream, ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = 1, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
    enc.barrier();
    tf.deepseek_v41.mhc.encodePost(enc, &model.pipelines.mhc, branch, stream, ws.view(.post), ws.view(.comb), stream, .{ .rows = 1, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.OperatorGpuFailure;
    try attentionMetric(f32, gpa, report, count, "hc_projection", "same_retained_stream_fp64_dot", diagnosticRow(f32, ws.view(.mixes), 24), &projected);
    try attentionMetric(f32, gpa, report, count, "hc_inverse", "same_retained_stream_reference_statistic", diagnosticRow(f32, ws.view(.inv), 1), &expected_inv);
    try attentionMetric(f32, gpa, report, count, "hc_pre", "same_retained_stream_chain", diagnosticRow(f32, ws.view(.attention_pre), 4), &expected_mix.pre);
    try attentionMetric(f32, gpa, report, count, "hc_post", "same_retained_stream_chain", diagnosticRow(f32, ws.view(.post), 4), &expected_mix.post);
    try attentionMetric(f32, gpa, report, count, "hc_comb", "same_retained_stream_chain", diagnosticRow(f32, ws.view(.comb), 16), &expected_mix.comb);
    var native_mix: math.Mix = undefined;
    @memcpy(&native_mix.pre, diagnosticRow(f32, ws.view(.attention_pre), 4));
    @memcpy(&native_mix.post, diagnosticRow(f32, ws.view(.post), 4));
    @memcpy(&native_mix.comb, diagnosticRow(f32, ws.view(.comb), 16));
    var collapsed: [5120]u16 = undefined;
    math.collapse(&input, native_mix.pre, &collapsed);
    try attentionMetric(u16, gpa, report, count, "hc_collapse", "same_retained_stream_and_native_coefficients", diagnosticRow(u16, ws.view(.collapsed), 5120), &collapsed);
    math.expand(&input, diagnosticRow(u16, branch, 5120), native_mix);
    try attentionMetric(u16, gpa, report, count, "hc_expand", "same_retained_stream_branch_and_native_coefficients", diagnosticRow(u16, stream, 20480), &input);
    if (layer + 1 < model.weights.layers.len and stream_bytes.len >= 61488 + 40960) {
        @memcpy(stream.buffer.contents()[stream.offset..][0..40960], stream_bytes[61488..][0..40960]);
        @memcpy(&input, diagnosticRow(u16, stream, input.len));
        const norm = model.weights.layers[layer + 1].attn_norm;
        const next_cb = model.queue.commandBufferUnretained();
        const next_enc = next_cb.compute(.serial);
        next_enc.useResource(norm.buffer, mtl.ResourceUsage.read);
        tf.deepseek_v41.mhc.encodeCollapse(next_enc, &model.pipelines.mhc, stream, ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = 1, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
        next_enc.barrier();
        tf.deepseek_v41.ops.encodeRms(next_enc, &model.pipelines.ops, ws.view(.collapsed), .{ .buffer = norm.buffer, .offset = norm.offset }, ws.view(.branch), .{ .rows = 1, .width = 5120, .x_stride = 5120, .y_stride = 5120, .eps = cfg.rms_norm_eps });
        next_enc.end();
        next_cb.commit();
        next_cb.wait();
        if (next_cb.failure() != null) return error.OperatorGpuFailure;
        math.collapse(&input, native_mix.pre, &collapsed);
        try attentionMetric(u16, gpa, report, count, "next_collapse", "same_retained_ffn_stream_and_native_pre", diagnosticRow(u16, ws.view(.collapsed), 5120), &collapsed);
        math.collapse(&input, expected_mix.pre, &collapsed);
        try attentionMetric(u16, gpa, report, count, "next_collapse", "same_retained_ffn_stream_reference_pre", diagnosticRow(u16, ws.view(.collapsed), 5120), &collapsed);
        math.rms(&collapsed, norm, cfg.rms_norm_eps, &collapsed);
        try attentionMetric(u16, gpa, report, count, "next_normalized", "same_retained_ffn_stream_reference_pre_and_norm", diagnosticRow(u16, ws.view(.branch), 5120), &collapsed);
    }
}

pub fn runHyperconnections(gpa: std.mem.Allocator, io: std.Io, model: *Model, snapshot: []const u8, dir: []const u8, layer: usize) !void {
    if (layer >= model.weights.layers.len) return error.InvalidHyperconnectionLayer;
    const blob = try std.Io.Dir.cwd().readFileAlloc(io, snapshot, gpa, .limited(8 << 20));
    defer gpa.free(blob);
    const offset = layer * 2 * 61488;
    if (blob.len < offset + 2 * 61488) return error.InvalidHyperconnectionSnapshot;
    try model.reset();
    const branch = model.workspace.view(.branch);
    @memcpy(branch.buffer.contents()[branch.offset..][0..10240], blob[offset + 61488 + 51200 ..][0..10240]);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    var count: usize = 0;
    var header: [256]u8 = undefined;
    try report.appendSlice(gpa, try std.fmt.bufPrint(&header, "{{\"mode\":\"hc-sameinput\",\"token\":null,\"token_identity_source\":\"unknown_headerless_snapshot\",\"layer\":{d},\"qualification_gate_applied\":false,\"cases\":[\n", .{layer}));
    try hyperconnections(gpa, model, blob[offset..], branch, &report, &count, layer);
    try report.appendSlice(gpa, "\n]}\n");
    const path = try std.fs.path.join(gpa, &.{ dir, "hc-sameinput.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
}
