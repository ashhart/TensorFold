//! Single-position retained-input attention isolation.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const projection = @import("projection.zig");
const math = @import("math.zig");
const attention = tf.deepseek_v41.attention;
const Model = tf.deepseek_v41.Model;
const diagnostic = @import("operator_diagnostic.zig");
const diagnosticRow = diagnostic.diagnosticRow;
const attentionMetric = diagnostic.attentionMetric;
const attention_math = @import("attention_math.zig");
const AttentionStage = diagnostic.AttentionStage;
const AttentionCapture = diagnostic.AttentionCapture;
const attention_elements = diagnostic.attention_elements;
const attentionProjection = diagnostic.attentionProjection;
const hyperconnections = @import("operator_hc.zig").hyperconnections;

/// Local isolation receipt, not a replacement for the full eight-token oracle or its 1% gate.
pub fn runAttention(gpa: std.mem.Allocator, io: std.Io, model: *Model, snapshot: []const u8, dir: []const u8, layer: usize) !void {
    const blob = try std.Io.Dir.cwd().readFileAlloc(io, snapshot, gpa, .limited(8 << 20));
    defer gpa.free(blob);
    const cfg = model.cfg;
    if (layer >= model.weights.layers.len) return error.InvalidAttentionLayer;
    const snapshot_offset = layer * 2 * 61488;
    if (blob.len < snapshot_offset + 40960 + 10240) return error.InvalidAttentionSnapshot;
    try model.reset();
    const input = model.workspace.view(.normalized);
    @memcpy(input.buffer.contents()[input.offset..][0..10240], blob[snapshot_offset + 40960 ..][0..10240]);
    const input_row = @as([*]const u16, @ptrCast(@alignCast(input.buffer.contents() + input.offset)))[0..5120];
    const native = try model.device.buffer(attention_elements * 2, mtl.ResourceOptions.shared);
    defer native.deinit();
    const reference_rows = try gpa.alloc(u16, attention_elements);
    defer gpa.free(reference_rows);
    var capture = AttentionCapture{ .model = model, .native = native, .reference = reference_rows };
    model.residency.add(native);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        model.residency.remove(native);
        model.residency.commit();
    }
    var scratch = try projection.Scratch.init(gpa, 8192, 32768);
    defer scratch.deinit();
    var state = try attention_math.State.init(gpa, 1);
    defer state.deinit();
    var output: [5120]u16 = undefined;
    // Replay prerequisite normalized layers; complete each GPU command before reusing the CPU input buffer.
    for (0..layer) |prior| {
        const prior_offset = prior * 2 * 61488 + 40960;
        @memcpy(input.buffer.contents()[input.offset..][0..10240], blob[prior_offset..][0..10240]);
        const warm_cb = model.queue.commandBufferUnretained();
        const warm_enc = warm_cb.compute(.serial);
        var warm_encoding = true;
        errdefer if (warm_encoding) warm_enc.end();
        for (model.checkpoint.shards.items) |shard| warm_enc.useResource(shard.buffer, mtl.ResourceUsage.read);
        try attention.encode(&model.pipelines.attention, warm_enc, &model.weights.layers[prior].attn, &model.cache, &model.attention_workspace, input, model.workspace.view(.branch), &model.pipelines.projection, model.workspace.projection(), @intCast(prior), 1, cfg);
        warm_enc.end();
        warm_encoding = false;
        warm_cb.commit();
        warm_cb.wait();
        if (warm_cb.failure() != null) return error.OperatorGpuFailure;
        try state.apply(model.weights.layers[prior].attn, prior, input_row, 0, cfg, &output, &scratch);
    }
    @memcpy(input.buffer.contents()[input.offset..][0..10240], blob[snapshot_offset + 40960 ..][0..10240]);
    const weights = &model.weights.layers[layer].attn;
    const cb = model.queue.commandBufferUnretained();
    const enc = cb.compute(.serial);
    var encoding = true;
    errdefer if (encoding) enc.end();
    for (model.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
    try attention.encodeDiagnostic(&model.pipelines.attention, enc, weights, &model.cache, &model.attention_workspace, input, model.workspace.view(.branch), &model.pipelines.projection, model.workspace.projection(), @intCast(layer), 1, cfg, true, .{ .ctx = &capture, .capture = AttentionCapture.gpu });
    enc.end();
    encoding = false;
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.OperatorGpuFailure;
    const sparse_snapshot = try gpa.alloc(u16, 32768);
    defer gpa.free(sparse_snapshot);
    state.sparse_snapshot = sparse_snapshot;
    state.observer = .{ .ctx = &capture, .capture = AttentionCapture.cpu };
    try state.apply(weights.*, layer, input_row, 0, cfg, &output, &scratch);
    state.observer = null;
    state.sparse_snapshot = null;

    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    var count: usize = 0;
    var header: [512]u8 = undefined;
    try report.appendSlice(gpa, try std.fmt.bufPrint(&header, "{{\"mode\":\"attention-sameinput\",\"token\":0,\"layer\":{d},\"input_offset_bytes\":{d},\"input_length_bytes\":10240,\"identical_normalized_input\":true,\"retained_normalized_prefix_layers\":{d},\"compression_ratio\":{d},\"reference_dot\":\"independent_EXL3_FP64\",\"qualification_gate_applied\":false,\"cases\":[\n", .{ layer, snapshot_offset + 40960, layer, cfg.compress_ratios[layer] }));
    inline for (std.enums.values(AttentionStage)) |stage|
        try attentionMetric(u16, gpa, &report, &count, @tagName(stage), "same_normalized_input_chain", capture.row(stage), capture.cpuRow(stage));

    const expected = try gpa.alloc(f32, 32768);
    defer gpa.free(expected);
    try attentionProjection(gpa, model, weights.wq_a, input, "wq_a", &scratch, expected, &report, &count);
    try attentionProjection(gpa, model, weights.wq_b, capture.view(.qr_norm), "wq_b", &scratch, expected, &report, &count);
    try attentionProjection(gpa, model, weights.wkv, input, "wkv", &scratch, expected, &report, &count);
    for (weights.wo_a, 0..) |matrix, group| {
        var name: [32]u8 = undefined;
        try attentionProjection(gpa, model, matrix, capture.view(.o).shifted(group * 4096 * 2), try std.fmt.bufPrint(&name, "wo_a_group{d}", .{group}), &scratch, expected, &report, &count);
    }
    try attentionProjection(gpa, model, weights.wo_b, capture.view(.grouped), "wo_b", &scratch, expected, &report, &count);

    var same_input: [32768]u16 = undefined;
    math.rms(capture.row(.qr_projection), weights.q_norm, cfg.eps, same_input[0..1280]);
    try attentionMetric(u16, gpa, &report, &count, "qr_norm", "same_native_input_reference_rms", capture.row(.qr_norm), same_input[0..1280]);
    math.rms(capture.row(.kv_projection), weights.kv_norm, cfg.eps, same_input[0..512]);
    try attentionMetric(u16, gpa, &report, &count, "kv_norm", "same_native_input_reference_rms", capture.row(.kv_norm), same_input[0..512]);
    @memcpy(same_input[0..512], capture.row(.kv_rope));
    math.quantize(same_input[0..512], .window);
    try attentionMetric(u16, gpa, &report, &count, "kv_quantized", "same_native_input_reference_quantize", capture.row(.kv_quantized), same_input[0..512]);
    if (cfg.compress_ratios[layer] == 1) {
        const source = try model.cache.source(@intCast(layer), 1);
        const compressed = model.workspace.view(.collapsed);
        const unpack_cb = model.queue.commandBufferUnretained();
        const unpack_enc = unpack_cb.compute(.serial);
        var unpack_encoding = true;
        errdefer if (unpack_encoding) unpack_enc.end();
        try model.pipelines.attention.cache.encodeUnpack(unpack_enc, source.main, compressed, .{ .rows = 1, .width = 512, .stride = 512, .packed_stride = 288, .block = 16, .kind = .main_fp4 });
        unpack_enc.end();
        unpack_encoding = false;
        unpack_cb.commit();
        unpack_cb.wait();
        if (unpack_cb.failure() != null) return error.OperatorGpuFailure;
        const actual_compressed = diagnosticRow(u16, compressed, 512);
        const reference_compressed = state.sources[state.active_owner * state.capacity * 640 ..][0..512];
        try attentionMetric(u16, gpa, &report, &count, "compressed_kv", "same_retained_normalized_prefix", actual_compressed, reference_compressed);
        var sinks: [64]f32 = undefined;
        for (&sinks, 0..) |*v, head| v.* = math.value(weights.sink, head);
        inline for (.{ "attention-compressed-native.bf16", "attention-compressed-reference.bf16", "attention-sinks.f32" }, .{ std.mem.sliceAsBytes(actual_compressed), std.mem.sliceAsBytes(reference_compressed), std.mem.sliceAsBytes(&sinks) }) |name, data| {
            const path = try std.fs.path.join(gpa, &.{ dir, name });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        }
        @memcpy(reference_compressed, actual_compressed);
        const indices = [_]i32{ 0, 128 };
        for (0..64) |head|
            state.sparseHead(capture.row(.q)[head * 512 ..][0..512], capture.row(.kv_quantized), &indices, sinks[head], same_input[head * 512 ..][0..512]);
        try attentionMetric(u16, gpa, &report, &count, "sparse_o", "same_native_q_window_and_compressed_kv_reference_sparse", capture.row(.sparse_o), &same_input);
    }
    if (cfg.compress_ratios[layer] == 0) {
        const indices = [_]i32{0};
        for (0..64) |head|
            state.sparseHead(capture.row(.q)[head * 512 ..][0..512], capture.row(.kv_quantized), &indices, math.value(weights.sink, head), same_input[head * 512 ..][0..512]);
        try attentionMetric(u16, gpa, &report, &count, "sparse_o", "same_native_q_and_quantized_kv_reference_sparse", capture.row(.sparse_o), &same_input);
    }
    try hyperconnections(gpa, model, blob[snapshot_offset..], capture.view(.wo_b), &report, &count, layer);
    try report.appendSlice(gpa, "\n]}\n");
    inline for (.{ "attention-sameinput.json", "attention-native.bf16", "attention-reference.bf16", "attention-input.bf16" }, .{ report.items, std.mem.sliceAsBytes(native.slice(u16, attention_elements)), std.mem.sliceAsBytes(reference_rows), std.mem.sliceAsBytes(input_row) }) |name, data| {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
    }
}
