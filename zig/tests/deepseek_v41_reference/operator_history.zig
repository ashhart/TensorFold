//! Retained-input attention history and source-layout receipts.
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
const attention_widths = diagnostic.attention_widths;
const attention_elements = diagnostic.attention_elements;
const attentionOffset = diagnostic.attentionOffset;
const attentionProjection = diagnostic.attentionProjection;
const attentionWrite = diagnostic.attentionWrite;

// Source indices address physical ring slots then compressed rows; reference reserves 128 window rows even at zero.
fn attentionHistoryIndices(position: usize, selected: []const i32, compressed_offset: usize, out: *[640]i32) []const i32 {
    const window_columns: usize = if (position == 0) 1 else 128;
    if (position == 0) {
        out[0] = 0;
    } else {
        for (0..128) |slot| {
            const at = (position % 128 + 1 + slot) % 128;
            out[slot] = if (at > position) -1 else @intCast(at);
        }
    }
    for (selected, 0..) |row, slot|
        out[window_columns + slot] = if (row < 0) -1 else @intCast(compressed_offset + @as(usize, @intCast(row)));
    return out[0 .. window_columns + selected.len];
}

/// Replay prerequisite layers at each position; native/reference cursors advance without borrowed intermediates.
pub fn runAttentionHistory(gpa: std.mem.Allocator, io: std.Io, model: *Model, trace_dir: []const u8, dir: []const u8, token: usize, layer: usize) !void {
    if (layer >= model.weights.layers.len) return error.InvalidAttentionLayer;
    if (token >= model.cache.plan.capacity) return error.InvalidAttentionHistory;
    if (token >= attention_math.max_capacity) return error.DiagnosticHistoryTooLarge;
    const prefix_elements = try std.math.mul(usize, token + 1, (layer + 1) * 5120);
    if (prefix_elements > (256 << 20) / @sizeOf(u16)) return error.DiagnosticHistoryTooLarge;
    const cfg = model.cfg;
    const history_count = token + 1;
    const ratio = cfg.compress_ratios.items()[layer];
    const compressed_rows = if (ratio == 0) 0 else history_count / ratio;
    const topk = @min(compressed_rows, cfg.index_topk);
    const window_rows: usize = if (token == 0) 1 else 128;
    const snapshot_bytes = 61488;
    const snapshot_offset = layer * 2 * snapshot_bytes;
    const weights = &model.weights.layers[layer].attn;
    const input = model.workspace.view(.normalized);
    const native = try model.device.buffer(attention_elements * 2, mtl.ResourceOptions.shared);
    defer native.deinit();
    const decoded = try model.device.buffer((128 * 512 + compressed_rows * 640) * 2, mtl.ResourceOptions.shared);
    defer decoded.deinit();
    const reference_rows = try gpa.alloc(u16, attention_elements);
    defer gpa.free(reference_rows);
    const sparse_snapshot = try gpa.alloc(u16, 32768);
    defer gpa.free(sparse_snapshot);
    const normalized_prefix = try gpa.alloc(u16, prefix_elements);
    defer gpa.free(normalized_prefix);
    var retained_branch: [5120]u16 = undefined;
    var capture = AttentionCapture{ .model = model, .native = native, .reference = reference_rows };
    for ([_]mtl.Buffer{ native, decoded }) |buffer| model.residency.add(buffer);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        for ([_]mtl.Buffer{ native, decoded }) |buffer| model.residency.remove(buffer);
        model.residency.commit();
    }
    var scratch = try projection.Scratch.init(gpa, 8192, 32768);
    defer scratch.deinit();
    var state = try attention_math.State.init(gpa, history_count);
    defer state.deinit();
    var output: [5120]u16 = undefined;
    try model.reset();
    for (0..history_count) |position| {
        var name: [64]u8 = undefined;
        const trace_name = try std.fmt.bufPrint(&name, "trace-{d}-gpu.bin", .{position});
        const path = try std.fs.path.join(gpa, &.{ trace_dir, trace_name });
        defer gpa.free(path);
        const blob = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(8 << 20));
        defer gpa.free(blob);
        if (blob.len != model.weights.layers.len * 2 * snapshot_bytes) return error.InvalidAttentionSnapshot;
        if (model.cache.position != position) return error.InvalidAttentionHistoryCursor;
        for (0..layer + 1) |prior| {
            const offset = prior * 2 * snapshot_bytes + 40960;
            const row = normalized_prefix[(position * (layer + 1) + prior) * 5120 ..][0..5120];
            @memcpy(std.mem.sliceAsBytes(row), blob[offset..][0..10240]);
            @memcpy(input.buffer.contents()[input.offset..][0..10240], std.mem.sliceAsBytes(row));
            const target = position == token and prior == layer;
            const cb = model.queue.commandBufferUnretained();
            const enc = cb.compute(.serial);
            var encoding = true;
            errdefer if (encoding) enc.end();
            for (model.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
            if (target) {
                try attention.encodeDiagnostic(&model.pipelines.attention, enc, &model.weights.layers[prior].attn, model.cache, &model.attention_workspace, input, model.workspace.view(.branch), &model.pipelines.projection, model.workspace.projection(), @intCast(prior), 1, cfg, true, .{ .ctx = &capture, .capture = AttentionCapture.gpu });
                @memcpy(std.mem.sliceAsBytes(&retained_branch), blob[snapshot_offset + 51200 ..][0..10240]);
            } else {
                try attention.encode(&model.pipelines.attention, enc, &model.weights.layers[prior].attn, model.cache, &model.attention_workspace, input, model.workspace.view(.branch), &model.pipelines.projection, model.workspace.projection(), @intCast(prior), 1, cfg);
            }
            enc.end();
            encoding = false;
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.OperatorGpuFailure;
            // Wait before the next retained row replaces the native workspace input.
            if (target) {
                state.observer = .{ .ctx = &capture, .capture = AttentionCapture.cpu };
                state.sparse_snapshot = sparse_snapshot;
            }
            try state.apply(model.weights.layers[prior].attn, prior, row, position, cfg, &output, &scratch);
            state.observer = null;
            state.sparse_snapshot = null;
        }
        try model.cache.append(1);
    }

    // Export post-insertion physical ring slots, masking never-written slots in source indices.
    const decoded_view = attention.View{ .buffer = decoded };
    const native_window = decoded.slice(u16, 128 * 512 + compressed_rows * 640)[0 .. 128 * 512];
    const native_compressed = decoded.slice(u16, 128 * 512 + compressed_rows * 640)[128 * 512 ..][0 .. compressed_rows * 512];
    const native_index_keys = decoded.slice(u16, 128 * 512 + compressed_rows * 640)[128 * 512 + compressed_rows * 512 ..][0 .. compressed_rows * 128];
    const unpack_cb = model.queue.commandBufferUnretained();
    const unpack_enc = unpack_cb.compute(.serial);
    var unpack_encoding = true;
    errdefer if (unpack_encoding) unpack_enc.end();
    const ring = try model.cache.ring(@intCast(layer));
    try model.pipelines.attention.cache.encodeUnpack(unpack_enc, ring, decoded_view, .{ .rows = 128, .width = 512, .stride = 512, .packed_stride = 528, .block = 32, .kind = .window_fp8 });
    if (compressed_rows != 0) {
        const source = try model.cache.source(@intCast(layer), ratio);
        try model.pipelines.attention.cache.encodeUnpack(unpack_enc, source.main, decoded_view.shifted(128 * 512 * 2), .{ .rows = @intCast(compressed_rows), .width = 512, .stride = 512, .packed_stride = 288, .block = 16, .kind = .main_fp4 });
        try model.pipelines.attention.cache.encodeUnpack(unpack_enc, source.index, decoded_view.shifted((128 * 512 + compressed_rows * 512) * 2), .{ .rows = @intCast(compressed_rows), .width = 128, .stride = 128, .packed_stride = 68, .block = 32, .kind = .index_fp4 });
    }
    unpack_enc.end();
    unpack_encoding = false;
    unpack_cb.commit();
    unpack_cb.wait();
    if (unpack_cb.failure() != null) return error.OperatorGpuFailure;
    const reference_window = state.window[layer * 128 * 512 ..][0 .. 128 * 512];
    const reference_compressed = try gpa.alloc(u16, compressed_rows * 640);
    defer gpa.free(reference_compressed);
    const reference_main = reference_compressed[0 .. compressed_rows * 512];
    const reference_index_keys = reference_compressed[compressed_rows * 512 ..];
    for (0..compressed_rows) |row| {
        const source_row = state.sources[state.active_owner * state.capacity * 640 + row * 640 ..][0..640];
        @memcpy(reference_main[row * 512 ..][0..512], source_row[0..512]);
        @memcpy(reference_index_keys[row * 128 ..][0..128], source_row[512..640]);
    }
    const native_selected = diagnosticRow(i32, model.attention_workspace.selection.positions(), 512)[0..topk];
    var reference_selected: [512]i32 = undefined;
    for (state.selected[0..topk], 0..) |row, i| reference_selected[i] = @intCast(row);
    var native_indices_storage: [640]i32 = undefined;
    var reference_indices_storage: [640]i32 = undefined;
    const native_indices = attentionHistoryIndices(token, native_selected, window_rows, &native_indices_storage);
    const reference_indices = attentionHistoryIndices(token, reference_selected[0..topk], window_rows, &reference_indices_storage);
    var sinks: [64]f32 = undefined;
    for (&sinks, 0..) |*value, head| value.* = math.value(weights.sink, head);

    var first_stage: ?[]const u8 = null;
    inline for (std.enums.values(AttentionStage)) |stage|
        if (first_stage == null and !std.mem.eql(u16, capture.row(stage), capture.cpuRow(stage))) {
            first_stage = @tagName(stage);
        };
    const header = try std.json.Stringify.valueAlloc(gpa, .{
        .mode = "attention-history-sameinput",
        .token = token,
        .layer = layer,
        .history_count = history_count,
        .token_identity_source = "caller_supplied_TOKEN_and_trace_POSITION_gpu_filenames_raw_snapshots_have_no_token_header",
        .normalized_prefix_replay = .{ .file_pattern = "trace-{position}-gpu.bin", .origin = "retained_native_GPU_normalized_attention_inputs", .first_position = 0, .last_position = token, .first_layer = 0, .last_layer = layer, .order = "position_major_then_layer_ascending_inclusive", .snapshot_count_per_file = model.weights.layers.len * 2, .snapshot_length_bytes = snapshot_bytes, .attention_snapshot_index = "2*layer", .normalized_input_offset_within_snapshot_bytes = 40960, .normalized_input_length_bytes = 10240, .native_cache_reset = true, .native_cache_cursor_matches_each_position = true, .native_cache_append_rows_per_position = 1, .reference_position_passed_explicitly = true, .reference_capacity = state.capacity },
        .identical_retained_normalized_inputs = true,
        .independent_reference_history = true,
        .native_history_replay_branch_matches_retained_trace = std.mem.eql(u16, capture.row(.wo_b), &retained_branch),
        .first_differing_same_history_input_stage = first_stage,
        .window_history_equal = std.mem.eql(u16, native_window, reference_window),
        .compressed_main_equal = std.mem.eql(u16, native_compressed, reference_main),
        .compressed_index_keys_equal = std.mem.eql(u16, native_index_keys, reference_index_keys),
        .selection_equal = std.mem.eql(i32, native_selected, reference_selected[0..topk]),
        .compression_ratio = ratio,
        .compressed_rows = compressed_rows,
        .selection_count = topk,
        .reference_dot = "independent_EXL3_FP64",
        .reference_sparse_dot = "independent_FP64",
        .qualification_gate_applied = false,
        .qualification_oracle_modified_to_reuse_native_values = false,
        .same_native_query_cache_diagnostic = "separate_reference_State_with_captured_native_window_main_selection_and_q_not_used_in_independent_chain",
        .official_revision = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027",
        .receipt_dtype_endianness = "little_endian",
        .receipt_strides = "contiguous_element_strides",
        .files = .{
            .stage_chain = .{ .native = "attention-native.bf16", .reference = "attention-reference.bf16", .dtype = "bf16", .shape = .{attention_elements}, .strides = .{1}, .layout = "stages_table_below" },
            .normalized_prefix = .{ .file = "attention-normalized-prefix.bf16", .dtype = "bf16", .shape = .{ history_count, layer + 1, 5120 }, .strides = .{ (layer + 1) * 5120, 5120, 1 } },
            .target_input = .{ .file = "attention-input.bf16", .dtype = "bf16", .shape = .{5120}, .strides = .{1} },
            .retained_branch = .{ .file = "attention-retained-branch.bf16", .dtype = "bf16", .shape = .{5120}, .strides = .{1} },
            .query = .{ .native = "attention-query-native.bf16", .reference = "attention-query-reference.bf16", .dtype = "bf16", .shape = .{ 1, 1, 64, 512 }, .strides = .{ 32768, 32768, 512, 1 } },
            .window = .{ .native = "attention-window-native.bf16", .reference = "attention-window-reference.bf16", .dtype = "bf16", .shape = .{ 1, 128, 512 }, .strides = .{ 65536, 512, 1 }, .layout = "physical_ring_slot_position_modulo_128_includes_current_quantized_KV" },
            .compressed_main = .{ .native = "attention-compressed-native.bf16", .reference = "attention-compressed-reference.bf16", .dtype = "bf16", .shape = .{ 1, compressed_rows, 512 }, .strides = .{ compressed_rows * 512, 512, 1 } },
            .compressed_index_keys = .{ .native = "attention-index-keys-native.bf16", .reference = "attention-index-keys-reference.bf16", .dtype = "bf16", .shape = .{ 1, compressed_rows, 128 }, .strides = .{ compressed_rows * 128, 128, 1 } },
            .selection = .{ .native = "attention-selection-native.i32", .reference = "attention-selection-reference.i32", .dtype = "i32", .shape = .{ 1, topk }, .strides = .{ topk, 1 }, .layout = "actual_compressed_row_ids_compact_export_native_workspace_row_stride_is_512" },
            .source_kv = .{ .native = "attention-source-kv-native.bf16", .reference = "attention-source-kv-reference.bf16", .dtype = "bf16", .shape = .{ 1, window_rows + compressed_rows, 512 }, .strides = .{ (window_rows + compressed_rows) * 512, 512, 1 }, .layout = "source_sparse_attn_KV_window_then_compressed_rows_no_requantization" },
            .source_indices = .{ .native = "attention-source-indices-native.i32", .reference = "attention-source-indices-reference.i32", .dtype = "i32", .shape = .{ 1, 1, native_indices.len }, .strides = .{ native_indices.len, native_indices.len, 1 }, .layout = "source_get_window_topk_idxs_then_actual_selection_plus_window_rows_minus_one_remains_masked" },
            .sinks = .{ .file = "attention-sinks.f32", .dtype = "f32", .shape = .{64}, .strides = .{1} },
            .sparse_output = .{ .native = "attention-sparse-native.bf16", .reference = "attention-sparse-reference.bf16", .same_native_input_reference = "attention-sparse-same-native-input-reference.bf16", .dtype = "bf16", .shape = .{ 1, 1, 64, 512 }, .strides = .{ 32768, 32768, 512, 1 } },
        },
        .source_sparse_call = "sparse_attn(query,source_kv,sinks,source_indices,1/sqrt(512)); output_before_inverse_RoPE",
        .source_sparse_replay_is_operator_evidence_only = true,
    }, .{});
    defer gpa.free(header);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, header[0 .. header.len - 1]);
    try report.appendSlice(gpa, ",\"cases\":[\n");
    var count: usize = 0;
    inline for (std.enums.values(AttentionStage)) |stage|
        try attentionMetric(u16, gpa, &report, &count, @tagName(stage), "same_retained_normalized_history_input_chain", capture.row(stage), capture.cpuRow(stage));
    try attentionMetric(u16, gpa, &report, &count, "retained_branch", "native_history_replay_vs_retained_native_trace", capture.row(.wo_b), &retained_branch);
    try attentionMetric(u16, gpa, &report, &count, "window_history", "same_retained_normalized_history_input_chain", native_window, reference_window);
    if (compressed_rows != 0) {
        try attentionMetric(u16, gpa, &report, &count, "compressed_kv", "same_retained_normalized_history_input_chain", native_compressed, reference_main);
        try attentionMetric(u16, gpa, &report, &count, "compressed_index_keys", "same_retained_normalized_history_input_chain", native_index_keys, reference_index_keys);
    }

    // This separate diagnostic State never overwrites independent retained-history reference data.
    var native_input_state = try attention_math.State.init(gpa, history_count);
    defer native_input_state.deinit();
    for (0..compressed_rows) |row|
        @memcpy(native_input_state.sources[row * 640 ..][0..512], native_compressed[row * 512 ..][0..512]);
    var helper_indices_storage: [640]i32 = undefined;
    const helper_indices = attentionHistoryIndices(token, native_selected, 128, &helper_indices_storage);
    var same_input: [32768]u16 = undefined;
    for (0..64) |head|
        native_input_state.sparseHead(capture.row(.q)[head * 512 ..][0..512], native_window, helper_indices, sinks[head], same_input[head * 512 ..][0..512]);
    try attentionMetric(u16, gpa, &report, &count, "sparse_o", "same_native_query_window_compressed_cache_selection_reference_sparse_diagnostic", capture.row(.sparse_o), &same_input);
    try attentionWrite(gpa, io, dir, "attention-sparse-same-native-input-reference.bf16", std.mem.sliceAsBytes(&same_input));
    math.rms(capture.row(.qr_projection), weights.q_norm, cfg.eps, same_input[0..1280]);
    try attentionMetric(u16, gpa, &report, &count, "qr_norm", "same_native_input_reference_rms_diagnostic", capture.row(.qr_norm), same_input[0..1280]);
    math.rms(capture.row(.kv_projection), weights.kv_norm, cfg.eps, same_input[0..512]);
    try attentionMetric(u16, gpa, &report, &count, "kv_norm", "same_native_input_reference_rms_diagnostic", capture.row(.kv_norm), same_input[0..512]);
    @memcpy(&same_input, capture.row(.q_projection));
    for (0..64) |head| math.rope(same_input[head * 512 ..][0..512], token, ratio != 0, false, cfg);
    try attentionMetric(u16, gpa, &report, &count, "q", "same_native_input_reference_rope_diagnostic", capture.row(.q), &same_input);
    @memcpy(same_input[0..512], capture.row(.kv_norm));
    math.rope(same_input[0..512], token, ratio != 0, false, cfg);
    try attentionMetric(u16, gpa, &report, &count, "kv_rope", "same_native_input_reference_rope_diagnostic", capture.row(.kv_rope), same_input[0..512]);
    @memcpy(same_input[0..512], capture.row(.kv_rope));
    math.quantize(same_input[0..512], .window);
    try attentionMetric(u16, gpa, &report, &count, "kv_quantized", "same_native_input_reference_quantize_diagnostic", capture.row(.kv_quantized), same_input[0..512]);
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
    try report.appendSlice(gpa, "\n],\"stages\":[");
    inline for (std.enums.values(AttentionStage), 0..) |stage, i| {
        const descriptor = try std.json.Stringify.valueAlloc(gpa, .{ .name = @tagName(stage), .offset_bytes = attentionOffset(stage) * 2, .elements = attention_widths[i] }, .{});
        defer gpa.free(descriptor);
        if (i != 0) try report.appendSlice(gpa, ",");
        try report.appendSlice(gpa, descriptor);
    }
    try report.appendSlice(gpa, "]}\n");
    const source_kv = try gpa.alloc(u16, 2 * (window_rows + compressed_rows) * 512);
    defer gpa.free(source_kv);
    const source_native = source_kv[0 .. (window_rows + compressed_rows) * 512];
    const source_reference = source_kv[(window_rows + compressed_rows) * 512 ..];
    @memcpy(source_native[0 .. window_rows * 512], native_window[0 .. window_rows * 512]);
    @memcpy(source_native[window_rows * 512 ..], native_compressed);
    @memcpy(source_reference[0 .. window_rows * 512], reference_window[0 .. window_rows * 512]);
    @memcpy(source_reference[window_rows * 512 ..], reference_main);
    const input_row = normalized_prefix[(history_count * (layer + 1) - 1) * 5120 ..][0..5120];
    inline for (.{
        "attention-history-sameinput.json",    "attention-native.bf16",                  "attention-reference.bf16",
        "attention-normalized-prefix.bf16",    "attention-input.bf16",                   "attention-retained-branch.bf16",
        "attention-query-native.bf16",         "attention-query-reference.bf16",         "attention-window-native.bf16",
        "attention-window-reference.bf16",     "attention-compressed-native.bf16",       "attention-compressed-reference.bf16",
        "attention-index-keys-native.bf16",    "attention-index-keys-reference.bf16",    "attention-selection-native.i32",
        "attention-selection-reference.i32",   "attention-source-kv-native.bf16",        "attention-source-kv-reference.bf16",
        "attention-source-indices-native.i32", "attention-source-indices-reference.i32", "attention-sinks.f32",
        "attention-sparse-native.bf16",        "attention-sparse-reference.bf16",
    }, .{
        report.items,                                      std.mem.sliceAsBytes(native.slice(u16, attention_elements)), std.mem.sliceAsBytes(reference_rows),
        std.mem.sliceAsBytes(normalized_prefix),           std.mem.sliceAsBytes(input_row),                             std.mem.sliceAsBytes(&retained_branch),
        std.mem.sliceAsBytes(capture.row(.q)),             std.mem.sliceAsBytes(capture.cpuRow(.q)),                    std.mem.sliceAsBytes(native_window),
        std.mem.sliceAsBytes(reference_window),            std.mem.sliceAsBytes(native_compressed),                     std.mem.sliceAsBytes(reference_main),
        std.mem.sliceAsBytes(native_index_keys),           std.mem.sliceAsBytes(reference_index_keys),                  std.mem.sliceAsBytes(native_selected),
        std.mem.sliceAsBytes(reference_selected[0..topk]), std.mem.sliceAsBytes(source_native),                         std.mem.sliceAsBytes(source_reference),
        std.mem.sliceAsBytes(native_indices),              std.mem.sliceAsBytes(reference_indices),                     std.mem.sliceAsBytes(&sinks),
        std.mem.sliceAsBytes(capture.row(.sparse_o)),      std.mem.sliceAsBytes(capture.cpuRow(.sparse_o)),
    }) |name, data| try attentionWrite(gpa, io, dir, name, data);
}
