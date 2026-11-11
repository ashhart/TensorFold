//! Retained-input MoE routing and expert isolation.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const projection = @import("projection.zig");
const math = @import("math.zig");
const attention = tf.deepseek_v41.attention;
const Model = tf.deepseek_v41.Model;
const ex = tf.exl3_projection;
const diagnosticRow = @import("operator_diagnostic.zig").diagnosticRow;

const moe_reference = @import("reference.zig");
const MoeRows = struct {
    gate: [7][2304]u16 = undefined,
    up: [7][2304]u16 = undefined,
    activation: [7][2304]u16 = undefined,
    down: [7][5120]u16 = undefined,
    branch: [5120]u16 = undefined,
};
const MoeFirst = struct {
    source_chain: ?[]const u8 = null,
    source_expert_stage: ?[]const u8 = null,
    native_input_diagnostic: ?[]const u8 = null,
    native_weights_reference_chain_diagnostic: ?[]const u8 = null,
    retained_weights_reference_chain_diagnostic: ?[]const u8 = null,
    native_down_reference_combine_diagnostic_differs: bool = false,
};

fn moeMetric(comptime T: type, gpa: std.mem.Allocator, report: *std.ArrayList(u8), count: *usize, name: []const u8, comparison: []const u8, diagnostic_reuse: bool, actual: []const T, expected: []const T) !bool {
    if (actual.len == 0 or actual.len != expected.len) return error.InvalidMoeMetricShape;
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    var max_abs: f64 = 0;
    var worst: usize = 0;
    var unequal: usize = 0;
    var first: ?usize = null;
    for (actual, expected, 0..) |a, r, i| {
        const x = if (T == u16) math.f(a) else a;
        const y = if (T == u16) math.f(r) else r;
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteMoeDiagnostic;
        const delta = @as(f64, x) - y;
        squared += delta * delta;
        magnitude += @as(f64, y) * y;
        if (@abs(delta) > max_abs) {
            max_abs = @abs(delta);
            worst = i;
        }
        const ab = if (T == u16) a else @as(u32, @bitCast(a));
        const rb = if (T == u16) r else @as(u32, @bitCast(r));
        if (ab != rb) {
            unequal += 1;
            if (first == null) first = i;
        }
    }
    const line = try std.json.Stringify.valueAlloc(gpa, .{
        .name = name,
        .comparison = comparison,
        .diagnostic_native_value_reuse = diagnostic_reuse,
        .qualification_gate_applied = false,
        .elements = actual.len,
        .output_dtype = if (T == u16) "bf16" else "f32",
        .normalized_rms_error = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude),
        .max_absolute_error = max_abs,
        .worst_index = worst,
        .bit_unequal_count = unequal,
        .first_bit_unequal_index = first,
        .worst_native = if (T == u16) math.f(actual[worst]) else actual[worst],
        .worst_reference = if (T == u16) math.f(expected[worst]) else expected[worst],
    }, .{});
    defer gpa.free(line);
    if (count.* != 0) try report.appendSlice(gpa, ",\n");
    try report.appendSlice(gpa, line);
    count.* += 1;
    std.debug.print("{s}\n", .{line});
    return unequal != 0;
}

fn moeWrite(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

fn moeWriteRows(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, prefix: []const u8, rows: *const MoeRows) !void {
    inline for (.{ "gate", "up", "activation", "down", "branch" }) |stage| {
        const name = try std.fmt.allocPrint(gpa, "{s}-{s}.bf16", .{ prefix, stage });
        defer gpa.free(name);
        try moeWrite(gpa, io, dir, name, std.mem.asBytes(&@field(rows.*, stage)));
    }
}

fn moeMatrices(model: *Model, layer: usize, expert: ?u32) ![3]ex.Matrix {
    if (expert == null) return model.weights.layers[layer].moe.shared;
    var matrices: [3]ex.Matrix = undefined;
    var name: [128]u8 = undefined;
    inline for (.{ "w1", "w3", "w2" }, 0..) |part, i| {
        const key = try std.fmt.bufPrint(&name, "layers.{d}.ffn.experts.{d}.{s}", .{ layer, expert.?, part });
        matrices[i] = try tf.deepseek_v41_weights.lookup(&model.checkpoint, key, if (i == 2) 2304 else 5120, if (i == 2) 5120 else 2304);
    }
    return matrices;
}

fn moeCombineDown(ids: *const [6]u32, down: *const [7][5120]u16, branch: *[5120]u16) void {
    @setFloatMode(.strict);
    var sum: [5120]f32 = @splat(0);
    // Keep slots in top-six order in receipts, but match the official ascending expert traversal.
    for (0..384) |expert| {
        for (ids, 0..) |id, slot| {
            if (id == expert)
                for (&sum, down[slot]) |*s, value| {
                    s.* += math.f(value);
                };
        }
    }
    for (branch, sum, down[6]) |*v, routed, shared| v.* = math.b(routed + math.f(shared));
}

fn moeCombine(ids: *const [6]u32, rows: *MoeRows) void {
    moeCombineDown(ids, &rows.down, &rows.branch);
}

fn moeReferenceRows(model: *Model, layer: usize, input: []const u16, ids: *const [6]u32, routing: *const [6]f32, rows: *MoeRows, scratch: *projection.Scratch, previous: ?*const MoeRows, previous_ids: ?*const [6]u32) !void {
    for (0..7) |slot| {
        const matrices = try moeMatrices(model, layer, if (slot == 6) null else ids[slot]);
        var reuse: ?usize = if (slot == 6 and previous != null) 6 else null;
        if (slot != 6 and previous_ids != null)
            for (previous_ids.?, 0..) |id, prior| {
                if (id == ids[slot]) {
                    reuse = prior;
                    break;
                }
            };
        if (reuse) |prior| {
            rows.gate[slot] = previous.?.gate[prior];
            rows.up[slot] = previous.?.up[prior];
        } else {
            try projection.apply(.{ .trellis = matrices[0] }, input, &rows.gate[slot], scratch);
            try projection.apply(.{ .trellis = matrices[1] }, input, &rows.up[slot], scratch);
        }
        moe_reference.moeActivation(&rows.gate[slot], &rows.up[slot], if (slot == 6) 1 else routing[slot], model.cfg, &rows.activation[slot]);
        // A shared or identical-weight expert result can be reused only when the BF16 input is identical.
        if (reuse != null and std.mem.eql(u16, &rows.activation[slot], &previous.?.activation[reuse.?])) {
            rows.down[slot] = previous.?.down[reuse.?];
        } else {
            try projection.apply(.{ .trellis = matrices[2] }, &rows.activation[slot], &rows.down[slot], scratch);
        }
    }
    moeCombine(ids, rows);
}

fn moeEncodeNative(model: *Model, layer: usize, select_routes: bool) !void {
    const ws = &model.workspace;
    const p = &model.pipelines;
    const w = &model.weights.layers[layer].moe;
    const cb = model.queue.commandBufferUnretained();
    const enc = cb.compute(.serial);
    var encoding = true;
    errdefer if (encoding) enc.end();
    for (model.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
    for (w.descriptors) |buffer| enc.useResource(buffer, mtl.ResourceUsage.read);
    if (select_routes) {
        tf.deepseek_v41.ops.encodeDense(enc, &p.ops, ws.view(.normalized), .{ .buffer = w.gate.buffer, .offset = w.gate.offset }, ws.view(.router_logits), .{ .rows = 1, .k = 5120, .n = 384, .x_stride = 5120, .w_stride = 5120, .y_stride = 384, .w_dtype = .f16, .y_dtype = .f32 });
        enc.barrier();
        // Bind the existing production router; no new native interface or diagnostic math kernel.
        enc.setPipeline(p.route.select);
        inline for (.{ ws.view(.router_logits), attention.View{ .buffer = w.bias.buffer, .offset = w.bias.offset }, ws.view(.route_ids), ws.view(.route_weights) }, 0..) |view, i|
            enc.setBuffer(view.buffer, view.offset, i);
        const Shape = extern struct { rows: u32, logit_stride: u32, output_stride: u32, temperature: f32, bias_dtype: u32 };
        enc.setValue(Shape{ .rows = 1, .logit_stride = 384, .output_stride = 6, .temperature = 1, .bias_dtype = 2 }, 4);
        enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
        enc.barrier();
    }
    var routed_rows = ex.Rows{ .count = 6, .x_stride = 5120, .y_stride = 2304, .routes = .{ .ids = .{ .buffer = ws.view(.route_ids).buffer, .offset = ws.view(.route_ids).offset }, .repeat = 6 } };
    for (0..2) |i| {
        const target = ws.view(if (i == 0) .gate else .up);
        const input = ws.view(.normalized);
        try ex.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = w.descriptors[i] }, .count = 384, .k = 5120, .n = 2304 }, .{ .buffer = input.buffer, .offset = input.offset }, .{ .buffer = target.buffer, .offset = target.offset }, ws.projection(), routed_rows);
        enc.barrier();
    }
    tf.deepseek_v41.ops.encodeSwiGLU(enc, &p.ops, ws.view(.gate), ws.view(.up), ws.view(.route_weights), ws.view(.activation), .{ .rows = 6, .width = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = true });
    enc.barrier();
    routed_rows.x_stride = 2304;
    routed_rows.y_stride = 5120;
    routed_rows.routes.?.repeat = 1;
    const activation = ws.view(.activation);
    const down = ws.view(.routed);
    try ex.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = w.descriptors[2] }, .count = 384, .k = 2304, .n = 5120 }, .{ .buffer = activation.buffer, .offset = activation.offset }, .{ .buffer = down.buffer, .offset = down.offset }, ws.projection(), routed_rows);
    enc.barrier();
    for (0..2) |i| {
        const target = ws.view(if (i == 0) .shared_gate else .shared_up);
        const input = ws.view(.normalized);
        ex.useMatrix(enc, w.shared[i]);
        try ex.encode(&p.projection, enc, w.shared[i], .{ .buffer = input.buffer, .offset = input.offset }, .{ .buffer = target.buffer, .offset = target.offset }, ws.projection(), .{ .count = 1, .x_stride = 5120, .y_stride = 2304 });
        enc.barrier();
    }
    tf.deepseek_v41.ops.encodeSwiGLU(enc, &p.ops, ws.view(.shared_gate), ws.view(.shared_up), ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = 1, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
    enc.barrier();
    const shared_activation = ws.view(.shared_activation);
    const shared_down = ws.view(.shared_down);
    ex.useMatrix(enc, w.shared[2]);
    try ex.encode(&p.projection, enc, w.shared[2], .{ .buffer = shared_activation.buffer, .offset = shared_activation.offset }, .{ .buffer = shared_down.buffer, .offset = shared_down.offset }, ws.projection(), .{ .count = 1, .x_stride = 2304, .y_stride = 5120 });
    enc.barrier();
    tf.deepseek_v41.ops.encodeCombine(enc, &p.ops, ws.view(.routed), ws.view(.shared_down), ws.view(.route_ids), ws.view(.branch), .{ .rows = 1, .width = 5120, .routed_row_stride = 6 * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120 });
    enc.end();
    encoding = false;
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.OperatorGpuFailure;
}

fn moeCaptureNative(model: *Model, rows: *MoeRows) void {
    const ws = &model.workspace;
    inline for (.{ "gate", "up", "activation", "down" }, .{ .gate, .up, .activation, .routed }, .{ .shared_gate, .shared_up, .shared_activation, .shared_down }) |stage, routed, shared| {
        const width = if (comptime std.mem.eql(u8, stage, "down")) 5120 else 2304;
        const source = diagnosticRow(u16, ws.view(routed), 6 * width);
        for (0..6) |slot| @memcpy(&@field(rows.*, stage)[slot], source[slot * width ..][0..width]);
        @memcpy(&@field(rows.*, stage)[6], diagnosticRow(u16, ws.view(shared), width));
    }
    @memcpy(&rows.branch, diagnosticRow(u16, ws.view(.branch), 5120));
}

fn moeCompareRows(gpa: std.mem.Allocator, report: *std.ArrayList(u8), count: *usize, comparison: []const u8, reuse: bool, actual: *const MoeRows, actual_ids: *const [6]u32, expected: *const MoeRows, expected_ids: *const [6]u32, first: *?[]const u8) !void {
    // Compare like expert IDs rather than assuming independent top-six orders are identical.
    inline for (.{ "gate", "up", "activation", "down" }) |stage| {
        var differs = false;
        for (0..7) |slot| {
            var reference_slot: ?usize = if (slot == 6) 6 else null;
            if (slot != 6)
                for (expected_ids, 0..) |id, i| {
                    if (id == actual_ids[slot]) {
                        reference_slot = i;
                        break;
                    }
                };
            if (reference_slot == null) continue;
            var name: [64]u8 = undefined;
            const label = if (slot == 6) try std.fmt.bufPrint(&name, "shared_{s}", .{stage}) else try std.fmt.bufPrint(&name, "expert_{d}_{s}", .{ actual_ids[slot], stage });
            differs = (try moeMetric(u16, gpa, report, count, label, comparison, reuse, &@field(actual.*, stage)[slot], &@field(expected.*, stage)[reference_slot.?])) or differs;
        }
        if (differs and first.* == null) first.* = stage;
    }
    if (try moeMetric(u16, gpa, report, count, "combined_branch", comparison, reuse, &actual.branch, &expected.branch))
        if (first.* == null) {
            first.* = "combined_branch";
        };
}

/// Retained local snapshot isolation. MoE has no history; qualification computes every value independently.
pub fn runMoe(gpa: std.mem.Allocator, io: std.Io, model: *Model, snapshot: []const u8, dir: []const u8, layer: usize) !void {
    if (layer >= model.weights.layers.len) return error.InvalidMoeLayer;
    const blob = try std.Io.Dir.cwd().readFileAlloc(io, snapshot, gpa, .limited(8 << 20));
    defer gpa.free(blob);
    const snapshot_offset = (layer * 2 + 1) * 61488;
    if (blob.len != 80 * 61488 or blob.len < snapshot_offset + 61488) return error.InvalidMoeSnapshot;
    try model.reset();
    const input = model.workspace.view(.normalized);
    @memcpy(input.buffer.contents()[input.offset..][0..10240], blob[snapshot_offset + 40960 ..][0..10240]);
    const input_row = diagnosticRow(u16, input, 5120);
    var retained_ids: [6]u32 = undefined;
    var retained_weights: [6]f32 = undefined;
    var retained_branch: [5120]u16 = undefined;
    for (&retained_ids, 0..) |*id, i| id.* = std.mem.readInt(u32, blob[snapshot_offset + 61440 + 4 * i ..][0..4], .little);
    for (&retained_weights, 0..) |*weight, i|
        weight.* = @bitCast(std.mem.readInt(u32, blob[snapshot_offset + 61464 + 4 * i ..][0..4], .little));
    @memcpy(std.mem.asBytes(&retained_branch), blob[snapshot_offset + 51200 ..][0..10240]);
    for (retained_ids, retained_weights, 0..) |id, weight, i| {
        if (id >= 384 or !std.math.isFinite(weight)) return error.InvalidMoeRetainedRouting;
        for (retained_ids[0..i]) |prior| if (id == prior) return error.InvalidMoeRetainedRouting;
    }
    const native = try gpa.create(MoeRows);
    defer gpa.destroy(native);
    const source = try gpa.create(MoeRows);
    defer gpa.destroy(source);
    const native_weights_reference = try gpa.create(MoeRows);
    defer gpa.destroy(native_weights_reference);
    const retained_reference = try gpa.create(MoeRows);
    defer gpa.destroy(retained_reference);
    const native_inputs_reference = try gpa.create(MoeRows);
    defer gpa.destroy(native_inputs_reference);
    const retained_native = try gpa.create(MoeRows);
    defer gpa.destroy(retained_native);
    var scratch = try projection.Scratch.init(gpa, 5120, 5120);
    defer scratch.deinit();

    try moeEncodeNative(model, layer, true);
    moeCaptureNative(model, native);
    var native_ids: [6]u32 = undefined;
    var native_weights: [6]f32 = undefined;
    var native_logits_routing: moe_reference.MoeRouting = .{};
    @memcpy(&native_ids, diagnosticRow(u32, model.workspace.view(.route_ids), 6));
    @memcpy(&native_weights, diagnosticRow(f32, model.workspace.view(.route_weights), 6));
    @memcpy(&native_logits_routing.logits, diagnosticRow(f32, model.workspace.view(.router_logits), 384));
    moe_reference.moeSelect(model.weights.layers[layer].moe, model.cfg, &native_logits_routing.logits, &native_logits_routing.scores, &native_logits_routing.ids, &native_logits_routing.weights);
    var source_routing: moe_reference.MoeRouting = .{};
    math.dense(model.weights.layers[layer].moe.gate, input_row, &source_routing.logits);
    moe_reference.moeSelect(model.weights.layers[layer].moe, model.cfg, &source_routing.logits, &source_routing.scores, &source_routing.ids, &source_routing.weights);
    try moeReferenceRows(model, layer, input_row, &source_routing.ids, &source_routing.weights, source, &scratch, null, null);
    try moeReferenceRows(model, layer, input_row, &native_ids, &native_weights, native_weights_reference, &scratch, source, &source_routing.ids);
    try moeReferenceRows(model, layer, input_row, &retained_ids, &retained_weights, retained_reference, &scratch, native_weights_reference, &native_ids);
    for (0..7) |slot| {
        native_inputs_reference.gate[slot] = native.gate[slot];
        native_inputs_reference.up[slot] = native.up[slot];
        moe_reference.moeActivation(&native.gate[slot], &native.up[slot], if (slot == 6) 1 else native_weights[slot], model.cfg, &native_inputs_reference.activation[slot]);
        // Deliberately use the captured native activation, not the preceding reference activation.
        if (std.mem.eql(u16, &native.activation[slot], &native_weights_reference.activation[slot])) {
            native_inputs_reference.down[slot] = native_weights_reference.down[slot];
        } else {
            const matrices = try moeMatrices(model, layer, if (slot == 6) null else native_ids[slot]);
            try projection.apply(.{ .trellis = matrices[2] }, &native.activation[slot], &native_inputs_reference.down[slot], &scratch);
        }
    }
    moeCombine(&native_ids, native_inputs_reference);
    @memcpy(diagnosticRow(u32, model.workspace.view(.route_ids), 6), &retained_ids);
    @memcpy(diagnosticRow(f32, model.workspace.view(.route_weights), 6), &retained_weights);
    try moeEncodeNative(model, layer, false);
    moeCaptureNative(model, retained_native);

    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    const header = try std.json.Stringify.valueAlloc(gpa, .{
        .mode = "moe-sameinput",
        .token = @as(?usize, null),
        .layer = layer,
        .token_identity_source = "caller_supplied_snapshot_argument_raw_snapshot_has_no_token_header",
        .retained_route_origin = "native_GPU_snapshot_command_contract",
        .snapshot_count = 80,
        .snapshot_length_bytes = 61488,
        .snapshot_index = layer * 2 + 1,
        .snapshot_stage = "moe",
        .input_offset_bytes = snapshot_offset + 40960,
        .input_length_bytes = 10240,
        .retained_ids_offset_bytes = snapshot_offset + 61440,
        .retained_weights_offset_bytes = snapshot_offset + 61464,
        .identical_normalized_input = true,
        .reference_dot = "independent_EXL3_FP64",
        .qualification_gate_applied = false,
        .qualification_oracle_modified_to_reuse_native_values = false,
        .official_revision = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027",
        .source_ids = source_routing.ids,
        .source_weights = source_routing.weights,
        .native_ids = native_ids,
        .native_weights = native_weights,
        .retained_ids = retained_ids,
        .retained_weights = retained_weights,
        .source_ids_equal_native_order = std.mem.eql(u32, &source_routing.ids, &native_ids),
        .native_ids_equal_retained_order = std.mem.eql(u32, &native_ids, &retained_ids),
        .source_select_on_native_logits_ids = native_logits_routing.ids,
        .source_select_on_native_logits_weights = native_logits_routing.weights,
        .score_receipt_origin = "source_sqrt_softplus_applied_to_captured_native_logits_not_GPU_score_capture",
        .expert_receipt_layout = "slot_major_top6_ids_order_then_shared_slot6",
        .expert_sum_order = "ascending_expert_id_FP32_shared_last_final_BF16",
        .receipt_prefixes = .{ "moe-native", "moe-source", "moe-native-weights-reference-diagnostic", "moe-native-inputs-reference-diagnostic", "moe-retained-weights-reference-diagnostic", "moe-retained-route-native-diagnostic" },
        .receipt_shapes = .{ .gate = .{ 7, 2304 }, .up = .{ 7, 2304 }, .activation = .{ 7, 2304 }, .down = .{ 7, 5120 }, .branch = .{5120} },
        .native_inputs_reference_contract = "activation:reference_on_native_gate_up_and_weight;down:FP64_on_native_activation;branch:reference_combine_of_these_down_outputs",
    }, .{});
    defer gpa.free(header);
    try report.appendSlice(gpa, header[0 .. header.len - 1]);
    try report.appendSlice(gpa, ",\"cases\":[\n");
    var count: usize = 0;
    var first: MoeFirst = .{};
    if (try moeMetric(f32, gpa, &report, &count, "route_logits", "source_chain", false, &native_logits_routing.logits, &source_routing.logits)) first.source_chain = "route_logits";
    if (try moeMetric(f32, gpa, &report, &count, "route_scores_source_transform", "source_score_transform_on_captured_native_logits_diagnostic", true, &native_logits_routing.scores, &source_routing.scores))
        if (first.source_chain == null) {
            first.source_chain = "route_scores_source_transform";
        };
    if (!std.mem.eql(u32, &source_routing.ids, &native_ids) and first.source_chain == null) first.source_chain = "route_ids";
    if (std.mem.eql(u32, &source_routing.ids, &native_ids)) {
        if (try moeMetric(f32, gpa, &report, &count, "route_weights", "source_chain", false, &native_weights, &source_routing.weights))
            if (first.source_chain == null) {
                first.source_chain = "route_weights";
            };
    }
    if (std.mem.eql(u32, &native_logits_routing.ids, &native_ids))
        _ = try moeMetric(f32, gpa, &report, &count, "route_weights", "source_select_on_same_native_logits_diagnostic", true, &native_weights, &native_logits_routing.weights);
    if (std.mem.eql(u32, &retained_ids, &native_ids))
        _ = try moeMetric(f32, gpa, &report, &count, "route_weights", "retained_native_weights_vs_reexecuted_native", true, &native_weights, &retained_weights);
    try moeCompareRows(gpa, &report, &count, "source_chain", false, native, &native_ids, source, &source_routing.ids, &first.source_expert_stage);
    if (first.source_chain == null) first.source_chain = first.source_expert_stage;
    try moeCompareRows(gpa, &report, &count, "same_native_ids_weights_reference_chain_diagnostic", true, native, &native_ids, native_weights_reference, &native_ids, &first.native_weights_reference_chain_diagnostic);
    inline for (.{ "activation", "down" }) |stage| {
        var differs = false;
        for (0..7) |slot| {
            var name: [64]u8 = undefined;
            const label = if (slot == 6) try std.fmt.bufPrint(&name, "shared_{s}", .{stage}) else try std.fmt.bufPrint(&name, "expert_{d}_{s}", .{ native_ids[slot], stage });
            differs = (try moeMetric(u16, gpa, &report, &count, label, if (comptime std.mem.eql(u8, stage, "activation")) "same_native_gate_up_weight_reference_activation_diagnostic" else "same_native_activation_FP64_down_diagnostic", true, &@field(native.*, stage)[slot], &@field(native_inputs_reference.*, stage)[slot])) or differs;
        }
        if (differs and first.native_input_diagnostic == null) first.native_input_diagnostic = stage;
    }
    _ = try moeMetric(u16, gpa, &report, &count, "combined_branch", "same_native_activation_FP64_down_then_reference_combine_diagnostic", true, &native.branch, &native_inputs_reference.branch);
    // Isolate combination arithmetic independently of every projection and route-weight difference.
    var native_down_reference_branch: [5120]u16 = undefined;
    moeCombineDown(&native_ids, &native.down, &native_down_reference_branch);
    first.native_down_reference_combine_diagnostic_differs =
        try moeMetric(u16, gpa, &report, &count, "combined_branch", "same_native_down_reference_combine_diagnostic", true, &native.branch, &native_down_reference_branch);
    try moeWrite(gpa, io, dir, "moe-native-down-reference-combine-diagnostic.bf16", std.mem.asBytes(&native_down_reference_branch));
    try moeCompareRows(gpa, &report, &count, "same_retained_native_ids_weights_reference_chain_diagnostic", true, retained_native, &retained_ids, retained_reference, &retained_ids, &first.retained_weights_reference_chain_diagnostic);
    _ = try moeMetric(u16, gpa, &report, &count, "combined_branch", "retained_native_branch_vs_reexecuted_native_routing", true, &native.branch, &retained_branch);
    _ = try moeMetric(u16, gpa, &report, &count, "combined_branch", "retained_native_branch_vs_reexecuted_retained_routes_diagnostic", true, &retained_native.branch, &retained_branch);
    const finding = try std.json.Stringify.valueAlloc(gpa, first, .{});
    defer gpa.free(finding);
    try report.appendSlice(gpa, "\n],\"first_differing_intermediate\":");
    try report.appendSlice(gpa, finding);
    try report.appendSlice(gpa, "}\n");
    try moeWrite(gpa, io, dir, "moe-sameinput.json", report.items);
    inline for (.{ "moe-native", "moe-source", "moe-native-weights-reference-diagnostic", "moe-native-inputs-reference-diagnostic", "moe-retained-weights-reference-diagnostic", "moe-retained-route-native-diagnostic" }, .{ native, source, native_weights_reference, native_inputs_reference, retained_reference, retained_native }) |prefix, rows|
        try moeWriteRows(gpa, io, dir, prefix, rows);
    inline for (.{ "moe-input.bf16", "moe-retained-branch.bf16", "moe-native-logits.f32", "moe-source-logits.f32", "moe-native-logits-source-scores.f32", "moe-source-scores.f32", "moe-native-ids.u32", "moe-source-ids.u32", "moe-retained-ids.u32", "moe-native-weights.f32", "moe-source-weights.f32", "moe-retained-weights.f32", "moe-native-logits-source-select-ids.u32", "moe-native-logits-source-select-weights.f32" }, .{ std.mem.sliceAsBytes(input_row), std.mem.asBytes(&retained_branch), std.mem.asBytes(&native_logits_routing.logits), std.mem.asBytes(&source_routing.logits), std.mem.asBytes(&native_logits_routing.scores), std.mem.asBytes(&source_routing.scores), std.mem.asBytes(&native_ids), std.mem.asBytes(&source_routing.ids), std.mem.asBytes(&retained_ids), std.mem.asBytes(&native_weights), std.mem.asBytes(&source_routing.weights), std.mem.asBytes(&retained_weights), std.mem.asBytes(&native_logits_routing.ids), std.mem.asBytes(&native_logits_routing.weights) }) |name, bytes|
        try moeWrite(gpa, io, dir, name, bytes);
}
