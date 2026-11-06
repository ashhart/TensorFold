//! Bounded investigation snapshots; file I/O and independent CPU work are not performance measurements.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const reference_mod = @import("reference.zig");
const Reference = reference_mod.Reference;
const engram_math = @import("engram_math.zig");
const engram_rows = tf.deepseek_v41_engram_rows;
fn streamRow(p: *anyopaque, layer: usize, row: u32, out: []u16) anyerror!void {
    const r: *engram_rows.Rows = @ptrCast(@alignCast(p));
    return r.rowsU32(layer, &.{row}, 256, out);
}
const Stage = tf.deepseek_v41_profiler.Stage;
const widths = [_]usize{ 20480 * 2, 5120 * 2, 5120 * 2, 6 * 4, 6 * 4 };
const stride = 61488;
const bytes = stride * 80;
/// A second, additive record per layer for the Engram stage: the kv projection the gate multiplies, plus the four gate
/// scalars the reference applied. Layers 1 and 14 only; other slots stay zero. The gathered rows are not copied here —
/// the streamed reader stages them in its own buffer and `engram_kv` is a projection of exactly those rows, so a wrong
/// gather cannot survive this comparison (`tools/rowcheck.py` already hashes the reader's bytes independently).
const engram_slot = 25600 * 2 + 4 * 4;
const engram_bytes = engram_slot * 40;
/// Router logits, both sides, 384 f32 a layer: the only way to separate "the logits differ" from "the score transform
/// differs", since both feed the same six normalised weights.
const router_slot = 384 * 4;
const router_bytes = router_slot * 40;

const Capture = struct {
    buffer: mtl.Buffer,
    pipeline: mtl.Pipeline,
    cpu: []u8,
    engram_buffer: mtl.Buffer,
    engram_cpu: []u8,
    router_buffer: mtl.Buffer,
    router_cpu: []u8,

    fn gpu(ctx: *anyopaque, enc: mtl.ComputeEncoder, layer: u32, stage: Stage, model: *const Model) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        var destination: usize = (@as(usize, layer) * 2 + @intFromBool(stage == .moe)) * stride;
        const ws = &model.workspace;
        inline for (.{ ws.view(.stream), ws.view(.normalized), ws.view(.branch), ws.view(.route_ids), ws.view(.route_weights) }, widths, 0..) |source, count, field| {
            if (stage == .moe or field < 3) {
                enc.setPipeline(self.pipeline);
                enc.setBuffer(source.buffer, source.offset, 0);
                enc.setBuffer(self.buffer, destination, 1);
                enc.setValue(@as(u32, @intCast(count / 4)), 2);
                enc.dispatchThreads(.{ .width = count / 4 }, .{ .width = 256 });
                destination += count;
            }
        }
        // The .attention observation point is after the Engram block, so the kv the gate just consumed is still live.
        if (stage == .attention and model.weights.layers[@as(usize, layer)].engram != null) {
            const kv = ws.view(.engram_kv);
            enc.setPipeline(self.pipeline);
            enc.setBuffer(kv.buffer, kv.offset, 0);
            enc.setBuffer(self.engram_buffer, @as(usize, layer) * engram_slot, 1);
            enc.setValue(@as(u32, 25600 * 2 / 4), 2);
            enc.dispatchThreads(.{ .width = 25600 * 2 / 4 }, .{ .width = 256 });
        }
        if (stage == .moe) {
            const logits = ws.view(.router_logits);
            enc.setPipeline(self.pipeline);
            enc.setBuffer(logits.buffer, logits.offset, 0);
            enc.setBuffer(self.router_buffer, @as(usize, layer) * router_slot, 1);
            enc.setValue(@as(u32, 384), 2);
            enc.dispatchThreads(.{ .width = 384 }, .{ .width = 384 });
        }
        enc.barrier();
    }
    fn reference(ctx: *anyopaque, layer: usize, stage: Stage, chain: *anyopaque) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        // ROUND-377: the chain is generic over its precision now, so it cannot name itself in its own observer
        // callback; the reader casts the untyped pointer back to the source-contract chain.
        const ref: *const Reference = @ptrCast(@alignCast(chain));
        const start = (layer * 2 + @intFromBool(stage == .moe)) * stride;
        var destination: []u8 = self.cpu[start..][0..stride];
        inline for (.{ std.mem.sliceAsBytes(&ref.h), std.mem.sliceAsBytes(&ref.input), std.mem.sliceAsBytes(&ref.output), std.mem.sliceAsBytes(&ref.last_routes), std.mem.sliceAsBytes(&ref.last_routing) }, 0..) |source, field| {
            if (stage == .moe or field < 3) {
                @memcpy(destination[0..source.len], source);
                destination = destination[source.len..];
            }
        }
        if (stage == .attention and ref.w.layers[layer].engram != null) {
            const base = layer * engram_slot;
            @memcpy(self.engram_cpu[base..][0..51200], std.mem.sliceAsBytes(&ref.engram_state.kv));
            @memcpy(self.engram_cpu[base + 51200 ..][0..16], std.mem.sliceAsBytes(&ref.engram_state.gates));
        }
        if (stage == .moe)
            @memcpy(self.router_cpu[layer * router_slot ..][0..router_slot], std.mem.sliceAsBytes(&ref.route_logits));
    }
};
fn normalizedError(a: []const u8, b: []const u8) !f64 {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 2) {
        const x: f32 = @bitCast(@as(u32, std.mem.readInt(u16, a[i..][0..2], .little)) << 16);
        const y: f32 = @bitCast(@as(u32, std.mem.readInt(u16, b[i..][0..2], .little)) << 16);
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteTraceValues;
        squared += (@as(f64, x) - y) * (@as(f64, x) - y);
        denominator += @as(f64, y) * y;
    }
    return if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator);
}

/// Count differing bf16 elements between two byte ranges (not differing bytes).
fn bf16Mismatches(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i + 1 < a.len) : (i += 2) {
        if (std.mem.readInt(u16, a[i..][0..2], .little) != std.mem.readInt(u16, b[i..][0..2], .little)) n += 1;
    }
    return n;
}

fn f32BitsEqual(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Largest difference between two f32 encodings, counted in representable steps.
fn f32Ulps(a: []const u8, b: []const u8) u32 {
    var worst: u32 = 0;
    var i: usize = 0;
    while (i + 3 < a.len) : (i += 4) {
        const x = std.mem.readInt(u32, a[i..][0..4], .little);
        const y = std.mem.readInt(u32, b[i..][0..4], .little);
        const d = if (x > y) x - y else y - x;
        if (d > worst) worst = d;
    }
    return worst;
}

fn argmax(values: []const f32) u32 {
    var best: u32 = 0;
    var high = -std.math.inf(f32);
    for (values, 0..) |v, i| if (v > high) {
        high = v;
        best = @intCast(i);
    };
    return best;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    return std.mem.span(raw);
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !void {
    return runWith(gpa, io, model, ids, token_map, dir, false);
}

/// `tf-dsv41-run gate`: identical capture, ids pinned to the engine's own selection.
pub fn runGate(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8) !void {
    return runWith(gpa, io, model, ids, token_map, dir, true);
}

fn runWith(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8, force_pin: bool) !void {
    if (ids.len == 0 or ids.len > 16) return error.InvalidOracleWindow;
    // Diagnostic-only switches. None of them touches the engine; the order variants change the *reference's* own
    // reduction order, which is what an envelope of legal f32 orders is measured with.
    const pin_enabled = force_pin or env("TF_DSV41_PIN") != null;
    const pin_shift = env("TF_DSV41_PIN_SHIFT") != null;
    const form: reference_mod.ScoreForm =
        if (env("TF_DSV41_SCORE_FORM")) |f|
            if (std.mem.eql(u8, f, "kernel_lp")) .kernel_lp else if (std.mem.eql(u8, f, "stable")) .stable else .libm
        else
            .libm;
    const order: reference_mod.WeightOrder =
        if (env("TF_DSV41_WEIGHT_ORDER")) |o|
            if (std.mem.eql(u8, o, "pairwise")) .f32_pairwise else if (std.mem.eql(u8, o, "ascending")) .f32_ascending else .f64_sequential
        else
            .f64_sequential;
    if (env("TF_DSV41_FAULT_GATHER_ULP") != null) engram_math.State.Faults.gather_xor = 1;
    if (env("TF_DSV41_FAULT_GATHER_GROSS") != null) engram_math.State.Faults.gather_xor = 0x3C00;
    if (env("TF_DSV41_FAULT_GATE_SCALE")) |s| engram_math.State.Faults.gate_scale = std.fmt.parseFloat(f32, s) catch 1;
    const lib = try mtl.Library.fromSource(model.device, "#include <metal_stdlib>\nusing namespace metal;\nkernel void trace_copy(const device uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<n) y[i]=x[i]; }", .{});
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(model.device, lib, "trace_copy", false);
    defer pipeline.deinit();
    const buffer = try model.device.buffer(bytes, mtl.ResourceOptions.shared);
    defer buffer.deinit();
    const cpu = try gpa.alloc(u8, bytes);
    defer gpa.free(cpu);
    const engram_buffer = try model.device.buffer(engram_bytes, mtl.ResourceOptions.shared);
    defer engram_buffer.deinit();
    const engram_cpu = try gpa.alloc(u8, engram_bytes);
    defer gpa.free(engram_cpu);
    const router_buffer = try model.device.buffer(router_bytes, mtl.ResourceOptions.shared);
    defer router_buffer.deinit();
    const router_cpu = try gpa.alloc(u8, router_bytes);
    defer gpa.free(router_cpu);
    var capture = Capture{ .buffer = buffer, .pipeline = pipeline, .cpu = cpu, .engram_buffer = engram_buffer, .engram_cpu = engram_cpu, .router_buffer = router_buffer, .router_cpu = router_cpu };
    model.residency.add(buffer);
    model.residency.add(engram_buffer);
    model.residency.add(router_buffer);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        model.residency.remove(buffer);
        model.residency.remove(engram_buffer);
        model.residency.remove(router_buffer);
        model.residency.commit();
    }
    var reference = try Reference.init(gpa, model.cfg, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    // Same row source the serve uses, so a streamed model can be traced at all.
    var stream_rows: ?*engram_rows.Rows = null;
    if (std.c.getenv("TF_DSV41_ENGRAM_STREAM_DIR")) |d| {
        stream_rows = try engram_rows.Rows.open(gpa, io, std.mem.span(d), 0, 1, &.{ 1, 14 }, 4096);
        reference.attachRows(.{ .ptr = stream_rows.?, .rowFn = streamRow });
    }
    defer if (stream_rows) |r| r.close();
    defer reference.deinit();
    reference.weight_order = order;
    reference.score_form = form;
    reference.observer = .{ .ctx = &capture, .read = Capture.reference };
    const expected = try gpa.alloc(f32, ids.len * 129280);
    defer gpa.free(expected);
    try model.reset();
    for (ids, 0..) |id, i| @memcpy(expected[i * 129280 ..][0..129280], try model.forward(&.{id}, .last));
    try model.reset();
    const previous_observer = model.observer;
    model.observer = .{ .ctx = &capture, .encode = Capture.gpu };
    defer model.observer = previous_observer;
    var pin_ids: [40][6]u32 = undefined;
    var shift_ids: [40][6]u32 = undefined;
    for (&pin_ids) |*row| @memset(row, 0);
    for (&shift_ids) |*row| @memset(row, 0);
    var greedy: usize = 0;
    var engram_mismatch: [40]usize = @splat(0);
    var weight_ulp_max: u32 = 0;
    var weight_ulp_layer: ?usize = null;
    var weight_layers_differing: usize = 0;
    var weight_exact_logit_layers: usize = 0;
    var weight_exact_logit_ulp: u32 = 0;
    var weight_exact_logit_layer: ?usize = null;
    var natural_divergent: [40]bool = @splat(false);
    var natural_first: ?usize = null;
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "[\n");
    for (ids, 0..) |id, token| {
        @memset(cpu, 0);
        @memset(engram_cpu, 0);
        @memset(buffer.contents()[0..bytes], 0);
        @memset(engram_buffer.contents()[0..engram_bytes], 0);
        @memset(router_cpu, 0);
        @memset(router_buffer.contents()[0..router_bytes], 0);
        const actual = try model.forward(&.{id}, .last);
        for (actual, expected[token * 129280 ..][0..129280]) |a, b|
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return error.TraceChangedLogits;
        // The pin is the engine's own ids, harvested from the capture this forward pass just wrote: ids only, so the
        // reference still selects on its own logits and still normalises with its own arithmetic.
        const capture_base: [*]u8 = buffer.contents();
        for (0..40) |layer| {
            const at = (@as(usize, layer) * 2 + 1) * stride + 61440;
            const ptr: [*]const u32 = @ptrCast(@alignCast(capture_base + at));
            @memcpy(pin_ids[layer][0..], ptr[0..6]);
        }
        if (pin_enabled) {
            if (pin_shift) {
                for (0..40) |layer| shift_ids[layer] = pin_ids[(layer + 1) % 40];
                reference.pin = .{ .ids = &shift_ids };
            } else reference.pin = .{ .ids = &pin_ids };
        }
        const teacher = try reference.step(id);
        // End-to-end agreement, the number the project's own gate pairs with its 1% trip-wire.
        if (argmax(actual) != argmax(teacher)) greedy += 1;
        std.debug.print("token {d}: engine={d} reference={d} {s}\n", .{ token, argmax(actual), argmax(teacher), if (argmax(actual) == argmax(teacher)) "agree" else "DISAGREE" });
        const gpu = buffer.contents()[0..bytes];
        const engram_gpu = engram_buffer.contents()[0..engram_bytes];
        const router_gpu = router_buffer.contents()[0..router_bytes];
        for (0..80) |point| {
            const offset = point * stride;
            const g = gpu[offset..][0..stride];
            const c = cpu[offset..][0..stride];
            const stream_error = try normalizedError(g[0..40960], c[0..40960]);
            const input_error = try normalizedError(g[40960..51200], c[40960..51200]);
            const branch_error = try normalizedError(g[51200..61440], c[51200..61440]);
            const gr: [*]const u32 = @ptrCast(@alignCast(g.ptr + 61440));
            const cr: [*]const u32 = @ptrCast(@alignCast(c.ptr + 61440));
            const routes_match = point % 2 == 0 or std.mem.eql(u32, gr[0..6], cr[0..6]);
            const layer = point / 2;
            const engram_layer = model.weights.layers[layer].engram != null;
            const kv_error: f64 = if (engram_layer and point % 2 == 0)
                try normalizedError(engram_gpu[layer * engram_slot ..][0..51200], engram_cpu[layer * engram_slot ..][0..51200])
            else
                -1;
            const margin: f32 = if (point % 2 == 0) 0 else reference.route_margins[layer];
            if (force_pin) {
                if (engram_layer and point % 2 == 0)
                    engram_mismatch[layer] = bf16Mismatches(engram_gpu[layer * engram_slot ..][0..51200], engram_cpu[layer * engram_slot ..][0..51200]);
                if (point % 2 == 1) {
                    const ulps = f32Ulps(g[61464..61488], c[61464..61488]);
                    // The only place a ulp figure means anything: layers where both sides computed the same router
                    // input bits. Past the first divergence the reference is comparing different inputs.
                    if (f32BitsEqual(router_gpu[layer * router_slot ..][0..router_slot], router_cpu[layer * router_slot ..][0..router_slot])) {
                        weight_exact_logit_layers += 1;
                        if (ulps > weight_exact_logit_ulp) {
                            weight_exact_logit_ulp = ulps;
                            weight_exact_logit_layer = layer;
                        }
                    }
                    if (ulps > weight_ulp_max) {
                        weight_ulp_max = ulps;
                        weight_ulp_layer = layer;
                    }
                    if (ulps != 0) weight_layers_differing += 1;
                    if (!std.mem.eql(u32, &pin_ids[layer], &reference.natural_ids[layer])) {
                        natural_divergent[layer] = true;
                        if (natural_first == null) natural_first = layer;
                    }
                }
            }
            try report.print(gpa, "{s}{{\"token\":{d},\"layer\":{d},\"stage\":\"{s}\",\"stream_nrms\":{d},\"input_nrms\":{d},\"branch_nrms\":{d},\"engram_kv_nrms\":{d},\"route_margin\":{d},\"routes_match\":{s}}}", .{ if (token == 0 and point == 0) "" else ",\n", token, layer, if (point % 2 == 0) "attention" else "moe", stream_error, input_error, branch_error, kv_error, margin, if (routes_match) "true" else "false" });
            if (branch_error > 0.01 or !routes_match) std.debug.print("trace token={d} layer={d} stage={s} stream={d} input={d} branch={d} routes_match={}\n", .{ token, layer, if (point % 2 == 0) "attention" else "moe", stream_error, input_error, branch_error, routes_match });
        }
        inline for (.{ "gpu", "cpu" }, .{ gpu, cpu }) |name, data| {
            const path = try std.fmt.allocPrint(gpa, "{s}/trace-{d}-{s}.bin", .{ dir, token, name });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        }
        {
            const path = try std.fmt.allocPrint(gpa, "{s}/engram-{d}.bin", .{ dir, token });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = engram_gpu });
        }
        inline for (.{ "router-gpu", "router-cpu" }, .{ router_buffer.contents()[0..router_bytes], router_cpu }) |name, data| {
            const path = try std.fmt.allocPrint(gpa, "{s}/{s}-{d}.bin", .{ dir, name, token });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        }
        std.debug.print("trace captured token {d}/{d}, same_logits=true pin={} shift={} order={s}\n", .{ token + 1, ids.len, pin_enabled, pin_shift, @tagName(order) });
    }
    try report.appendSlice(gpa, "\n]\n");
    if (force_pin) {
        // Exactness-only verdict. No stage tolerance: a clean engine's per-stage divergence ranges 0 to 1.0e-02
        // depending on the token (receipt 2026-10-24-token-set-variance), so no constant separates correct from not.
        var engram_total: usize = 0;
        var engram_samples: usize = 0;
        for (0..40) |layer| {
            if (model.weights.layers[layer].engram == null) continue;
            engram_samples += 1;
            engram_total += engram_mismatch[layer];
        }
        var natural_count: usize = 0;
        for (natural_divergent) |d| natural_count += @intFromBool(d);
        const ok = engram_samples > 0 and engram_total == 0;
        std.debug.print(
            "gate: engram_layers={d} engram_bit_mismatches={d} layers_with_bit_equal_router_inputs={d} weight_ulp_max_where_router_inputs_equal={d} at_layer={?} weight_ulp_max_anywhere={d} natural_id_divergent_layers={d} first_natural_divergence={?} greedy_disagreements={d}/{d}\n",
            .{ engram_samples, engram_total, weight_exact_logit_layers, weight_exact_logit_ulp, weight_exact_logit_layer, weight_ulp_max, natural_count, natural_first, greedy, ids.len },
        );
        const verdict_json = try std.json.Stringify.valueAlloc(gpa, .{
            .mode = "gate",
            .verdict = if (ok) "PASS" else "FAIL",
            .basis = "exactness_only_no_stage_tolerance",
            .tokens = ids.len,
            .engram_layers = engram_samples,
            .engram_bit_mismatches = engram_total,
            .layers_with_bit_equal_router_inputs = weight_exact_logit_layers,
            .weight_ulp_max_where_router_inputs_equal = weight_exact_logit_ulp,
            .weight_ulp_layer = weight_exact_logit_layer,
            .weight_ulp_max_anywhere = weight_ulp_max,
            .natural_id_divergent_layers = natural_count,
            .first_natural_divergence = natural_first,
            .greedy_disagreements = greedy,
            .router_form = @tagName(form),
            .router_order = @tagName(order),
        }, .{});
        defer gpa.free(verdict_json);
        const vpath = try std.fs.path.join(gpa, &.{ dir, "gate-verdict.json" });
        defer gpa.free(vpath);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = vpath, .data = verdict_json });
        std.debug.print("GATE: {s} (exactness only; router spread and token agreement are reported, never thresholded)\n", .{if (ok) "PASS" else "FAIL"});
        if (!ok) return error.GateExactnessFailed;
    }
    std.debug.print("argmax: {d}/{d} agree ({d} disagreements) pin={} order={s} form={s}\n", .{ ids.len - greedy, ids.len, greedy, pin_enabled, @tagName(order), @tagName(form) });
    const path = try std.fs.path.join(gpa, &.{ dir, "layer-trace.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
}
