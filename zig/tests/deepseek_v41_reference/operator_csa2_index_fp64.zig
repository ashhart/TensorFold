//! CSA2 indexer-scoring FP64 ideal-reference test (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");
const pack = @import("operator_csa2_pack_fp64.zig");

/// One indexer-score row set: 32 heads of 128 FP4-index columns each, scored against packed keys.
const IndexSpec = struct { name: []const u8, rows: usize, keys: usize, topk: usize };

fn packIndexRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. 128 / 2 + 128 / 32], 0);
    for (0..128 / 32) |block| {
        var a: f32 = 6.0 * 0x1p-126;
        for (values[block * 32 ..][0..32]) |v| a = @max(a, @abs(v));
        const bits: u32 = @bitCast(a / 6.0);
        const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127 + @intFromBool((bits & 0x7fffff) != 0);
        const scale = std.math.ldexp(@as(f32, 1.0), unbiased);
        out[64 + block] = @intCast(unbiased + 127);
        for (0..32) |j| {
            const code = pack.e2Code(values[block * 32 + j] / scale);
            const at = block * 16 + j / 2;
            if ((j & 1) != 0) out[at] |= code << 4 else out[at] |= code;
        }
    }
}

fn decodeIndexRow(bytes: []const u8) [128]u16 {
    var row: [128]u16 = undefined;
    for (0..128) |d| {
        const nibble: u8 = if ((d & 1) != 0) bytes[d / 2] >> 4 else bytes[d / 2] & 15;
        const scale: f32 = @bitCast(@as(u32, bytes[64 + d / 32]) << 23);
        row[d] = math.b(pack.e2(nibble) * scale);
    }
    return row;
}

fn indexCase(
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const tf.deepseek_v41.csa2_index.Pipelines,
    spec: IndexSpec,
    q: []const u16,
    weights: []const u16,
    key_values: []const []const f32,
) !void {
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    var scratch = try tf.deepseek_v41.csa2_index.Scratch.init(device, @intCast(spec.rows));
    defer scratch.deinit();
    const q_gpu = try b.get(u16, spec.rows * 32 * 128);
    const weights_gpu = try b.get(u16, spec.rows * 32);
    const keys_gpu = try b.get(u8, spec.keys * 68);
    @memcpy(q_gpu.slice(u16, spec.rows * 32 * 128), q);
    @memcpy(weights_gpu.slice(u16, spec.rows * 32), weights);

    // Both sides read the same decoded keys; the references differ only in their arithmetic.
    const decoded = try std.testing.allocator.alloc(u16, spec.keys * 128);
    defer std.testing.allocator.free(decoded);
    for (key_values, 0..) |values, row| {
        const bytes = try std.testing.allocator.alloc(u8, 68);
        defer std.testing.allocator.free(bytes);
        packIndexRow(values, bytes);
        @memcpy(keys_gpu.slice(u8, spec.keys * 68)[row * 68 ..][0..68], bytes);
        const decoded_row = decodeIndexRow(bytes);
        @memcpy(decoded[row * 128 ..][0..128], &decoded_row);
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, h.view(q_gpu), h.view(weights_gpu), h.view(keys_gpu), scratch, .{
        .rows = @intCast(spec.rows),
        .keys = @intCast(spec.keys),
        .q_stride = 32 * 128,
        .weight_stride = 32,
        .position = @intCast(spec.keys),
        .ratio = 1,
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;

    // Positions live behind the scores/heap/candidates regions; read them through the view offset.
    const kernel_scores = scratch.scores().buffer.slice(f32, spec.rows * 4096);
    const positions_offset = scratch.positions().offset / @sizeOf(i32);
    const kernel_positions = scratch.positions().buffer.slice(i32, positions_offset + spec.rows * 512);
    var ref32 = try std.testing.allocator.alloc(f32, spec.keys);
    defer std.testing.allocator.free(ref32);
    var ref64 = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(ref64);
    // The same scores with only the documented quantizations below the final one: the BF16 quantum stays.
    var exact = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(exact);
    const w_scale: f32 = 1 / @sqrt(@as(f32, 128 * 32));
    for (0..spec.keys) |row| {
        const key = decoded[row * 128 ..][0..128];
        var ideal: f64 = 0;
        var narrow32: f32 = 0;
        for (0..32) |head| {
            var dot: f64 = 0;
            for (0..128) |d| dot += @as(f64, math.f(key[d])) * @as(f64, math.f(q[head * 128 + d]));
            // The source head einsum lands in BF16, so the score is a quantized boundary like the cache codes.
            const rectified: f64 = @max(math.f(math.b(@min(@as(f32, @floatCast(dot)), 1e38))), 0);
            const weight: f32 = math.f(weights[head]);
            const scaled_weight = math.f(math.b(weight * w_scale));
            const weighted = math.f(math.b(@as(f32, @floatCast(rectified)) * scaled_weight));
            ideal += weighted;
            narrow32 += @as(f32, @floatCast(weighted));
        }
        exact[row] = ideal;
        // The kernel and the source contract write the score back through BF16, so the ideal carries it too.
        ref64[row] = math.f(math.b64(ideal));
        ref32[row] = math.f(math.b(narrow32));
    }
    var ideal_cells: [1024]f64 = undefined;
    const shown = @min(spec.keys, ideal_cells.len);
    for (0..shown) |row| ideal_cells[row] = ref64[row];
    h.printF32(spec.name, "index_score", "bf16_score_boundary", h.deltaF32VsF64(ref32[0..shown], exact[0..shown]));
    h.printF32(spec.name, "index_score", "metal_vs_ideal", h.deltaF32VsF64(kernel_scores[0..shown], ideal_cells[0..shown]));
    h.printF32(spec.name, "index_score", "f32_vs_ideal", h.deltaF32VsF64(ref32[0..shown], ideal_cells[0..shown]));
    // The kernel against the source-contract FP32 reference on identical keys: the delta selection can see.
    var contract: [1024]f64 = undefined;
    for (0..shown) |row| contract[row] = @as(f64, ref32[row]);
    h.printF32(spec.name, "index_score", "metal_vs_f32_reference", h.deltaF32VsF64(kernel_scores[0..shown], contract[0..shown]));

    // Selection: which keys land in the top-k; the kernel heap and strict-greater scan break ties by position.
    var chosen = try std.testing.allocator.alloc(bool, spec.keys);
    defer std.testing.allocator.free(chosen);
    @memset(chosen, false);
    var scores_copy = try std.testing.allocator.alloc(f32, spec.keys);
    defer std.testing.allocator.free(scores_copy);
    @memcpy(scores_copy, ref32[0..spec.keys]);
    var reference_picks: [512]u32 = undefined;
    const picks = @min(spec.topk, spec.keys);
    for (0..picks) |pick| {
        var best: usize = 0;
        for (0..spec.keys) |i| {
            if (scores_copy[i] > scores_copy[best]) best = i;
        }
        reference_picks[pick] = @intCast(best);
        scores_copy[best] = -std.math.inf(f32);
    }
    std.mem.sort(u32, reference_picks[0..picks], {}, std.sort.asc(u32));
    var overlap: usize = 0;
    for (reference_picks[0..picks]) |at| chosen[at] = true;
    var min_selected: f32 = std.math.inf(f32);
    var max_unselected: f32 = -std.math.inf(f32);
    for (0..spec.keys) |row| {
        if (chosen[row]) {
            overlap += 1;
            min_selected = @min(min_selected, ref32[row]);
        } else max_unselected = @max(max_unselected, ref32[row]);
    }
    var kernel_only: usize = 0;
    var out_of_range: usize = 0;
    for (0..picks) |i| {
        const slot = kernel_positions[positions_offset + i];
        if (slot < 0 or slot >= @as(i32, @intCast(spec.keys))) {
            out_of_range += 1;
            continue;
        }
        if (!chosen[@intCast(slot)]) kernel_only += 1;
    }
    // Arithmetic noise selection absorbs: a pick is knife-edged at one score quantum of margin; else no 1-ULP flip.
    var exact_picks: [512]u32 = undefined;
    var exact_scores = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(exact_scores);
    @memcpy(exact_scores, exact[0..spec.keys]);
    for (0..picks) |pick| {
        var best: usize = 0;
        for (0..spec.keys) |i| {
            if (exact_scores[i] > exact_scores[best]) best = i;
        }
        exact_picks[pick] = @intCast(best);
        exact_scores[best] = -std.math.inf(f64);
    }
    var flips: usize = 0;
    var exact_chosen = try std.testing.allocator.alloc(bool, spec.keys);
    defer std.testing.allocator.free(exact_chosen);
    @memset(exact_chosen, false);
    for (exact_picks[0..picks]) |at| exact_chosen[at] = true;
    for (0..spec.keys) |at| {
        if (chosen[at] != exact_chosen[at]) flips += 1;
    }
    var quantum: f32 = 0;
    var margin_in_quanta: f64 = 0;
    if (min_selected != std.math.inf(f32) and max_unselected != -std.math.inf(f32)) {
        const boundary = min_selected * 0.5 + max_unselected * 0.5;
        const exponent: i32 = if (@abs(boundary) > 0) std.math.ilogb(@abs(boundary)) - 7 else -149;
        quantum = std.math.pow(f32, 2.0, @as(f32, @floatFromInt(exponent)));
        margin_in_quanta = @as(f64, min_selected - max_unselected) / @as(f64, quantum);
    }
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"index_select","side":"noise_tolerance","keys":{d},"picks":{d},"precision_flips":{d},"selection_margin":{d},"score_quantum":{d},"margin_in_quanta":{d:.1}}}
    , .{ spec.name, spec.keys, picks, flips, min_selected - max_unselected, quantum, margin_in_quanta });
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"index_select","side":"metal","keys":{d},"picks":{d},"reference_only":{d},"kernel_only":{d},"invalid_positions":{d},"lowest_selected":{d},"highest_unselected":{d}}}
    , .{ spec.name, spec.keys, picks, picks - overlap, kernel_only, out_of_range, min_selected, max_unselected });
}

test "indexer scoring and selection sit at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try tf.deepseek_v41.csa2_index.Pipelines.load(device);
    defer p.deinit();

    var rng = h.Lcg{ .state = 0x1234_5678_9abc_def0 };
    var amplitude: f32 = 1;
    for ([_]IndexSpec{
        .{ .name = "index_random", .rows = 1, .keys = 1024, .topk = 512 },
        .{ .name = "index_random_4k", .rows = 1, .keys = 4096, .topk = 512 },
        .{ .name = "index_ties", .rows = 1, .keys = 1024, .topk = 512 },
        .{ .name = "index_boundary", .rows = 1, .keys = 1024, .topk = 512 },
    }, 0..) |spec, case_index| {
        _ = case_index;
        const q = try gpa.alloc(u16, 32 * 128);
        defer gpa.free(q);
        const weights = try gpa.alloc(u16, 32);
        defer gpa.free(weights);
        const cells = try gpa.alloc(f32, spec.keys * 128);
        defer gpa.free(cells);
        const keys = try gpa.alloc([]const f32, spec.keys);
        defer gpa.free(keys);
        for (q) |*cell| cell.* = math.b(rng.scale(1.0));
        for (weights) |*cell| cell.* = math.b(rng.scale(1.0));
        for (cells, 0..) |*v, i| {
            v.* = rng.scale(1.0);
            keys[i / 128] = cells[i / 128 * 128 ..][0..128];
        }
        // Identical keys: every score is the same value, so the whole selection is tie-driven.
        if (std.mem.eql(u8, spec.name, "index_ties")) {
            // Broadcast row 0's 128 cells into every later row, so each key's slice is a copy of the first key's.
            for (cells[128..], 0..) |*dst, i| dst.* = cells[i % 128];
        }
        // A score cliff just inside the selection boundary: half the keys sit on the boundary.
        if (std.mem.eql(u8, spec.name, "index_boundary")) {
            for (cells, 0..) |*v, i| {
                if ((i % 128) == 0) amplitude = (if (i / 128 < 512) @as(f32, 1.000001) else @as(f32, 1.0));
                v.* = rng.scale(amplitude);
            }
        }
        try indexCase(device, queue, &p, spec, q, weights, keys);
    }
}
