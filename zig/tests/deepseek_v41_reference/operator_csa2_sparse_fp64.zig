//! CSA2 sparse-attention FP64 ideal-reference tests (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const attention_math = @import("attention_math.zig");
const h = @import("operator_csa2_fp64.zig");
const pack = @import("operator_csa2_pack_fp64.zig");
const csa2 = tf.deepseek_v41.csa2;

const SparseSpec = struct {
    name: []const u8,
    heads: usize,
    rows: usize,
    position: usize,
    ratio: usize,
    compressed_rows: usize,
    topk: usize,
};

/// The kernel's own column addressing, so the reference reads the identical key per column slot.
fn columnKey(spec: SparseSpec, row: usize, slot: usize, indices: []const i32) ?KeySource {
    const query = spec.position + row;
    const window_columns: usize = if (spec.position + row == 0) 1 else 128;
    if (slot < window_columns) {
        const pos: usize = if (query == 0) 0 else if (query + 1 < 128) (query + 1 + slot) % 128 else blk: {
            const begin = if (query + 1 > 128) query + 1 - 128 else 0;
            break :blk begin + slot;
        };
        if (pos > query) return null;
        return .{ .row = pos % 128 };
    }
    if (spec.ratio == 0) return null;
    const pos_i = indices[row * spec.topk + slot - window_columns];
    const limit = @min(spec.compressed_rows, (query + 1) / spec.ratio);
    if (pos_i < 0 or @as(usize, @intCast(pos_i)) >= limit) return null;
    return .{ .compressed = true, .row = @intCast(pos_i) };
}

const KeySource = struct { compressed: bool = false, row: usize };

fn sparseCase(
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    spec: SparseSpec,
    q: []const u16,
    window_rows: []const []const f32,
    compressed_rows: []const []const f32,
    indices: []const i32,
    sinks: []const f32,
) !void {
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const q_stride = spec.heads * h.width;
    const q_gpu = try b.get(u16, spec.rows * q_stride);
    const ring_gpu = try b.get(u8, 128 * 528);
    const chunk_gpu = try b.get(u8, @max(spec.rows, 1) * 528);
    const main_gpu = try b.get(u8, @max(spec.compressed_rows, 1) * 288);
    const positions_gpu = try b.get(i32, spec.rows * spec.topk);
    const sink_gpu = try b.get(f32, spec.heads);
    const out_gpu = try b.get(u16, spec.rows * q_stride);
    @memcpy(q_gpu.slice(u16, spec.rows * q_stride), q);
    @memcpy(positions_gpu.slice(i32, spec.rows * spec.topk), indices);
    @memcpy(sink_gpu.slice(f32, spec.heads), sinks);

    // The reference key table is the kernel's decoded cache: window columns by slot, then selected compressed rows.
    var window_table = try std.testing.allocator.alloc(u16, 128 * h.width);
    defer std.testing.allocator.free(window_table);
    @memset(window_table, 0);
    // The reference table is indexed 128 + row, so it needs one row per compressed row.
    const compressed_cells = @max(compressed_rows.len, 1);
    var compressed_table = try std.testing.allocator.alloc(u16, compressed_cells * h.width);
    defer std.testing.allocator.free(compressed_table);
    @memset(compressed_table, 0);
    // Rows are ring slots: a logical position reads ring[pos % 128] unless this call publishes it.
    for (window_rows, 0..) |values, logical| {
        const bytes = try std.testing.allocator.alloc(u8, 528);
        defer std.testing.allocator.free(bytes);
        pack.packWindowRow(values, bytes);
        const slot = logical % 128;
        @memcpy(ring_gpu.slice(u8, 128 * 528)[slot * 528 ..][0..528], bytes);
        const decoded = pack.decodeWindowRow(bytes);
        @memcpy(window_table[slot * h.width ..][0..h.width], decoded[0..h.width]);
        if (logical >= spec.position and logical < spec.position + spec.rows)
            @memcpy(chunk_gpu.slice(u8, @max(spec.rows, 1) * 528)[(logical - spec.position) * 528 ..][0..528], bytes);
    }
    for (compressed_rows, 0..) |values, row| {
        const bytes = try std.testing.allocator.alloc(u8, 288);
        defer std.testing.allocator.free(bytes);
        pack.packMainRow(values, bytes);
        @memcpy(main_gpu.slice(u8, @max(spec.compressed_rows, 1) * 288)[row * 288 ..][0..288], bytes);
        const decoded = pack.decodeMainRow(bytes);
        @memcpy(compressed_table[row * h.width ..][0..h.width], decoded[0..h.width]);
    }

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, h.view(q_gpu), h.view(ring_gpu), h.view(chunk_gpu), h.view(main_gpu), h.view(positions_gpu), h.view(sink_gpu), h.view(out_gpu), .{
        .rows = @intCast(spec.rows),
        .heads = @intCast(spec.heads),
        .q_stride = @intCast(q_stride),
        .position = @intCast(spec.position),
        .chunk_rows = @intCast(spec.rows),
        .ratio = @intCast(spec.ratio),
        .compressed_rows = @intCast(spec.compressed_rows),
        .topk = @intCast(spec.topk),
        .index_stride = @intCast(spec.topk),
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;
    const metal = out_gpu.slice(u16, spec.rows * q_stride);

    var column_indices = try std.testing.allocator.alloc(i32, 640);
    defer std.testing.allocator.free(column_indices);
    var wide32: [h.width]u16 = undefined;
    var wide64: [h.width]u16 = undefined;
    for (0..spec.rows) |row| {
        const window_columns: usize = if (spec.position + row == 0) 1 else 128;
        const compressed_columns: usize = if (spec.ratio == 0) 0 else @min(spec.topk, @min(spec.compressed_rows, (spec.position + row + 1) / spec.ratio));
        const count = window_columns + compressed_columns;
        for (0..count) |slot| {
            if (columnKey(spec, row, slot, indices)) |key| {
                column_indices[slot] = if (key.compressed) @intCast(128 + key.row) else @intCast(key.row);
            } else column_indices[slot] = -1;
        }
        const keys = attention_math.KeyTable{
            .window = window_table,
            .window_stride = h.width,
            .compressed = compressed_table,
            .compressed_stride = h.width,
        };
        for (0..spec.heads) |head| {
            const head_q = q[row * q_stride + head * h.width ..][0..h.width];
            var pre32: [h.width]f32 = undefined;
            var pre64: [h.width]f64 = undefined;
            attention_math.sparseHeadPre(f32, head_q, keys, column_indices[0..count], sinks[head], wide32[0..h.width], &pre32);
            attention_math.sparseHeadPre(f64, head_q, keys, column_indices[0..count], sinks[head], wide64[0..h.width], &pre64);
            var ideal: [h.width]f64 = undefined;
            for (wide64, 0..) |cell, d| ideal[d] = math.f(cell);
            var energy: f64 = 0;
            var peak: f64 = 0;
            var live: usize = 0;
            for (column_indices[0..count]) |at| live += @intFromBool(at >= 0);
            for (ideal) |v| {
                energy += v * v;
                peak = @max(peak, @abs(v));
            }
            std.debug.print(
                \\CSA64 {{"case":"{s}","stage":"scale","side":"{s}","columns":{d},"live":{d},"ideal_rms":{d},"ideal_peak":{d}}}
            , .{ spec.name, "sparse", count, live, @sqrt(energy / @as(f64, h.width)), peak });
            var stage_buffer: [32]u8 = undefined;
            const stage = std.fmt.bufPrint(&stage_buffer, "row{d}_head{d}", .{ row, head }) catch unreachable;
            // The intrinsic FP32-versus-FP64 level of the stage, before the BF16 quantum hides it; verdicts use it.
            h.printF32(spec.name, stage, "f32_vs_ideal_pre_rounding", h.deltaF32VsF64(&pre32, &pre64));
            h.printBf(spec.name, stage, "metal", h.deltaBf16(metal[row * q_stride + head * h.width ..][0..h.width], &ideal));
            h.printBf(spec.name, stage, "f32_reference", h.deltaBf16(wide32[0..h.width], &ideal));
            // Decisive for the verdict: kernel and source-contract reference on identical decoded rows.
            var contract_cells: [h.width]f64 = undefined;
            for (wide32, 0..) |cell, d| contract_cells[d] = math.f(cell);
            h.printBf(spec.name, stage, "metal_vs_f32_reference", h.deltaBf16(metal[row * q_stride + head * h.width ..][0..h.width], &contract_cells));
        }
    }
}

const KeyPattern = enum { random, cancelling, near_tie, one_sided };

fn sparseCaseFrom(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    spec: SparseSpec,
    rng: *h.Lcg,
    key_amplitude: f32,
    scale_exponent: i32,
    indices_mode: usize,
    sink: f32,
    pattern: KeyPattern,
) !void {
    try sparseCaseScaled(gpa, device, queue, p, spec, rng, key_amplitude, scale_exponent, scale_exponent, indices_mode, sink, pattern);
}

/// Rows with window and compressed cache on independent scale grids: one accumulator mixes large and tiny terms.
fn sparseCaseScaled(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    spec: SparseSpec,
    rng: *h.Lcg,
    key_amplitude: f32,
    window_exponent: i32,
    compressed_exponent: i32,
    indices_mode: usize,
    sink: f32,
    pattern: KeyPattern,
) !void {
    const q_stride = spec.heads * h.width;
    const q = try gpa.alloc(u16, spec.rows * q_stride);
    defer gpa.free(q);
    const q_scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(window_exponent)));
    for (q) |*cell| cell.* = math.b(rng.uniform() * q_scale);
    const logical_window = spec.position + spec.rows;
    const window_rows = try gpa.alloc([]const f32, logical_window);
    defer gpa.free(window_rows);
    for (window_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, h.width);
        fillPatternRow(cells, rng, i, key_amplitude, window_exponent, pattern);
        row.* = cells;
    }
    const compressed_rows = try gpa.alloc([]const f32, spec.compressed_rows);
    defer gpa.free(compressed_rows);
    for (compressed_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, h.width);
        fillPatternRow(cells, rng, i, key_amplitude, compressed_exponent, pattern);
        row.* = cells;
    }
    const indices = try gpa.alloc(i32, spec.rows * spec.topk);
    defer gpa.free(indices);
    const limit = @max(spec.compressed_rows, 1);
    for (0..spec.rows) |r| for (0..spec.topk) |k| {
        indices[r * spec.topk + k] = switch (indices_mode) {
            0 => @intCast(k * limit / @max(spec.topk, 1)), // spread and ascending
            1 => @intCast((limit - 1) - k), // descending (the reference sorts; the kernel does not)
            else => @as(i32, @intCast((r *% 7 +% 13 * k) % limit)), // arbitrary order, unsorted
        };
    };
    const sinks = try gpa.alloc(f32, spec.heads);
    defer gpa.free(sinks);
    for (sinks) |*v| v.* = sink;
    try sparseCase(device, queue, p, spec, q, window_rows, compressed_rows, indices, sinks);
}

fn fillPatternRow(row: []f32, rng: *h.Lcg, index: usize, amplitude: f32, exponent: i32, pattern: KeyPattern) void {
    const scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)));
    if (pattern == .random) {
        for (row) |*v| v.* = rng.uniform() * amplitude * scale;
        return;
    }
    for (row, 0..) |*v, d| {
        if (pattern == .cancelling) {
            // Same key shape: every score ties and the value sum cancels to the ULPs of the terms.
            const flip: f32 = if ((d & 1) == 0) 1 else -1;
            v.* = flip * amplitude * scale + rng.uniform() * amplitude * scale * 1e-3;
        } else if (pattern == .near_tie) {
            // Keys differ only far above the score rounding boundary: probabilities tie at the top.
            v.* = @as(f32, @floatFromInt(index)) * amplitude * scale + rng.uniform() * amplitude * scale * 1e-7;
        } else {
            // One dominant key against keys orders of magnitude smaller.
            v.* = rng.uniform() * amplitude * scale * @as(f32, if (index == 0) @as(f32, 1e4) else @as(f32, 1e-4));
        }
    }
}

test "sparse attention stages sit at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();

    var rng = h.Lcg{ .state = 0x2545f4914f6cdd1d };

    // Window only, the decode-step boundary at a wrapped ring.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "window_decode", .heads = 4, .rows = 1, .position = 255, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Compressed rows with a small selection, ascending indices.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "compressed_asc", .heads = 4, .rows = 1, .position = 255, .ratio = 2, .compressed_rows = 128, .topk = 16 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Unsorted indices: the kernel consumes the array in the given order; the reference must too.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "compressed_unsorted", .heads = 2, .rows = 1, .position = 9, .ratio = 2, .compressed_rows = 40, .topk = 12 }, &rng, 1.0, 0, 2, 0, KeyPattern.random);
    // Prefill rows, multi-row chunk with a wrapped ring.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "prefill_rows", .heads = 2, .rows = 4, .position = 250, .ratio = 2, .compressed_rows = 100, .topk = 8 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Wide dynamic range keys: the score sum runs across many orders of magnitude.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "wide_range", .heads = 1, .rows = 1, .position = 200, .ratio = 2, .compressed_rows = 32, .topk = 8 }, &rng, 1.0, -20, 0, 0, KeyPattern.random);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "wide_range_hi", .heads = 1, .rows = 1, .position = 200, .ratio = 2, .compressed_rows = 32, .topk = 8 }, &rng, 1.0, 20, 0, 0, KeyPattern.random);
    // Denormal-scale sink: the sink mass underflows FP32 while it is exact in FP64.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "sink_denormal", .heads = 1, .rows = 1, .position = 100, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, -16.0, KeyPattern.random);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "sink_large", .heads = 1, .rows = 1, .position = 100, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 12.0, KeyPattern.random);
    // Value-product cancellation: identical key shapes tie every score and the P.V sum cancels to the ULPs.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_values", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 0, 0, 0, KeyPattern.cancelling);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_wide", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 6, 0, 0, KeyPattern.cancelling);
    // Tied scores: the denominator sums equal exponentials; accumulation order picks each probability's side.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "score_tie", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 32, .topk = 32 }, &rng, 1.0, 0, 0, 0, KeyPattern.near_tie);
    // One dominant key against keys orders of magnitude smaller: denominator and accumulator at the ratio.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "one_dominant", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 16, .topk = 16 }, &rng, 1.0, 0, 0, 0, KeyPattern.one_sided);
    // Many tiles, so the online correction multiplies the accumulator once per 64 columns.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "many_tiles", .heads = 1, .rows = 1, .position = 511, .ratio = 2, .compressed_rows = 512, .topk = 384 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Cancellation in the value product: alternating-sign keys make the P.V sum small next to its terms.
    {
        const spec = SparseSpec{ .name = "pv_cancellation", .heads = 1, .rows = 1, .position = 64, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, h.width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows) |*row| {
            const cells = try gpa.alloc(f32, h.width);
            for (cells, 0..) |*v, d| v.* = if ((d & 1) == 0) rng.scale(64.0) else rng.scale(64.0) * 0.987654321;
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &p, spec, q, window_rows, empty, indices, sinks);
    }
    // Large cancelling keys beside small ones: score dot products run through near-cancelling terms.
    {
        const spec = SparseSpec{ .name = "score_cancellation", .heads = 1, .rows = 1, .position = 96, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, h.width);
        defer gpa.free(q);
        for (q, 0..) |*cell, d| cell.* = if ((d & 1) == 0) math.b(1.0) else math.b(-1.0);
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, h.width);
            const bias: f32 = @as(f32, @floatFromInt(@rem(@as(i32, @intCast(i)), 8))) * 4.0;
            for (cells, 0..) |*v, d| v.* = bias + (if ((d & 1) == 0) rng.scale(0.001) else rng.scale(0.001) * 0.99999);
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &p, spec, q, window_rows, empty, indices, sinks);
    }
}

/// One live column pins the maximum, so sink - max is the whole normalization across the FP32 exp limits.
fn sinkCase(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    name: []const u8,
    rng: *h.Lcg,
    sink: f32,
) !void {
    const spec = SparseSpec{ .name = name, .heads = 1, .rows = 1, .position = 0, .ratio = 0, .compressed_rows = 0, .topk = 0 };
    const q = try gpa.alloc(u16, h.width);
    defer gpa.free(q);
    for (q) |*cell| cell.* = math.b(rng.scale(1.0));
    const window_rows = try gpa.alloc([]const f32, 1);
    defer gpa.free(window_rows);
    const cells = try gpa.alloc(f32, h.width);
    defer gpa.free(cells);
    for (cells) |*v| v.* = rng.scale(1.0);
    window_rows[0] = cells;
    const sinks = try gpa.alloc(f32, 1);
    defer gpa.free(sinks);
    sinks[0] = sink;
    const empty = try gpa.alloc([]const f32, 0);
    defer gpa.free(empty);
    const indices = try gpa.alloc(i32, 0);
    defer gpa.free(indices);
    try sparseCase(device, queue, p, spec, q, window_rows, empty, indices, sinks);
}

test "sink normalization sits at rounding level across the exp overflow boundaries" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = h.Lcg{ .state = 0x51c0de51c0de51c0 };

    // One case per FP32 exp() boundary: underflow, overflow edge and the balanced middle where the sink decides.
    const sweep = [_]struct { sink: f32, name: []const u8 }{
        .{ .sink = -104, .name = "sink_underflow" },
        .{ .sink = -90, .name = "sink_near_underflow" },
        .{ .sink = -60, .name = "sink_tiny" },
        .{ .sink = -12, .name = "sink_small" },
        .{ .sink = -1, .name = "sink_minus_one" },
        .{ .sink = 0, .name = "sink_zero" },
        .{ .sink = 1, .name = "sink_one" },
        .{ .sink = 12, .name = "sink_medium" },
        .{ .sink = 60, .name = "sink_large" },
        .{ .sink = 80, .name = "sink_edge80" },
        .{ .sink = 83, .name = "sink_edge83" },
        .{ .sink = 84, .name = "sink_edge84" },
        .{ .sink = 85, .name = "sink_edge85" },
        .{ .sink = 86, .name = "sink_over_edge" },
        .{ .sink = 90, .name = "sink_overflow" },
    };
    for (sweep) |entry| try sinkCase(gpa, device, queue, &pipelines, entry.name, &rng, entry.sink);

    // Per-head sinks: the kernel indexes the sink array by head, so heads orders apart check the indexing.
    {
        const spec = SparseSpec{ .name = "sink_per_head", .heads = 4, .rows = 1, .position = 0, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        var head_rng = h.Lcg{ .state = 0xabcdef0123456789 };
        const q = try gpa.alloc(u16, spec.heads * h.width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(head_rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, 1);
        defer gpa.free(window_rows);
        const cells = try gpa.alloc(f32, h.width);
        defer gpa.free(cells);
        for (cells) |*v| v.* = head_rng.scale(1.0);
        window_rows[0] = cells;
        const sinks = try gpa.alloc(f32, spec.heads);
        defer gpa.free(sinks);
        for (0..spec.heads) |head| sinks[head] = @as(f32, @floatFromInt(@as(i32, @intCast(head)) * 24 - 36));
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}

// Tile denominators on flat, ramped and descending scores: the ramp drives the correction through FP32 underflow.
test "tile denominators sit at rounding level for every tile shape" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = h.Lcg{ .state = 0x1a2b3c4d5e6f7081 };

    // Ten tiles: 639 exponentials and 639 products; identical keys tie every score and cancel the sum.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_equal_scores", .heads = 1, .rows = 1, .position = 1022, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 0, 0, KeyPattern.cancelling);
    // Ascending scores by tile: the correction falls one power of two per step until FP32 flushes it.
    for ([_]i32{ -2, 0, 1, 2 }) |step| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "tiles_ramp{d}", .{step}) catch unreachable;
        try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = label, .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, step, 0, 0, KeyPattern.near_tie);
    }
    // Descending scores: the running maximum never moves, so every tile lands at full weight.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_descending", .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 1, 0, KeyPattern.near_tie);
    // One tile only, the boundary case of the online loop: the correction is applied zero times.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_single", .heads = 1, .rows = 1, .position = 20, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
}

// Output mix: window and compressed rows on different scale grids, and mirror rows whose shared shape cancels.
test "the output mix sits at rounding level when window and compressed rows cancel" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = h.Lcg{ .state = 0x77a1c0de_5b5e1a2b };

    // Large window rows against tiny compressed rows, then the reverse: one accumulator six orders apart.
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_window_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 14, -12, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_compressed_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, -12, 14, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_balanced_magnitudes", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 6, 6, 0, 0, KeyPattern.cancelling);

    // Amplification probe: 128 window rows of a large alternating shape that cancels from scores and sum alike.
    for ([_]i32{ 8, 16, 23 }) |shape_exponent| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "mirror_shape_e{d}", .{shape_exponent}) catch unreachable;
        const spec = SparseSpec{ .name = label, .heads = 1, .rows = 1, .position = 127, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, h.width);
        defer gpa.free(q);
        // q dot shape == 0: a uniform query against a sign-alternating shape, so the large part drops out.
        for (q) |*cell| cell.* = math.b(1.0);
        const window_rows = try gpa.alloc([]const f32, 128);
        defer gpa.free(window_rows);
        const shape = std.math.pow(f32, 2, @floatFromInt(shape_exponent));
        const shared = try gpa.alloc(f32, h.width);
        defer gpa.free(shared);
        // One 64th of the cancelling shape near the FP8 window floor: partial sums run 2^16 times the final.
        for (shared) |*v| v.* = rng.scale(shape * 0x1p-16);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, h.width);
            const side: f32 = if ((i & 1) == 0) 1 else -1;
            for (0..h.width) |d| {
                const flip: f32 = if ((d & 1) == 0) 1 else -1;
                cells[d] = side * flip * shape + shared[d];
            }
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}
