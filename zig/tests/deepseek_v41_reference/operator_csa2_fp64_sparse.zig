//! CSA2 sparse comparisons against the FP64 ideal.
const fixture = @import("csa2_fp64_fixture.zig");
const std = fixture.std;
const mtl = fixture.mtl;
const math = fixture.math;
const attention_math = fixture.attention_math;
const csa2 = fixture.csa2;
const width = fixture.width;
const ratio = fixture.ratio;
const deltaBf16 = fixture.deltaBf16;
const deltaF32VsF64 = fixture.deltaF32VsF64;
const printBf = fixture.printBf;
const printF32 = fixture.printF32;
const Lcg = fixture.Lcg;
const Buffers = fixture.Buffers;
const view = fixture.view;
const decodeWindowRow = fixture.decodeWindowRow;
const decodeMainRow = fixture.decodeMainRow;
const packWindowRow = fixture.packWindowRow;
const packMainRow = fixture.packMainRow;

pub const SparseSpec = struct {
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

pub fn sparseCase(
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
    var b = Buffers{ .device = device };
    defer b.deinit();
    const q_stride = spec.heads * width;
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

    // Decode the same cache as the kernel: window slots followed by compressed rows in selected-index order.
    var window_table = try std.testing.allocator.alloc(u16, 128 * width);
    defer std.testing.allocator.free(window_table);
    @memset(window_table, 0);
    // The reference table is indexed 128 + row, so it needs one row per compressed row.
    const compressed_cells = @max(compressed_rows.len, 1);
    var compressed_table = try std.testing.allocator.alloc(u16, compressed_cells * width);
    defer std.testing.allocator.free(compressed_table);
    @memset(compressed_table, 0);
    // Logical positions use ring[pos % 128], except rows published by this call, which use chunk.
    for (window_rows, 0..) |values, logical| {
        const bytes = try std.testing.allocator.alloc(u8, 528);
        defer std.testing.allocator.free(bytes);
        packWindowRow(values, bytes);
        const slot = logical % 128;
        @memcpy(ring_gpu.slice(u8, 128 * 528)[slot * 528 ..][0..528], bytes);
        const decoded = decodeWindowRow(bytes);
        @memcpy(window_table[slot * width ..][0..width], decoded[0..width]);
        if (logical >= spec.position and logical < spec.position + spec.rows)
            @memcpy(chunk_gpu.slice(u8, @max(spec.rows, 1) * 528)[(logical - spec.position) * 528 ..][0..528], bytes);
    }
    for (compressed_rows, 0..) |values, row| {
        const bytes = try std.testing.allocator.alloc(u8, 288);
        defer std.testing.allocator.free(bytes);
        packMainRow(values, bytes);
        @memcpy(main_gpu.slice(u8, @max(spec.compressed_rows, 1) * 288)[row * 288 ..][0..288], bytes);
        const decoded = decodeMainRow(bytes);
        @memcpy(compressed_table[row * width ..][0..width], decoded[0..width]);
    }

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q_gpu), view(ring_gpu), view(chunk_gpu), view(main_gpu), view(positions_gpu), view(sink_gpu), view(out_gpu), .{
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
    var wide32: [width]u16 = undefined;
    var wide64: [width]u16 = undefined;
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
            .window_stride = width,
            .compressed = compressed_table,
            .compressed_stride = width,
        };
        for (0..spec.heads) |head| {
            const head_q = q[row * q_stride + head * width ..][0..width];
            var pre32: [width]f32 = undefined;
            var pre64: [width]f64 = undefined;
            attention_math.sparseHeadPre(f32, head_q, keys, column_indices[0..count], sinks[head], wide32[0..width], &pre32);
            attention_math.sparseHeadPre(f64, head_q, keys, column_indices[0..count], sinks[head], wide64[0..width], &pre64);
            var ideal: [width]f64 = undefined;
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
            , .{ spec.name, "sparse", count, live, @sqrt(energy / @as(f64, width)), peak });
            var stage_buffer: [32]u8 = undefined;
            const stage = std.fmt.bufPrint(&stage_buffer, "row{d}_head{d}", .{ row, head }) catch unreachable;
            // Measure FP32-versus-FP64 rounding before final BF16 quantization hides it.
            printF32(spec.name, stage, "f32_vs_ideal_pre_rounding", deltaF32VsF64(&pre32, &pre64));
            printBf(spec.name, stage, "metal", deltaBf16(metal[row * q_stride + head * width ..][0..width], &ideal));
            printBf(spec.name, stage, "f32_reference", deltaBf16(wide32[0..width], &ideal));
            // Metal versus FP32 contract on identical decoded rows isolates kernel error.
            var contract_cells: [width]f64 = undefined;
            for (wide32, 0..) |cell, d| contract_cells[d] = math.f(cell);
            printBf(spec.name, stage, "metal_vs_f32_reference", deltaBf16(metal[row * q_stride + head * width ..][0..width], &contract_cells));
        }
    }
}

pub const KeyPattern = enum { random, cancelling, near_tie, one_sided };

pub fn sparseCaseFrom(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    spec: SparseSpec,
    rng: *Lcg,
    key_amplitude: f32,
    scale_exponent: i32,
    indices_mode: usize,
    sink: f32,
    pattern: KeyPattern,
) !void {
    try sparseCaseScaled(gpa, device, queue, p, spec, rng, key_amplitude, scale_exponent, scale_exponent, indices_mode, sink, pattern);
}

/// Independently scale window/compressed caches to exercise mixed-magnitude accumulation.
pub fn sparseCaseScaled(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    spec: SparseSpec,
    rng: *Lcg,
    key_amplitude: f32,
    window_exponent: i32,
    compressed_exponent: i32,
    indices_mode: usize,
    sink: f32,
    pattern: KeyPattern,
) !void {
    const q_stride = spec.heads * width;
    const q = try gpa.alloc(u16, spec.rows * q_stride);
    defer gpa.free(q);
    const q_scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(window_exponent)));
    for (q) |*cell| cell.* = math.b(rng.uniform() * q_scale);
    const logical_window = spec.position + spec.rows;
    const window_rows = try gpa.alloc([]const f32, logical_window);
    defer gpa.free(window_rows);
    for (window_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, width);
        fillPatternRow(cells, rng, i, key_amplitude, window_exponent, pattern);
        row.* = cells;
    }
    const compressed_rows = try gpa.alloc([]const f32, spec.compressed_rows);
    defer gpa.free(compressed_rows);
    for (compressed_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, width);
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

fn fillPatternRow(row: []f32, rng: *Lcg, index: usize, amplitude: f32, exponent: i32, pattern: KeyPattern) void {
    const scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)));
    if (pattern == .random) {
        for (row) |*v| v.* = rng.uniform() * amplitude * scale;
        return;
    }
    for (row, 0..) |*v, d| {
        if (pattern == .cancelling) {
            // Identically shaped keys tie all scores; the value sum cancels to the large terms' ULPs.
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

    var rng = Lcg{ .state = 0x2545f4914f6cdd1d };

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
    // Identical key shapes tie scores and make the P*V sum cancel to the terms' ULPs.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_values", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 0, 0, 0, KeyPattern.cancelling);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_wide", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 6, 0, 0, KeyPattern.cancelling);
    // Equal exponentials expose probability-rounding differences caused by accumulation order.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "score_tie", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 32, .topk = 32 }, &rng, 1.0, 0, 0, 0, KeyPattern.near_tie);
    // One dominant key against keys orders of magnitude smaller: denominator and accumulator run at the extreme ratio.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "one_dominant", .heads = 1, .rows = 1, .position = 300, .ratio = 2, .compressed_rows = 16, .topk = 16 }, &rng, 1.0, 0, 0, 0, KeyPattern.one_sided);
    // Many tiles, so the online correction multiplies the accumulator once per 64 columns.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "many_tiles", .heads = 1, .rows = 1, .position = 511, .ratio = 2, .compressed_rows = 512, .topk = 384 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Alternating-sign keys produce a P*V sum small relative to its terms.
    {
        const spec = SparseSpec{ .name = "pv_cancellation", .heads = 1, .rows = 1, .position = 64, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows) |*row| {
            const cells = try gpa.alloc(f32, width);
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
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        for (q, 0..) |*cell, d| cell.* = if ((d & 1) == 0) math.b(1.0) else math.b(-1.0);
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, width);
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
