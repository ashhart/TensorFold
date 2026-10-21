//! Engram hash, FP8 gather and signed-sqrt gate at the FP32 contract and the FP64 ideal; receipts print "ENG64".
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");

const engram = tf.deepseek_v41_engram;
const hash = tf.deepseek_v41_engram_hash;
const contract = @import("contract").spec;
const View = engram.View;

const dim: usize = 5120;
const copies: usize = 4;
const hidden_width: usize = dim * copies;
const kv_width: usize = dim * (copies + 1);
const gather_dim: usize = engram.gather_dim;
const head_dim: usize = engram.head_dim;
const eps: f32 = 1e-6;
const raw_ids: usize = 64;

/// Synthetic token map and history so the hash case needs no checkpoint bytes.
const hash_tokens: usize = 10;
const hash_chunk: usize = 8;

/// One gather case covers every column of the real table layout with fewer rows.
const gather_tokens: usize = 4;
const table_rows: usize = 128;

/// Two gate rows: the second carries a DEAD mask so the zero-gate path is checked exactly.
const gate_tokens: usize = 2;

const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    fn next(self: *Lcg) u64 {
        self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
        return self.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    fn uniform(self: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(self.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
    fn below(self: *Lcg, limit: usize) usize {
        return @intCast(self.next() % limit);
    }
};

const Delta = struct {
    cells: usize = 0,
    unequal: usize = 0,
    nrms: f64 = 0,
    max_abs: f64 = 0,
    worst: usize = 0,
};

/// The cells' BF16 disagreement with an FP64 ideal; a non-finite value on either side is a failure, never a skip.
fn deltaBf16(actual: []const u16, ideal: []const u16) !Delta {
    var d: Delta = .{ .cells = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |a, want, i| {
        const ref: f64 = math.d64(want);
        const got: f64 = math.d64(a);
        if (!std.math.isFinite(ref) or !std.math.isFinite(got)) return error.NonFiniteMetric;
        magnitude += ref * ref;
        if (a == want) continue;
        const diff = got - ref;
        if (!std.math.isFinite(diff)) return error.NonFiniteMetric;
        d.unequal += 1;
        if (@abs(diff) > d.max_abs) {
            d.max_abs = @abs(diff);
            d.worst = i;
        }
        squared += diff * diff;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

fn printBf16(stage: []const u8, case_name: []const u8, d: Delta) void {
    std.debug.print("ENG64 {{\"stage\":\"{s}\",\"case\":\"{s}\",\"cells\":{d},\"bf16_unequal\":{d},\"nrms\":{d},\"max_abs\":{d},\"worst\":{d}}}\n", .{ stage, case_name, d.cells, d.unequal, d.nrms, d.max_abs, d.worst });
}

fn printEq(stage: []const u8, case_name: []const u8, cells: usize, unequal: usize) void {
    std.debug.print("ENG64 {{\"stage\":\"{s}\",\"case\":\"{s}\",\"cells\":{d},\"metal_vs_contract_unequal\":{d}}}\n", .{ stage, case_name, cells, unequal });
}

/// The contract gate's per-cell verdict: how many cells passed their declared FP64 bound, and the closest call.
fn printBound(stage: []const u8, case_name: []const u8, cells: usize, failures: usize, max_ratio: f64) void {
    std.debug.print("ENG64 {{\"stage\":\"{s}\",\"case\":\"{s}\",\"cells\":{d},\"bound_failures\":{d},\"bound_max_ratio\":{d}}}\n", .{ stage, case_name, cells, failures, max_ratio });
}

fn printCell(check: []const u8, family: []const u8, token: usize, copy: usize, cell: usize, got: u16, oracle: u16, ideal: f64, diff: f64, allowed: f64) void {
    std.debug.print("ENG64_CELL {{\"stage\":\"gate\",\"check\":\"{s}\",\"case\":\"{s}\",\"token\":{d},\"copy\":{d},\"cell\":{d},\"got\":{d},\"oracle\":{d},\"ideal\":{d},\"diff\":{d},\"bound\":{d}}}\n", .{ check, family, token, copy, cell, got, oracle, ideal, diff, allowed });
}

fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

fn completed(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.EngramGpuFailure;
}

/// The pinned CPU oracle's gate: wide sums, the FP32 mean projection, then FP32 rstd, dot and gate.
fn contractProduct(value: f32, scale: f32) u16 {
    const product = value * scale;
    if (@abs(scale) < 0x1p-126 or @abs(product) < 0x1p-126) return 0;
    return math.b(product);
}

fn contractGate(hidden: []const u16, kv: []const u16, q: []const f32, k: []const f32, copy: usize, out: []u16) void {
    @setFloatMode(.strict);
    var hh: f64 = 0;
    var kk: f64 = 0;
    var dot: f64 = 0;
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.f(hidden[at]);
        const x = math.f(kv[at]);
        const weight = q[at] * k[at];
        hh += @as(f64, h) * @as(f64, h);
        kk += @as(f64, x) * @as(f64, x);
        dot += @as(f64, (h * weight) * x);
    }
    const h_inverse = math.idealRsqrt(math.meanFromWide(hh, dim) + eps);
    const k_inverse = math.idealRsqrt(math.meanFromWide(kk, dim) + eps);
    const rstd = h_inverse * k_inverse;
    const dot_wide: f32 = @floatCast(dot);
    const normalized = (dot_wide * rstd) * (1 / @sqrt(@as(f32, 5120.0)));
    const root = std.math.copysign(@sqrt(@max(@abs(normalized), @as(f32, 1e-6))), normalized);
    const gate = math.sigmoid(root);
    for (0..dim) |d| {
        const at = copy * dim + d;
        out[d] = math.b(math.f(hidden[at]) + gate * math.f(kv[hidden_width + d]));
    }
}

/// The same equations widened to FP64, never an emulation of the Metal kernel.
fn idealGate(hidden: []const u16, kv: []const u16, q: []const f32, k: []const f32, copy: usize, out: []u16) void {
    @setFloatMode(.strict);
    var hh: f64 = 0;
    var kk: f64 = 0;
    var dot: f64 = 0;
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.d64(hidden[at]);
        const x = math.d64(kv[at]);
        const weight = @as(f64, q[at]) * @as(f64, k[at]);
        hh += h * h;
        kk += x * x;
        dot += (h * weight) * x;
    }
    const h_inverse = 1 / @sqrt(hh / 5120.0 + @as(f64, eps));
    const k_inverse = 1 / @sqrt(kk / 5120.0 + @as(f64, eps));
    const normalized = (dot * (h_inverse * k_inverse)) / @sqrt(@as(f64, 5120.0));
    const root = std.math.copysign(@sqrt(@max(@abs(normalized), @as(f64, 1e-6))), normalized);
    const gate = 1 / (1 + @exp(-root));
    for (0..dim) |d| {
        const at = copy * dim + d;
        out[d] = math.b64(math.d64(hidden[at]) + gate * math.d64(kv[hidden_width + d]));
    }
}

/// The gate's kernel-contract registration: FP32 sums over the 5120-wide keys and a BF16 output.
const gate_shape = contract.Shape{ .rows = gate_tokens, .n = hidden_width, .k = dim };

/// The sums' longest chain: a thread's serial adds over the keys, the two simd_sum trees, the term's products.
fn gateDepth(shape: contract.Shape) usize {
    return (shape.k + 255) / 256 + 8 + 3;
}

/// The fp32 roundings after the sums: means, rsqrts, dot scales, sqrt, exp, sigmoid, gate, add.
const gate_spec = contract.Spec{
    .name = "engram_gate",
    .accum = .f32,
    .depth = gateDepth,
    .ops = 18,
    .out = .bf16,
    .served = &.{gate_shape},
};

/// A cell's FP64 value and the magnitude its sums and gate can move it by, in the contract's bound.
const GateWide = struct { want: [dim]f64, mag: [dim]f64 };

/// idealGate's values per cell plus the magnitude the sums and the gate can move them by, for contract.bound.
fn idealGateWide(hidden: []const u16, kv: []const u16, q: []const f32, k: []const f32, copy: usize, wide: *GateWide) void {
    @setFloatMode(.strict);
    var hh: f64 = 0;
    var kk: f64 = 0;
    var dot: f64 = 0;
    var dot_mag: f64 = 0;
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.d64(hidden[at]);
        const x = math.d64(kv[at]);
        const weight = @as(f64, q[at]) * @as(f64, k[at]);
        hh += h * h;
        kk += x * x;
        dot += (h * weight) * x;
        dot_mag += @abs((h * weight) * x);
    }
    const mean_h = hh / @as(f64, dim) + @as(f64, eps);
    const mean_k = kk / @as(f64, dim) + @as(f64, eps);
    const rstd = 1 / @sqrt(mean_h) / @sqrt(mean_k);
    const dot_n = (dot * rstd) / @sqrt(@as(f64, dim));
    const abs_dot = @max(@abs(dot_n), 1e-6);
    const root = std.math.copysign(@sqrt(abs_dot), dot_n);
    const gate = 1 / (1 + @exp(-root));
    const slope = if (@abs(dot_n) > 1e-6) 0.25 / (2 * @sqrt(abs_dot)) else 0;
    const spread = (rstd / @sqrt(@as(f64, dim))) * dot_mag + @abs(dot_n);
    for (0..dim) |d| {
        const at = copy * dim + d;
        const h = math.d64(hidden[at]);
        const v = math.d64(kv[hidden_width + d]);
        wide.want[d] = h + gate * v;
        wide.mag[d] = @abs(h) + gate * @abs(v) + @abs(v) * slope * spread;
    }
}

fn sampleBf16(lcg: *Lcg, family: usize, i: usize) u16 {
    const scale: f32 = switch (family) {
        0 => 1,
        1 => 16,
        2 => 0.0078125,
        else => if (i % 2 == 0) 1 else 0.0009765625,
    };
    return math.b(lcg.uniform() * scale);
}

const family_names = [_][]const u8{ "unit", "wide", "narrow", "cancellation" };

test "Engram hash columns equal the integer contract across a masked chunk and committed history" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var map_bytes: [raw_ids * 4]u8 = undefined;
    for (0..raw_ids) |i| std.mem.writeInt(u32, map_bytes[i * 4 ..][0..4], @intCast(500 + lcg.below(90000)), .little);
    const map = try device.buffer(map_bytes.len, 0);
    defer map.deinit();
    @memcpy(map.slice(u8, map_bytes.len), &map_bytes);
    const chunk_ids = [_]u32{ 7, 9, 11, 3, 13, 15, 17, 19, 21, 23 };
    const alive = [_]u8{ 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 };
    const ids = try device.buffer(hash_tokens * 4, 0);
    defer ids.deinit();
    @memcpy(ids.slice(u32, hash_tokens), &chunk_ids);
    const mask = try device.buffer(hash_tokens, 0);
    defer mask.deinit();
    @memcpy(mask.slice(u8, hash_tokens), &alive);
    const history = try device.buffer(engram.history_bytes, 0);
    defer history.deinit();
    const first = try device.buffer(hash_tokens * hash.columns * 4, 0);
    defer first.deinit();
    const second = try device.buffer(hash_tokens * hash.columns * 4, 0);
    defer second.deinit();
    const cb = queue.commandBuffer();
    var e = cb.compute(.serial);
    engram.encodeReset(e, &pipelines, view(history));
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        try engram.encodeHash(e, &pipelines, hash.layers[layer], view(ids), view(map), view(mask), view(history), view(out), @intCast(hash_chunk));
    }
    try engram.encodeCommit(e, &pipelines, view(ids), view(map), view(mask), view(history), @intCast(hash_chunk));
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        try engram.encodeHash(e, &pipelines, hash.layers[layer], view(ids).shifted(hash_chunk * 4), view(map), view(mask).shifted(hash_chunk), view(history), view(out).shifted(hash_chunk * hash.columns * 4), @intCast(hash_tokens - hash_chunk));
    }
    e.end();
    try completed(cb);
    var expected: [2][hash_tokens][hash.columns]u32 = undefined;
    for (0..2) |layer| {
        var cpu: hash.History = .{};
        try hash.hashChunk(&cpu, .{ .bytes = &map_bytes }, hash.layers[layer], &chunk_ids, &alive, &expected[layer]);
    }
    var unequal: usize = 0;
    for (0..2) |layer| {
        const out = if (layer == 0) first else second;
        const got = out.slice(u32, hash_tokens * hash.columns);
        for (expected[layer], 0..) |row, t| {
            for (row, 0..) |want, col| {
                if (got[t * hash.columns + col] != want) unequal += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), unequal);
    printEq("hash", "masked-chunk-and-history-carry", 2 * hash_tokens * hash.columns, unequal);
}

test "Engram gather narrows the E4M3 row grid to BF16 exactly like the FP32 contract" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var weight_bytes: [table_rows * head_dim]u8 = undefined;
    var scale_bytes: [table_rows * (head_dim / 32)]u8 = undefined;
    for (&weight_bytes) |*b| b.* = @intCast(lcg.below(127));
    for (&scale_bytes) |*b| b.* = @intCast(lcg.below(240));
    const weight = try device.buffer(weight_bytes.len, 0);
    defer weight.deinit();
    @memcpy(weight.slice(u8, weight_bytes.len), &weight_bytes);
    const scale = try device.buffer(scale_bytes.len, 0);
    defer scale.deinit();
    @memcpy(scale.slice(u8, scale_bytes.len), &scale_bytes);
    const row_ids = try device.buffer(gather_tokens * hash.columns * 4, 0);
    defer row_ids.deinit();
    const row_slice = row_ids.slice(u32, gather_tokens * hash.columns);
    for (row_slice) |*id| id.* = @intCast(lcg.below(table_rows));
    const gathered = try device.buffer(gather_tokens * gather_dim * 2, 0);
    defer gathered.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try engram.encodeGather(e, &pipelines, .{ .weight = view(weight), .scale = view(scale), .rows = @intCast(table_rows) }, view(row_ids), view(gathered), @intCast(gather_tokens), gather_dim);
    e.end();
    try completed(cb);
    const got = gathered.slice(u16, gather_tokens * gather_dim);
    var oracle: [gather_tokens * gather_dim]u16 = undefined;
    var ideal: [gather_tokens * gather_dim]u16 = undefined;
    for (0..gather_tokens) |t| {
        for (0..hash.columns) |col| {
            for (0..head_dim) |d| {
                const row: usize = row_slice[t * hash.columns + col];
                const code = weight_bytes[row * head_dim + d];
                const block = scale_bytes[row * (head_dim / 32) + d / 32];
                oracle[t * gather_dim + col * head_dim + d] = contractProduct(math.e4(code), math.e8(block));
                ideal[t * gather_dim + col * head_dim + d] = math.b64(@as(f64, math.e4(code)) * @as(f64, math.e8(block)));
            }
        }
    }
    var unequal: usize = 0;
    for (got, oracle[0..]) |a, want| {
        if (a != want) unequal += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), unequal);
    printEq("gather", "fp8-row-grid", got.len, unequal);
    printBf16("gather", "metal-vs-fp64-ideal", try deltaBf16(got, &ideal));
    printBf16("gather", "fp32-contract-vs-fp64-ideal", try deltaBf16(&oracle, &ideal));
    const row_bytes = head_dim + head_dim / 32;
    const rows = try device.buffer(table_rows * row_bytes, 0);
    defer rows.deinit();
    const row_data = rows.slice(u8, table_rows * row_bytes);
    for (0..table_rows) |row| {
        @memcpy(row_data[row * row_bytes ..][0..head_dim], weight_bytes[row * head_dim ..][0..head_dim]);
        @memcpy(row_data[row * row_bytes + head_dim ..][0 .. head_dim / 32], scale_bytes[row * (head_dim / 32) ..][0 .. head_dim / 32]);
    }
    const row_gather = try device.buffer(gather_tokens * gather_dim * 2, 0);
    defer row_gather.deinit();
    const table = engram.Table{
        .weight = view(rows),
        .scale = view(rows).shifted(head_dim),
        .rows = @intCast(table_rows),
        .weight_stride_bytes = row_bytes,
        .scale_stride_bytes = row_bytes,
    };
    const row_cb = queue.commandBuffer();
    const row_e = row_cb.compute(.serial);
    try engram.encodeGather(row_e, &pipelines, table, view(row_ids), view(row_gather), @intCast(gather_tokens), gather_dim);
    row_e.end();
    try completed(row_cb);
    const row_got = row_gather.slice(u16, gather_tokens * gather_dim);
    var row_unequal: usize = 0;
    for (row_got, oracle[0..]) |a, want| {
        if (a != want) row_unequal += 1;
    }
    printEq("gather", "fp8-interleaved-264-byte-rows", row_got.len, row_unequal);
    try std.testing.expectEqual(@as(usize, 0), row_unequal);
    printBf16("gather", "interleaved-metal-vs-fp64-ideal", try deltaBf16(row_got, &ideal));
}

test "Engram gate reaches BF16 at the FP32 contract over matching equations" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try engram.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var lcg = Lcg{};
    var q: [hidden_width]f32 = undefined;
    var k: [hidden_width]f32 = undefined;
    for (&q, &k) |*q_value, *k_value| {
        q_value.* = lcg.uniform();
        k_value.* = lcg.uniform();
    }
    var hidden: [gate_tokens * hidden_width]u16 = undefined;
    var kv: [gate_tokens * kv_width]u16 = undefined;
    var ideal_row: [gate_tokens * hidden_width]u16 = undefined;
    var contract_block: [dim]u16 = undefined;
    var wide: GateWide = undefined;
    const q_weight = try device.buffer(hidden_width * 4, 0);
    defer q_weight.deinit();
    @memcpy(q_weight.slice(f32, hidden_width), &q);
    const k_weight = try device.buffer(hidden_width * 4, 0);
    defer k_weight.deinit();
    @memcpy(k_weight.slice(f32, hidden_width), &k);
    const hidden_buffer = try device.buffer(hidden.len * 2, 0);
    defer hidden_buffer.deinit();
    const kv_buffer = try device.buffer(kv.len * 2, 0);
    defer kv_buffer.deinit();
    const gate_mask = try device.buffer(gate_tokens, 0);
    defer gate_mask.deinit();
    for (gate_mask.slice(u8, gate_tokens), 0..) |*m, i| m.* = @intFromBool(i == 0);
    const output = try device.buffer(gate_tokens * hidden_width * 2, 0);
    defer output.deinit();
    const masked_output = try device.buffer(gate_tokens * hidden_width * 2, 0);
    defer masked_output.deinit();
    const gate = engram.Gate{
        .q_weight = view(q_weight),
        .k_weight = view(k_weight),
        .dim = @intCast(dim),
        .copies = @intCast(copies),
        .hidden_stride = hidden_width,
        .kv_stride = kv_width,
        .output_stride = hidden_width,
        .eps = eps,
    };
    var cells_total: usize = 0;
    var unequal_total: usize = 0;
    var failures_total: usize = 0;
    var ratio_max: f64 = 0;
    var shown: usize = 0;
    for (family_names, 0..) |family, index| {
        for (&hidden, 0..) |*v, i| v.* = sampleBf16(&lcg, index, i);
        for (&kv, 0..) |*v, i| v.* = sampleBf16(&lcg, index, i + 7);
        @memcpy(hidden_buffer.slice(u16, hidden.len), &hidden);
        @memcpy(kv_buffer.slice(u16, kv.len), &kv);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try engram.encodeGate(e, &pipelines, gate, view(hidden_buffer), view(kv_buffer), null, view(output), @intCast(gate_tokens));
        e.end();
        try completed(cb);
        const got = output.slice(u16, gate_tokens * hidden_width);
        var unequal: usize = 0;
        var failures: usize = 0;
        var ratio: f64 = 0;
        for (0..gate_tokens) |t| {
            for (0..copies) |copy| {
                const hidden_row = hidden[t * hidden_width ..][0..hidden_width];
                const kv_row = kv[t * kv_width ..][0..kv_width];
                contractGate(hidden_row, kv_row, &q, &k, copy, &contract_block);
                idealGate(hidden_row, kv_row, &q, &k, copy, ideal_row[t * hidden_width + copy * dim ..][0..dim]);
                idealGateWide(hidden_row, kv_row, &q, &k, copy, &wide);
                for (0..dim) |d| {
                    const at = t * hidden_width + copy * dim + d;
                    const want = wide.want[d];
                    const allowed = contract.bound(gate_spec, gate_shape, want, wide.mag[d]);
                    const got_value = math.d64(got[at]);
                    if (!std.math.isFinite(got_value) or !std.math.isFinite(want) or !std.math.isFinite(allowed)) return error.NonFiniteMetric;
                    const diff = got_value - want;
                    if (@abs(diff) / allowed > ratio) ratio = @abs(diff) / allowed;
                    if (got[at] != contract_block[d]) {
                        unequal += 1;
                        if (shown < 8) {
                            printCell("metal-vs-contract", family, t, copy, at, got[at], contract_block[d], want, diff, allowed);
                            shown += 1;
                        }
                    }
                    if (@abs(diff) > allowed) {
                        failures += 1;
                        if (shown < 8) {
                            printCell("contract-bound", family, t, copy, at, got[at], contract_block[d], want, diff, allowed);
                            shown += 1;
                        }
                    }
                }
            }
        }
        printEq("gate", family, gate_tokens * hidden_width, unequal);
        printBf16("gate-ideal", family, try deltaBf16(got, &ideal_row));
        printBound("gate-bound", family, gate_tokens * hidden_width, failures, ratio);
        cells_total += gate_tokens * hidden_width;
        unequal_total += unequal;
        failures_total += failures;
        if (ratio > ratio_max) ratio_max = ratio;
        try std.testing.expectEqual(@as(usize, 0), failures);
    }
    printEq("gate", "all-families-metal-vs-contract", cells_total, unequal_total);
    printBound("gate-bound", "all-families", cells_total, failures_total, ratio_max);
    try std.testing.expectEqual(@as(usize, 0), failures_total);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try engram.encodeGate(e, &pipelines, gate, view(hidden_buffer), view(kv_buffer), view(gate_mask), view(masked_output), @intCast(gate_tokens));
    e.end();
    try completed(cb);
    const masked = masked_output.slice(u16, gate_tokens * hidden_width);
    try std.testing.expectEqualSlices(u16, hidden[hidden_width..][0..hidden_width], masked[hidden_width..][0..hidden_width]);
    printEq("gate", "dead-mask-zero-gate", hidden_width, 0);
}
