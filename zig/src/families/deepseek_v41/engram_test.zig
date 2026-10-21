//! Behavioral hash, rollback, selected-row dequantization and per-copy signed-sqrt gate regressions.
const std = @import("std");
const hash = @import("engram_hash.zig");
const engram = @import("engram.zig");
const mtl = @import("metal");
const View = engram.View;

// Generated offline with the official NumPy multipliers and prime-bucket algorithm, not the native code.
const first = [_]u32{
    8939594,   28321402,  40265357,  55817216,  71147020,  86924254,  110813108, 121427189,
    128454377, 156247504, 165979524, 188168291, 198564788, 211763979, 239563936, 241631034,
    263018334, 286373017, 297618161, 319241270, 333656732, 339279600, 364226324, 373889199,
};
const third = [_]u32{
    13176822,  30752650,  42532473,  56771564,  70133176,  85254602,  108815600, 115334181,
    138863141, 149049926, 167314083, 191090346, 204680718, 211476972, 237672992, 245604111,
    269210034, 282954889, 298312790, 313992267, 329778842, 345458689, 359433926, 374584261,
};
const after_dead = [_]u32{
    762556,    18016262,  38494153,  58317688,  64056565,  81970637,  106927994, 122902996,
    129060773, 153088767, 160206490, 178949865, 205667218, 211027556, 235070602, 251994813,
    263310652, 278975004, 292595948, 315407209, 325281398, 348093105, 353903080, 377405725,
};

test "official hash IDs preserve order, padding and globally unique layer buckets" {
    var history: hash.History = .{};
    const a = try history.push(7, hash.layers[0]);
    try std.testing.expectEqualSlices(u32, &first, &a);
    _ = try history.push(9, hash.layers[0]);
    const c = try history.push(11, hash.layers[0]);
    try std.testing.expectEqualSlices(u32, &third, &c);
    for (hash.layers) |meta| for (meta.primes, meta.offsets) |p, offset| {
        try std.testing.expect(p >= 16000000);
        try std.testing.expect(offset + p <= meta.rows);
    };
    var high: hash.History = .{ .tail = .{ 1234, 5678, 42 }, .position = 3 };
    const result = try high.push(99091, hash.layers[1]);
    const expected = [_]u32{
        3315102,   30010869,  33829090,  51274514,  77267621,  84401643,  106522093, 127296527,
        137402994, 151593930, 162661867, 180852958, 195297507, 213361283, 228864003, 255457679,
        262256667, 282063670, 300635716, 304966933, 321424067, 350721784, 358839309, 370654406,
    };
    try std.testing.expectEqualSlices(u32, &expected, &result);
}

test "dead image positions block all older lookbacks and snapshot rollback restores the exact boundary" {
    var history: hash.History = .{};
    _ = try history.push(7, hash.layers[0]);
    _ = try history.push(9, hash.layers[0]);
    const saved = history.snapshot();
    _ = try history.push(hash.dead, hash.layers[0]);
    const got = try history.push(13, hash.layers[0]);
    try std.testing.expectEqualSlices(u32, &after_dead, &got);
    history.rollback(saved);
    const replay = try history.push(11, hash.layers[0]);
    try std.testing.expectEqualSlices(u32, &third, &replay);
    history.reset();
    const reset = try history.push(7, hash.layers[0]);
    try std.testing.expectEqualSlices(u32, &first, &reset);
    try std.testing.expectError(error.InvalidEngramCompressedId, history.push(-2, hash.layers[0]));
    try std.testing.expectEqual(@as(u64, 1), history.position);
}

test "all prefill/decode partitions preserve hashes including a dead token at the chunk boundary" {
    var bytes: [20 * 4]u8 = undefined;
    for (0..20) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @intCast(i), .little);
    const map = hash.TokenMap{ .bytes = &bytes };
    const ids = [_]u32{ 7, 9, 11, 3, 13, 15, 17 };
    const mask = [_]u8{ 1, 1, 1, 0, 1, 1, 1 };
    for (hash.layers) |meta| {
        var whole: hash.History = .{};
        var expected: [ids.len][hash.columns]u32 = undefined;
        try hash.hashChunk(&whole, map, meta, &ids, &mask, &expected);
        for (0..ids.len + 1) |split| {
            var chunked: hash.History = .{};
            var actual: [ids.len][hash.columns]u32 = undefined;
            try hash.hashChunk(&chunked, map, meta, ids[0..split], mask[0..split], actual[0..split]);
            try hash.hashChunk(&chunked, map, meta, ids[split..], mask[split..], actual[split..]);
            try std.testing.expectEqualDeep(expected, actual);
            try std.testing.expectEqualDeep(whole, chunked);
        }
        const saved = whole.snapshot();
        var invalid: [2][hash.columns]u32 = undefined;
        try std.testing.expectError(error.InvalidEngramToken, hash.hashChunk(&whole, map, meta, &.{ 1, 20 }, null, &invalid));
        try std.testing.expectEqualDeep(saved, whole);
    }
}

test "raw tokenizer IDs are compressed before hashing and normalization-equivalent tokens share hashes" {
    var bytes: [4 * 4]u8 = undefined;
    for ([_]u32{ 7, 9, 11, 9 }, 0..) |id, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], id, .little);
    const map = hash.TokenMap{ .bytes = &bytes };
    var a: hash.History = .{};
    var b: hash.History = .{};
    var out_a: [3][hash.columns]u32 = undefined;
    var out_b: [3][hash.columns]u32 = undefined;
    try hash.hashChunk(&a, map, hash.layers[0], &.{ 0, 1, 2 }, null, &out_a);
    try hash.hashChunk(&b, map, hash.layers[0], &.{ 0, 3, 2 }, null, &out_b);
    try std.testing.expectEqualDeep(out_a, out_b);
    try std.testing.expectEqualSlices(u32, &first, &out_a[0]);
    try std.testing.expectEqualSlices(u32, &third, &out_a[2]);
}

fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}
fn bf(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
fn float(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn completed(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.EngramGPUFailure;
}

test "Metal hashes chunks with image breaks and rolls back bounded GPU history" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try engram.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const ids = try device.buffer(7 * 4, 0);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 7), &[_]u32{ 7, 9, 11, 3, 13, 15, 17 });
    const map = try device.buffer(20 * 4, 0);
    defer map.deinit();
    for (map.slice(u32, 20), 0..) |*id, i| id.* = @intCast(i);
    const mask = try device.buffer(7, 0);
    defer mask.deinit();
    @memcpy(mask.slice(u8, 7), &[_]u8{ 1, 1, 1, 0, 1, 1, 1 });
    const history = try device.buffer(engram.history_bytes, 0);
    defer history.deinit();
    const saved = try device.buffer(engram.history_bytes, 0);
    defer saved.deinit();
    const out = try device.buffer(7 * 24 * 4, 0);
    defer out.deinit();
    const replay = try device.buffer(24 * 4, 0);
    defer replay.deinit();
    const cb = queue.commandBuffer();
    var e = cb.compute(.serial);
    engram.encodeReset(e, &p, view(history));
    try engram.encodeHash(e, &p, hash.layers[0], view(ids), view(map), view(mask), view(history), view(out), 3);
    try engram.encodeCommit(e, &p, view(ids), view(map), view(mask), view(history), 3);
    e.end();
    var blit = cb.blit();
    engram.encodeSnapshot(blit, view(history), view(saved));
    blit.end();
    e = cb.compute(.serial);
    try engram.encodeHash(e, &p, hash.layers[0], view(ids).shifted(12), view(map), view(mask).shifted(3), view(history), view(out).shifted(3 * 24 * 4), 4);
    try engram.encodeCommit(e, &p, view(ids).shifted(12), view(map), view(mask).shifted(3), view(history), 4);
    e.end();
    blit = cb.blit();
    engram.encodeRollback(blit, view(saved), view(history));
    blit.end();
    e = cb.compute(.serial);
    try engram.encodeHash(e, &p, hash.layers[1], view(ids).shifted(16), view(map), null, view(history), view(replay), 1);
    e.end();
    try completed(cb);
    var cpu: hash.History = .{};
    for (ids.slice(u32, 7), mask.slice(u8, 7), 0..) |id, alive, i| {
        const want = try cpu.push(if (alive == 0) hash.dead else @intCast(id), hash.layers[0]);
        try std.testing.expectEqualSlices(u32, &want, out.slice(u32, 7 * 24)[i * 24 ..][0..24]);
    }
    cpu = .{ .tail = .{ 11, 9, 7 }, .position = 3 };
    const want = try cpu.push(13, hash.layers[1]);
    try std.testing.expectEqualSlices(u32, &want, replay.slice(u32, 24));
    try std.testing.expectEqualSlices(i32, &.{ 11, 9, 7 }, history.slice(i32, 4)[0..3]);
}

test "Metal gathers only selected FP8 rows with rowwise scales and gates each mHC copy independently" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try engram.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const table = try device.buffer(3 * 272, 0);
    defer table.deinit();
    @memset(table.slice(u8, 3 * 272), 0x7f);
    const scales = try device.buffer(3 * 12, 0);
    defer scales.deinit();
    @memset(scales.slice(u8, 3 * 12), 255);
    for (0..3) |r| {
        @memset(table.slice(u8, 3 * 272)[r * 272 ..][0..256], if (r == 0) 0x38 else if (r == 1) 0xb8 else 0x7e);
        for (0..8) |g| scales.slice(u8, 3 * 12)[r * 12 + g] = @intCast(127 + g);
    }
    table.slice(u8, 3 * 272)[2 * 272] = 0x01;
    table.slice(u8, 3 * 272)[2 * 272 + 1] = 0x80;
    scales.slice(u8, 3 * 12)[12 + 7] = 255;
    const ids = try device.buffer(24 * 4, 0);
    defer ids.deinit();
    for (ids.slice(u32, 24), 0..) |*id, i| id.* = @intCast((i * 2 + 1) % 3);
    const gathered = try device.buffer((engram.gather_dim + 16) * 2, 0);
    defer gathered.deinit();
    const hidden = try device.buffer(2 * 4 * 2, 0);
    defer hidden.deinit();
    for (hidden.slice(u16, 8), 0..) |*v, i| v.* = bf(if (i < 4) 1 else 2);
    const kv = try device.buffer(12 * 2, 0);
    defer kv.deinit();
    for (kv.slice(u16, 12), 0..) |*v, i| v.* = bf(if (i < 4) 1 else if (i < 8) -2 else 3);
    const q = try device.buffer(8 * 4, 0);
    defer q.deinit();
    for (q.slice(f32, 8), 0..) |*v, i| v.* = if (i < 4) 1 else 0.5;
    const k = try device.buffer(8 * 4, 0);
    defer k.deinit();
    for (k.slice(f32, 8), 0..) |*v, i| v.* = if (i < 4) 1 else 2;
    const output = try device.buffer(8 * 2, 0);
    defer output.deinit();
    const dead_mask = try device.buffer(1, 0);
    defer dead_mask.deinit();
    dead_mask.slice(u8, 1)[0] = 0;
    const masked_output = try device.buffer(8 * 2, 0);
    defer masked_output.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try engram.encodeGather(e, &p, .{ .weight = view(table), .scale = view(scales), .rows = 3, .weight_stride_bytes = 272, .scale_stride_bytes = 12 }, view(ids), view(gathered), 1, engram.gather_dim + 16);
    try engram.encodeGate(e, &p, .{ .q_weight = view(q), .k_weight = view(k), .dim = 4, .copies = 2, .hidden_stride = 8, .kv_stride = 12, .output_stride = 8, .eps = 1e-20 }, view(hidden), view(kv), null, view(output), 1);
    try engram.encodeGate(e, &p, .{ .q_weight = view(q), .k_weight = view(k), .dim = 4, .copies = 2, .hidden_stride = 8, .kv_stride = 12, .output_stride = 8, .eps = 1e-20 }, view(hidden), view(kv), view(dead_mask), view(masked_output), 1);
    e.end();
    try completed(cb);
    for (ids.slice(u32, 24), 0..) |r, col| for (0..256) |d| {
        const got = gathered.slice(u16, engram.gather_dim)[col * 256 + d];
        if (r == 1 and d >= 224) {
            try std.testing.expect(std.math.isNan(float(got)));
            continue;
        }
        const base: f32 = if (r == 2 and d == 0) 0x1p-9 else if (r == 2 and d == 1) -0.0 else if (r == 0) 1 else if (r == 1) -1 else 448;
        const scale: f32 = @floatFromInt(@as(u32, 1) << @as(u5, @intCast(d / 32)));
        try std.testing.expectEqual(bf(base * scale), got);
    };
    const gate_positive = 1.0 / (1.0 + @exp(-@sqrt(@as(f32, 2))));
    const gate_negative = 1.0 / (1.0 + @exp(@sqrt(@as(f32, 2))));
    for (output.slice(u16, 8), 0..) |v, i| {
        const want = bf(if (i < 4) 1 + 3 * gate_positive else 2 + 3 * gate_negative);
        try std.testing.expectApproxEqAbs(float(want), float(v), 0.015625);
    }
    try std.testing.expectEqualSlices(u16, hidden.slice(u16, 8), masked_output.slice(u16, 8));
}
