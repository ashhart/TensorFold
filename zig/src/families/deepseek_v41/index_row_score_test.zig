//! The indexer's row scorer against the tensor-tile scorer: every score a row's select reads is the same bit pattern.
const std = @import("std");
const mtl = @import("metal");
const index = @import("csa2_index.zig");
const kv = @import("kv.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const scratch_rows: u32 = 256;
const keys: u32 = 4100; // two score tiles, so the rows' tile-major slots differ by tile

fn fill(words: []u16, rnd: std.Random, lo: f32, ties: bool) void {
    for (words) |*w| {
        const v: f32 = if (ties) @as(f32, @floatFromInt(rnd.intRangeAtMost(i32, -1, 1))) * 0.5 + lo else rnd.float(f32) * 2 - 1 + lo;
        w.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
}

/// The visible scores of one indexer pass over seeded inputs, from the tensor-tile scorer when `nax`.
fn scores(device: mtl.Device, queue: mtl.Queue, ip: *const index.Pipelines, kp: *const kv.Pipelines, rows: u32, ratio: u32, ties: bool, nax: bool, out: []u32) !void {
    var prng = std.Random.DefaultPrng.init(0x5c0de + rows);
    const rnd = prng.random();
    const q = try device.buffer(@as(usize, rows) * 4096 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const w = try device.buffer(@as(usize, rows) * 32 * 2, mtl.ResourceOptions.shared);
    defer w.deinit();
    const raw = try device.buffer(@as(usize, keys) * 128 * 2, mtl.ResourceOptions.shared);
    defer raw.deinit();
    const storage = try device.buffer(@as(usize, keys) * 68, mtl.ResourceOptions.shared);
    defer storage.deinit();
    fill(q.slice(u16, @as(usize, rows) * 4096), rnd, 0, ties);
    fill(w.slice(u16, @as(usize, rows) * 32), rnd, 1.5, ties);
    fill(raw.slice(u16, @as(usize, keys) * 128), rnd, 0, ties);
    var scratch = try index.Scratch.init(device, scratch_rows);
    defer scratch.deinit();
    if (nax) _ = setenv("TF_DSV41_INDEX_ROW_SCORE", "0", 1) else _ = unsetenv("TF_DSV41_INDEX_ROW_SCORE");
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try kp.encodePack(enc, .{ .buffer = raw }, .{ .buffer = storage }, .{ .rows = keys, .width = 128, .stride = 128, .packed_stride = 68, .block = 32, .kind = .index_fp4 });
    // Every key is visible to the last row: its position plus one, over the ratio, is the key count.
    const args = index.Params{ .rows = rows, .q_stride = 4096, .weight_stride = 32, .position = keys * ratio - rows, .ratio = ratio, .keys = keys };
    try ip.encode(enc, .{ .buffer = q }, .{ .buffer = w }, .{ .buffer = storage }, scratch, args);
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("index row score: {s}", .{message});
        return error.GpuExecutionFailed;
    }
    // The one-shot select's tile-major slots (tile, row, key): a row's select reads its visible keys' scores only.
    const at = scratch.scores();
    const slots: [*]const u32 = @ptrCast(@alignCast(at.buffer.contents() + at.offset));
    const per_row: usize = index.score_tile_width;
    for (0..rows) |r| {
        const visible: u32 = @min(keys, (keys * ratio - rows + @as(u32, @intCast(r)) + 1) / ratio);
        for (0..visible) |k| out[r * keys + k] = slots[((k / per_row) * rows + r) * per_row + k % per_row];
        for (visible..keys) |k| out[r * keys + k] = 0;
    }
}

// Rows 9 to 127 around the 16- and 32-row block edges, both ratios; a tie-heavy input as well as a random one.
test "index row scorer scores every visible key as the tensor-tile scorer does" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ip = try index.Pipelines.load(device);
    defer ip.deinit();
    if (ip.row_score == null or ip.nax_score == null) return error.SkipZigTest;
    var kp = try kv.Pipelines.load(device);
    defer kp.deinit();
    defer _ = unsetenv("TF_DSV41_INDEX_ROW_SCORE");
    const gpa = std.testing.allocator;
    for ([_]u32{ 9, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100, 127 }) |rows| {
        for ([_]u32{ 1, 2 }) |ratio| {
            for ([_]bool{ false, true }) |ties| {
                const row = try gpa.alloc(u32, @as(usize, rows) * keys);
                defer gpa.free(row);
                const tile = try gpa.alloc(u32, @as(usize, rows) * keys);
                defer gpa.free(tile);
                try scores(device, queue, &ip, &kp, rows, ratio, ties, false, row);
                try scores(device, queue, &ip, &kp, rows, ratio, ties, true, tile);
                for (row, tile, 0..) |a, b, i| if (a != b) {
                    std.debug.print("rows {d} ratio {d}: row {d} key {d} scores {x}, tile scorer {x}\n", .{ rows, ratio, i / keys, i % keys, a, b });
                    return error.TestExpectedEqual;
                };
            }
        }
    }
}
