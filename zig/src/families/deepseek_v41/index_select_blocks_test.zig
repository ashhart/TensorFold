//! The indexer's one-shot select over row sub-blocks against the heap path: same inputs, same picks.
const std = @import("std");
const mtl = @import("metal");
const index = @import("csa2_index.zig");
const kv = @import("kv.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const Case = struct { rows: u32, keys: u32, ratio: u32, ties: bool };
const scratch_rows: u32 = 256;

fn fill(words: []u16, rnd: std.Random, ties: bool, lo: f32) void {
    for (words) |*w| {
        const v: f32 = if (ties) @as(f32, @floatFromInt(rnd.intRangeAtMost(i32, -1, 1))) * 0.5 + lo else rnd.float(f32) * 2 - 1 + lo;
        w.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
}

/// The picks of one indexer pass over seeded inputs, from the heap path when `heap`.
fn picks(device: mtl.Device, queue: mtl.Queue, ip: *const index.Pipelines, kp: *const kv.Pipelines, c: Case, heap: bool, out: []i32) !void {
    var prng = std.Random.DefaultPrng.init(0x5e1ec7 + c.keys);
    const rnd = prng.random();
    const q = try device.buffer(@as(usize, c.rows) * 4096 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const w = try device.buffer(@as(usize, c.rows) * 32 * 2, mtl.ResourceOptions.shared);
    defer w.deinit();
    const raw = try device.buffer(@as(usize, c.keys) * 128 * 2, mtl.ResourceOptions.shared);
    defer raw.deinit();
    const storage = try device.buffer(@as(usize, c.keys) * 68, mtl.ResourceOptions.shared);
    defer storage.deinit();
    fill(q.slice(u16, @as(usize, c.rows) * 4096), rnd, c.ties, 0);
    fill(w.slice(u16, @as(usize, c.rows) * 32), rnd, c.ties, 1.5);
    fill(raw.slice(u16, @as(usize, c.keys) * 128), rnd, c.ties, 0);
    var scratch = try index.Scratch.init(device, scratch_rows);
    defer scratch.deinit();
    if (heap) _ = setenv("TF_DSV41_INDEX_SELECT_ONE", "0", 1) else _ = unsetenv("TF_DSV41_INDEX_SELECT_ONE");
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try kp.encodePack(enc, .{ .buffer = raw }, .{ .buffer = storage }, .{ .rows = c.keys, .width = 128, .stride = 128, .packed_stride = 68, .block = 32, .kind = .index_fp4 });
    // Every key is visible to the last row: its position plus one, over the ratio, is the key count.
    const args = index.Params{ .rows = c.rows, .q_stride = 4096, .weight_stride = 32, .position = c.keys * c.ratio - c.rows, .ratio = c.ratio, .keys = c.keys };
    try ip.encode(enc, .{ .buffer = q }, .{ .buffer = w }, .{ .buffer = storage }, scratch, args);
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("index select blocks: {s}", .{message});
        return error.GpuExecutionFailed;
    }
    const at = scratch.positions();
    const ids: [*]const i32 = @ptrCast(@alignCast(at.buffer.contents() + at.offset));
    @memcpy(out, ids[0 .. @as(usize, c.rows) * 512]);
}

// The 256-row scratch holds 256 / tiles rows of scores: cases 1-4 run in sub-blocks, the last on the heap path in both.
test "index one-shot select in row sub-blocks picks what the heap path picks" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ip = try index.Pipelines.load(device);
    defer ip.deinit();
    var kp = try kv.Pipelines.load(device);
    defer kp.deinit();
    defer _ = unsetenv("TF_DSV41_INDEX_SELECT_ONE");
    const cases = [_]Case{
        .{ .rows = 150, .keys = 8200, .ratio = 1, .ties = false },
        .{ .rows = 256, .keys = 4097, .ratio = 1, .ties = false },
        .{ .rows = 200, .keys = 16384, .ratio = 2, .ties = false },
        .{ .rows = 130, .keys = 8200, .ratio = 1, .ties = true },
        .{ .rows = 70, .keys = 33000, .ratio = 1, .ties = false },
    };
    const gpa = std.testing.allocator;
    for (cases) |c| {
        const one = try gpa.alloc(i32, @as(usize, c.rows) * 512);
        defer gpa.free(one);
        const heap = try gpa.alloc(i32, @as(usize, c.rows) * 512);
        defer gpa.free(heap);
        try picks(device, queue, &ip, &kp, c, false, one);
        try picks(device, queue, &ip, &kp, c, true, heap);
        for (one, heap, 0..) |a, b, i| if (a != b) {
            std.debug.print("rows {d} keys {d}: row {d} slot {d} picks {d}, heap path {d}\n", .{ c.rows, c.keys, i / 512, i % 512, a, b });
            return error.TestExpectedEqual;
        };
    }
}
