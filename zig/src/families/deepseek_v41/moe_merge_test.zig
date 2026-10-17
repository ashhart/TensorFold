//! The merged MoE products (shared expert as each token's seventh row) against the separate path.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("../../core/exl3_projection.zig");
const work = @import("workspace.zig");

const Owned = struct {
    matrix: ex.Matrix,
    buffers: [3]mtl.Buffer,

    fn init(device: mtl.Device, k: usize, n: usize, halves: u16, seed: u64) !Owned {
        const words = @as(usize, halves) * 8 * (k / 16) * (n / 16);
        const t = try device.buffer(words * 2, mtl.ResourceOptions.shared);
        errdefer t.deinit();
        const su = try device.buffer(k * 2, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(n * 2, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.slice(i16, words)) |*x| x.* = prng.random().int(i16);
        for (su.slice(f16, k), 0..) |*x, i| x.* = if ((i + seed) % 3 == 0) -0.75 else 1.125;
        for (sv.slice(f16, n), 0..) |*x, i| x.* = if ((i + seed) % 5 == 0) -0.25 else 0.875;
        return .{ .matrix = .{ .trellis = .{ .buffer = t }, .suh = .{ .buffer = su }, .svh = .{ .buffer = sv }, .k = k, .n = n, .width = .{ .halves = halves } }, .buffers = .{ t, su, sv } };
    }
    fn deinit(w: Owned) void {
        for (w.buffers) |b| b.deinit();
    }
};

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

/// Four routed experts and a shared one at index 4 (the role of workspace.shared_expert).
const experts = 4;
const tokens = 2;
const plan_rows = 3;

fn table(device: mtl.Device, set: []const Owned) !mtl.Buffer {
    const out = try device.buffer(set.len * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    for (set, out.slice(ex.ExpertDescriptor, set.len)) |w, *d| d.* = try ex.descriptor(w.matrix);
    return out;
}

fn complete(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

test "a merged MoE projection puts routed rows at 6 r + j and the shared row at 6 t + r, same bits" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    // gate shape (k 256 -> n 128) and down shape (k 128 -> n 256), each with five matrices.
    var up_set: [experts + 1]Owned = undefined;
    var down_set: [experts + 1]Owned = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        up_set[i].deinit();
        down_set[i].deinit();
    };
    for (0..experts + 1) |i| {
        up_set[i] = try Owned.init(device, 256, 128, @intCast(3 + i % 4), 10 + i);
        down_set[i] = Owned.init(device, 128, 256, @intCast(4 + i % 3), 30 + i) catch |err| {
            up_set[i].deinit();
            return err;
        };
        made += 1;
    }
    const up_table = try table(device, &up_set);
    defer up_table.deinit();
    const down_table = try table(device, &down_set);
    defer down_table.deinit();

    const routed = [tokens * 6]u32{ 2, 0, 3, 1, 3, 2, 1, 1, 0, 2, 3, 0 };
    const ids = try device.buffer(tokens * 6 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, tokens * 6), &routed);
    const routes7 = try device.buffer(plan_rows * 7 * 4, mtl.ResourceOptions.shared);
    defer routes7.deinit();
    for (0..tokens) |r| {
        for (0..6) |j| routes7.slice(u32, plan_rows * 7)[r * 7 + j] = routed[r * 6 + j];
        routes7.slice(u32, plan_rows * 7)[r * 7 + 6] = experts;
    }
    const slots = try device.buffer(plan_rows * 7 * 4, mtl.ResourceOptions.shared);
    defer slots.deinit();
    for (0..plan_rows) |r| for (0..7) |j| {
        slots.slice(u32, plan_rows * 7)[r * 7 + j] = work.moeSlot(plan_rows, r, j);
    };

    const x = try device.buffer(tokens * 256 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(u16, tokens * 256), 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 16);
    // Down inputs: routed activations at rows 6 r + j, the shared activation at 6 t + r (and its own buffer).
    const act = try device.buffer(plan_rows * 7 * 128 * 2, mtl.ResourceOptions.shared);
    defer act.deinit();
    for (act.slice(u16, plan_rows * 7 * 128), 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) / 32);
    const shared_act = try device.buffer(tokens * 128 * 2, mtl.ResourceOptions.shared);
    defer shared_act.deinit();
    @memcpy(shared_act.slice(u16, tokens * 128), act.slice(u16, plan_rows * 7 * 128)[plan_rows * 6 * 128 ..][0 .. tokens * 128]);

    const gate_sep = try device.buffer(tokens * 6 * 128 * 2, mtl.ResourceOptions.shared);
    defer gate_sep.deinit();
    const shared_sep = try device.buffer(tokens * 128 * 2, mtl.ResourceOptions.shared);
    defer shared_sep.deinit();
    const gate_merged = try device.buffer(plan_rows * 7 * 128 * 2, mtl.ResourceOptions.shared);
    defer gate_merged.deinit();
    const down_sep = try device.buffer(tokens * 6 * 256 * 2, mtl.ResourceOptions.shared);
    defer down_sep.deinit();
    const shared_down_sep = try device.buffer(tokens * 256 * 2, mtl.ResourceOptions.shared);
    defer shared_down_sep.deinit();
    const down_merged = try device.buffer(plan_rows * 7 * 256 * 2, mtl.ResourceOptions.shared);
    defer down_merged.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(256, 256, plan_rows * 7), mtl.ResourceOptions.shared);
    defer scratch.deinit();

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (up_set) |w| ex.useMatrix(e, w.matrix);
    for (down_set) |w| ex.useMatrix(e, w.matrix);
    // Separate: routed indexed rows plus a plain shared product, as the grouped-free path ran.
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = up_table }, .count = experts, .k = 256, .n = 128 }, .{ .buffer = x }, .{ .buffer = gate_sep }, .{ .buffer = scratch }, .{ .count = tokens * 6, .x_stride = 256, .y_stride = 128, .input_rows = tokens, .output_rows = tokens * 6, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 6 } });
    e.barrier();
    try ex.encode(&p, e, up_set[experts].matrix, .{ .buffer = x }, .{ .buffer = shared_sep }, .{ .buffer = scratch }, .{ .count = tokens, .x_stride = 256, .y_stride = 128 });
    e.barrier();
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = down_table }, .count = experts, .k = 128, .n = 256 }, .{ .buffer = act }, .{ .buffer = down_sep }, .{ .buffer = scratch }, .{ .count = tokens * 6, .x_stride = 128, .y_stride = 256, .input_rows = tokens * 6, .output_rows = tokens * 6, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 } });
    e.barrier();
    try ex.encode(&p, e, down_set[experts].matrix, .{ .buffer = shared_act }, .{ .buffer = shared_down_sep }, .{ .buffer = scratch }, .{ .count = tokens, .x_stride = 128, .y_stride = 256 });
    e.barrier();
    // Merged: seven rows per token through the five-entry tables, placed by moeSlot.
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = up_table }, .count = experts + 1, .k = 256, .n = 128 }, .{ .buffer = x }, .{ .buffer = gate_merged }, .{ .buffer = scratch }, .{ .count = tokens * 7, .x_stride = 256, .y_stride = 128, .output_ids = .{ .buffer = slots }, .output_rows = plan_rows * 7, .routes = .{ .ids = .{ .buffer = routes7 }, .repeat = 7 } });
    e.barrier();
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = down_table }, .count = experts + 1, .k = 128, .n = 256 }, .{ .buffer = act }, .{ .buffer = down_merged }, .{ .buffer = scratch }, .{ .count = tokens * 7, .x_stride = 128, .y_stride = 256, .input_ids = .{ .buffer = slots }, .input_rows = plan_rows * 7, .output_ids = .{ .buffer = slots }, .output_rows = plan_rows * 7, .routes = .{ .ids = .{ .buffer = routes7 }, .repeat = 1 } });
    try complete(cb, e);

    const gm = gate_merged.slice(u16, plan_rows * 7 * 128);
    const dm = down_merged.slice(u16, plan_rows * 7 * 256);
    try std.testing.expectEqualSlices(u16, gate_sep.slice(u16, tokens * 6 * 128), gm[0 .. tokens * 6 * 128]);
    try std.testing.expectEqualSlices(u16, shared_sep.slice(u16, tokens * 128), gm[plan_rows * 6 * 128 ..][0 .. tokens * 128]);
    try std.testing.expectEqualSlices(u16, down_sep.slice(u16, tokens * 6 * 256), dm[0 .. tokens * 6 * 256]);
    try std.testing.expectEqualSlices(u16, shared_down_sep.slice(u16, tokens * 256), dm[plan_rows * 6 * 256 ..][0 .. tokens * 256]);
}
