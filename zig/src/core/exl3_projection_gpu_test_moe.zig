//! GPU tests for the decode MoE fusions: each fused pass against the unfused pipeline, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("exl3_projection.zig");
const support = @import("exl3_projection_gpu_test_support.zig");
const bf16 = support.bf16;
const complete = support.complete;

pub const hidden = 256;
pub const mid = 256;
pub const out_n = 256;
pub const experts = 4;
pub const tokens = 2;
pub const picks = 3;
pub const routed = tokens * picks;

/// One matrix of any shape with its own rotation scales and bias.
pub const Owned = struct {
    matrix: ex.Matrix,
    buffers: [4]mtl.Buffer,

    pub fn init(device: mtl.Device, k: usize, n: usize, halves: u16, seed: u64) !Owned {
        const words = @as(usize, halves) * 8 * (k / 16) * (n / 16);
        const t = try device.buffer(words * 2 + 16, mtl.ResourceOptions.shared);
        errdefer t.deinit();
        const su = try device.buffer(k * 2 + 16, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        const bias = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer bias.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const scales = [_]f16{ -1.5, -0.75, 0.5, 0.875, 1.125, 1.25 };
        for (t.slice(i16, words + 8)[8..]) |*x| x.* = prng.random().int(i16);
        for (su.slice(f16, k + 8)[8..]) |*x| x.* = scales[prng.random().uintLessThan(usize, scales.len)];
        for (sv.slice(f16, n + 8)[8..]) |*x| x.* = scales[prng.random().uintLessThan(usize, scales.len)];
        for (bias.slice(f16, n + 8)[8..]) |*x| x.* = @floatCast((prng.random().float(f32) - 0.5) * 0.25);
        return .{ .matrix = .{ .trellis = .{ .buffer = t, .offset = 16 }, .suh = .{ .buffer = su, .offset = 16 }, .svh = .{ .buffer = sv, .offset = 16 }, .bias = .{ .buffer = bias, .offset = 16 }, .k = k, .n = n, .width = .{ .halves = halves } }, .buffers = .{ t, su, sv, bias } };
    }

    pub fn deinit(w: Owned) void {
        for (w.buffers) |b| b.deinit();
    }
};

fn table(device: mtl.Device, count: usize) !mtl.Buffer {
    return device.buffer(count * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
}

/// A four-expert block with a shared expert, two tokens of three picks: the decode MoE's shapes at test size.
pub const Fixture = struct {
    device: mtl.Device,
    p: ex.Pipelines,
    mats: [3 * experts + 3]Owned,
    /// Gate descriptors, up descriptors, then the shared gate and up: the merged gate+up call's table.
    fused: mtl.Buffer,
    down: mtl.Buffer,
    /// The routed down descriptors and the shared down after them.
    down_shared: mtl.Buffer,
    shared_down: mtl.Buffer,
    zero_ids: mtl.Buffer,
    x: mtl.Buffer,
    route_ids: mtl.Buffer,
    weights: mtl.Buffer,
    fused_ids: mtl.Buffer,
    fused_sources: mtl.Buffer,

    pub fn init(device: mtl.Device, halves: u16) !Fixture {
        var f: Fixture = undefined;
        f.device = device;
        f.p = try ex.Pipelines.load(device);
        errdefer f.p.deinit();
        var made: usize = 0;
        errdefer for (f.mats[0..made]) |m| m.deinit();
        for (&f.mats, 0..) |*m, i| {
            const down = (i >= 2 * experts and i < 3 * experts) or i == 3 * experts + 2;
            m.* = try Owned.init(device, if (down) mid else hidden, if (down) out_n else mid, halves, 0x100 + @as(u64, i));
            made += 1;
        }
        const shared_gate = f.mats[3 * experts];
        const shared_up = f.mats[3 * experts + 1];
        f.fused = try table(device, 2 * experts + 2);
        errdefer f.fused.deinit();
        f.down = try table(device, experts);
        errdefer f.down.deinit();
        f.down_shared = try table(device, experts + 1);
        errdefer f.down_shared.deinit();
        f.shared_down = try table(device, 1);
        errdefer f.shared_down.deinit();
        const fd = f.fused.slice(ex.ExpertDescriptor, 2 * experts + 2);
        for (0..experts) |e| {
            fd[e] = try ex.descriptor(f.mats[e].matrix);
            fd[experts + e] = try ex.descriptor(f.mats[experts + e].matrix);
            f.down.slice(ex.ExpertDescriptor, experts)[e] = try ex.descriptor(f.mats[2 * experts + e].matrix);
        }
        fd[2 * experts] = try ex.descriptor(shared_gate.matrix);
        fd[2 * experts + 1] = try ex.descriptor(shared_up.matrix);
        f.shared_down.slice(ex.ExpertDescriptor, 1)[0] = try ex.descriptor(f.mats[3 * experts + 2].matrix);
        @memcpy(f.down_shared.slice(ex.ExpertDescriptor, experts + 1)[0..experts], f.down.slice(ex.ExpertDescriptor, experts));
        f.down_shared.slice(ex.ExpertDescriptor, experts + 1)[experts] = f.shared_down.slice(ex.ExpertDescriptor, 1)[0];
        f.zero_ids = try device.buffer(8 * 4, mtl.ResourceOptions.shared);
        errdefer f.zero_ids.deinit();
        @memset(f.zero_ids.slice(u32, 8), 0);
        f.x = try support.input(device, tokens);
        errdefer f.x.deinit();
        var prng = std.Random.DefaultPrng.init(0x5eed);
        for (f.x.slice(u16, tokens * (hidden + 16) + 16), 0..) |*v, i| v.* = bf16((@as(f32, @floatFromInt(i % 29)) - 14) / 16 + prng.random().float(f32) * 0.1);
        f.route_ids = try device.buffer((routed + tokens) * 4, mtl.ResourceOptions.shared);
        errdefer f.route_ids.deinit();
        @memcpy(f.route_ids.slice(u32, routed + tokens)[0..routed], &[_]u32{ 0, 1, 2, 3, 1, 0 });
        f.weights = try device.buffer(routed * 4, mtl.ResourceOptions.shared);
        errdefer f.weights.deinit();
        @memcpy(f.weights.slice(f32, routed), &[_]f32{ 0.75, 1.5, 0.125, 1.0625, 0.3, 2.0 });
        const count = 2 * routed + 2 * tokens;
        f.fused_ids = try device.buffer(count * 4, mtl.ResourceOptions.shared);
        errdefer f.fused_ids.deinit();
        f.fused_sources = try device.buffer(count * 4, mtl.ResourceOptions.shared);
        errdefer f.fused_sources.deinit();
        const ids = f.fused_ids.slice(u32, count);
        const src = f.fused_sources.slice(u32, count);
        for (0..routed) |i| {
            const e = f.route_ids.slice(u32, routed + tokens)[i];
            ids[i] = e;
            ids[routed + i] = e + experts;
            src[i] = @intCast(i / picks);
            src[routed + i] = @intCast(i / picks);
        }
        for (0..tokens) |t| {
            ids[2 * routed + t] = 2 * experts;
            ids[2 * routed + tokens + t] = 2 * experts + 1;
            src[2 * routed + t] = @intCast(t);
            src[2 * routed + tokens + t] = @intCast(t);
        }
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        for (f.mats) |m| m.deinit();
        for ([_]mtl.Buffer{ f.fused, f.down, f.down_shared, f.shared_down, f.zero_ids, f.x, f.route_ids, f.weights, f.fused_ids, f.fused_sources }) |b| b.deinit();
        f.p.deinit();
    }

    pub fn use(f: *const Fixture, e: mtl.ComputeEncoder) void {
        for (f.mats) |m| ex.useMatrix(e, m.matrix);
    }

    /// The merged gate+up call: both halves of every pick and the shared pair, `no_epilogue` leaving f32 sums.
    pub fn gateUp(f: *const Fixture, e: mtl.ComputeEncoder, y: mtl.Buffer, scratch: mtl.Buffer, no_epilogue: bool) !void {
        const count = 2 * routed + 2 * tokens;
        const rows = ex.Rows{ .count = count, .x_stride = hidden + 16, .y_stride = mid, .input_ids = .{ .buffer = f.fused_sources }, .input_rows = tokens, .output_rows = count, .routes = .{ .ids = .{ .buffer = f.fused_ids }, .repeat = 1 }, .no_epilogue = no_epilogue };
        try ex.encodeIndexed(&f.p, e, .{ .descriptors = .{ .buffer = f.fused }, .count = 2 * experts + 2, .k = hidden, .n = mid }, .{ .buffer = f.x, .offset = 32 }, .{ .buffer = y }, .{ .buffer = scratch }, rows);
        e.barrier();
    }
};

fn bytesOf(device: mtl.Device, n: usize) !mtl.Buffer {
    const b = try device.buffer(n, mtl.ResourceOptions.shared);
    @memset(b.slice(u8, n), 0);
    return b;
}

test "the fused gate+up pass keeps every bit of rotate_out, both SwiGLUs and the down rotate_in" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]u16{ 6, 10, 12 }) |halves| {
        var f = try Fixture.init(device, halves);
        defer f.deinit();
        const count = 2 * routed + 2 * tokens;
        const scratch_bytes = try ex.scratchBytes(hidden, mid, count);
        const gate = try bytesOf(device, count * mid * 2);
        defer gate.deinit();
        const scratch_ref = try bytesOf(device, scratch_bytes);
        defer scratch_ref.deinit();
        const scratch_shared = try bytesOf(device, scratch_bytes);
        defer scratch_shared.deinit();
        const scratch_fused = try bytesOf(device, scratch_bytes);
        defer scratch_fused.deinit();
        const got = try bytesOf(device, (routed + tokens + 3) * mid * 2);
        defer got.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        f.use(e);
        try f.gateUp(e, gate, scratch_ref, false);
        const up_routed = ex.View{ .buffer = gate, .offset = routed * mid * 2 };
        const shared_gate = ex.View{ .buffer = gate, .offset = 2 * routed * mid * 2 };
        const shared_up = ex.View{ .buffer = gate, .offset = (2 * routed + tokens) * mid * 2 };
        ex.encodeSwiGLUPreRotate(&f.p, e, .{ .buffer = gate }, up_routed, .{ .buffer = f.weights }, .{ .buffer = f.route_ids }, null, .{ .buffer = f.down }, .{ .buffer = scratch_ref }, .{ .rows = routed, .k = mid, .input_stride = mid, .output_stride = mid, .experts = experts });
        ex.encodeSwiGLUPreRotate(&f.p, e, shared_gate, shared_up, .{ .buffer = f.weights }, .{ .buffer = f.zero_ids }, null, .{ .buffer = f.shared_down }, .{ .buffer = scratch_shared }, .{ .rows = tokens, .k = mid, .input_stride = mid, .output_stride = mid, .weighted = false, .experts = 1 });
        e.barrier();
        try f.gateUp(e, gate, scratch_fused, true);
        const sums = ex.sumsView(.{ .buffer = scratch_fused }, count, hidden);
        ex.encodeGateUpSwiGLU(&f.p, e, sums, .{ .buffer = f.fused_ids }, .{ .buffer = f.weights }, .{ .buffer = f.route_ids }, .{ .buffer = f.fused }, .{ .buffer = f.down }, .{ .buffer = f.shared_down }, .{ .buffer = got }, .{ .rows = routed, .shared_rows = tokens, .k = mid, .experts = experts, .fused_experts = 2 * experts + 2, .shared_dst = routed + 3 });
        try complete(cb, e);
        const want_routed = scratch_ref.slice(u16, routed * mid);
        const want_shared = scratch_shared.slice(u16, tokens * mid);
        const have = got.slice(u16, (routed + tokens + 3) * mid);
        try std.testing.expectEqualSlices(u16, want_routed, have[0 .. routed * mid]);
        try std.testing.expectEqualSlices(u16, want_shared, have[(routed + 3) * mid ..][0 .. tokens * mid]);
    }
}

test "the shared expert's down inside the routed down call keeps every bit of the two separate calls" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]u16{ 4, 6, 8, 10, 12 }) |halves| {
        var f = try Fixture.init(device, halves);
        defer f.deinit();
        const count = 2 * routed + 2 * tokens;
        const scratch_bytes = try ex.scratchBytes(hidden, mid, count);
        const gate = try bytesOf(device, count * mid * 2);
        defer gate.deinit();
        const scratch_ref = try bytesOf(device, scratch_bytes);
        defer scratch_ref.deinit();
        const scratch_shared = try bytesOf(device, scratch_bytes);
        defer scratch_shared.deinit();
        const scratch_fused = try bytesOf(device, scratch_bytes);
        defer scratch_fused.deinit();
        const scratch_down = try bytesOf(device, try ex.scratchBytes(mid, out_n, routed + tokens));
        defer scratch_down.deinit();
        const routed_ref = try bytesOf(device, routed * out_n * 2);
        defer routed_ref.deinit();
        const shared_ref = try bytesOf(device, tokens * out_n * 2);
        defer shared_ref.deinit();
        const both = try bytesOf(device, (routed + tokens) * out_n * 2);
        defer both.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        f.use(e);
        // The separate calls: the gate+up epilogue and SwiGLUs, the routed down, then the shared down on its own.
        try f.gateUp(e, gate, scratch_ref, false);
        const up_routed = ex.View{ .buffer = gate, .offset = routed * mid * 2 };
        const shared_gate = ex.View{ .buffer = gate, .offset = 2 * routed * mid * 2 };
        const shared_up = ex.View{ .buffer = gate, .offset = (2 * routed + tokens) * mid * 2 };
        ex.encodeSwiGLUPreRotate(&f.p, e, .{ .buffer = gate }, up_routed, .{ .buffer = f.weights }, .{ .buffer = f.route_ids }, null, .{ .buffer = f.down }, .{ .buffer = scratch_ref }, .{ .rows = routed, .k = mid, .input_stride = mid, .output_stride = mid, .experts = experts });
        ex.encodeSwiGLUPreRotate(&f.p, e, shared_gate, shared_up, .{ .buffer = f.weights }, .{ .buffer = f.zero_ids }, null, .{ .buffer = f.shared_down }, .{ .buffer = scratch_shared }, .{ .rows = tokens, .k = mid, .input_stride = mid, .output_stride = mid, .weighted = false, .experts = 1 });
        e.barrier();
        const down_rows = ex.Rows{ .count = routed, .x_stride = mid, .y_stride = out_n, .routes = .{ .ids = .{ .buffer = f.route_ids }, .repeat = 1 }, .input_rows = routed, .prerotated = true };
        try ex.encodeIndexed(&f.p, e, .{ .descriptors = .{ .buffer = f.down }, .count = experts, .k = mid, .n = out_n }, .{ .buffer = scratch_ref }, .{ .buffer = routed_ref }, .{ .buffer = scratch_ref }, down_rows);
        try ex.encode(&f.p, e, f.mats[3 * experts + 2].matrix, .{ .buffer = scratch_shared }, .{ .buffer = shared_ref }, .{ .buffer = scratch_shared }, .{ .count = tokens, .x_stride = mid, .y_stride = out_n, .prerotated = true });
        // The fused calls: the shared rows follow the routed ones, and the shared id follows the picks.
        try f.gateUp(e, gate, scratch_fused, true);
        const sums = ex.sumsView(.{ .buffer = scratch_fused }, count, hidden);
        const shared_table = ex.View{ .buffer = f.down_shared, .offset = experts * @sizeOf(ex.ExpertDescriptor) };
        ex.encodeGateUpSwiGLU(&f.p, e, sums, .{ .buffer = f.fused_ids }, .{ .buffer = f.weights }, .{ .buffer = f.route_ids }, .{ .buffer = f.fused }, .{ .buffer = f.down_shared }, shared_table, .{ .buffer = scratch_down }, .{ .rows = routed, .shared_rows = tokens, .k = mid, .experts = experts, .fused_experts = 2 * experts + 2, .shared_dst = routed, .shared_id = experts });
        e.barrier();
        const joined = ex.Rows{ .count = routed + tokens, .x_stride = mid, .y_stride = out_n, .routes = .{ .ids = .{ .buffer = f.route_ids }, .repeat = 1 }, .input_rows = routed + tokens, .prerotated = true };
        try ex.encodeIndexed(&f.p, e, .{ .descriptors = .{ .buffer = f.down_shared }, .count = experts + 1, .k = mid, .n = out_n }, .{ .buffer = scratch_down }, .{ .buffer = both }, .{ .buffer = scratch_down }, joined);
        try complete(cb, e);
        const have = both.slice(u16, (routed + tokens) * out_n);
        try std.testing.expectEqualSlices(u16, routed_ref.slice(u16, routed * out_n), have[0 .. routed * out_n]);
        try std.testing.expectEqualSlices(u16, shared_ref.slice(u16, tokens * out_n), have[routed * out_n ..]);
    }
}
