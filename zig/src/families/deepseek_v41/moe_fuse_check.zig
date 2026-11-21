//! The decode MoE's fused down epilogue and combine against the separate rotate_out calls and the combine kernel.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("core").exl3_projection;
const moe = @import("core").exl3_projection_gpu_test.moe;
const ops = @import("ops.zig");
const support = @import("test_support.zig");
const view = support.view;
const done = support.done;

const tokens = moe.tokens;
const picks = moe.picks;
const routed = moe.routed;
const width = moe.out_n;

fn zeroed(device: mtl.Device, bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared);
    @memset(b.slice(u8, bytes), 0);
    return b;
}

test "V4.1 fused down rotate_out and combine keeps every bit of the separate calls, ties and poisoned rows included" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ops_p = try ops.Pipelines.load(device);
    defer ops_p.deinit();
    // Token 0's picks and token 1's: distinct, tied, and one past 383 (the combine's NaN row).
    const cases = [_][routed]u32{ .{ 0, 1, 2, 3, 1, 0 }, .{ 1, 1, 0, 2, 3, 2 }, .{ 2, 0, 1, 3, 3, 400 } };
    for ([_]u16{ 6, 10, 12 }) |halves| for (cases) |ids| {
        var f = try moe.Fixture.init(device, halves);
        defer f.deinit();
        @memcpy(f.route_ids.slice(u32, routed + tokens)[0..routed], &ids);
        // The shared rows' id after the picks: what the fused gate+up pass writes.
        @memset(f.route_ids.slice(u32, routed + tokens)[routed..], moe.experts);
        const k = moe.mid;
        const scratch_bytes = try ex.scratchBytes(k, width, routed + tokens);
        const rotated_a = try zeroed(device, scratch_bytes);
        defer rotated_a.deinit();
        const rotated_b = try zeroed(device, scratch_bytes);
        defer rotated_b.deinit();
        const rotated_f = try zeroed(device, scratch_bytes);
        defer rotated_f.deinit();
        var prng = std.Random.DefaultPrng.init(0xd0d0 + @as(u64, halves));
        for (rotated_a.slice(f16, routed * k)) |*v| v.* = @floatCast((prng.random().float(f32) - 0.5) * 2);
        for (rotated_b.slice(f16, tokens * k)) |*v| v.* = @floatCast((prng.random().float(f32) - 0.5) * 2);
        @memcpy(rotated_f.slice(f16, routed * k), rotated_a.slice(f16, routed * k));
        @memcpy(rotated_f.slice(f16, (routed + tokens) * k)[routed * k ..], rotated_b.slice(f16, tokens * k));
        const routed_out = try zeroed(device, (routed + tokens) * width * 2);
        defer routed_out.deinit();
        const shared_out = try zeroed(device, tokens * width * 2);
        defer shared_out.deinit();
        const want = try zeroed(device, tokens * width * 2);
        defer want.deinit();
        const got = try zeroed(device, tokens * width * 2);
        defer got.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        f.use(e);
        const down_rows = ex.Rows{ .count = routed, .x_stride = k, .y_stride = width, .routes = .{ .ids = .{ .buffer = f.route_ids }, .repeat = 1 }, .input_rows = routed, .prerotated = true };
        try ex.encodeIndexed(&f.p, e, .{ .descriptors = .{ .buffer = f.down }, .count = moe.experts, .k = k, .n = width }, .{ .buffer = rotated_a }, .{ .buffer = routed_out }, .{ .buffer = rotated_a }, down_rows);
        try ex.encode(&f.p, e, f.mats[3 * moe.experts + 2].matrix, .{ .buffer = rotated_b }, .{ .buffer = shared_out }, .{ .buffer = rotated_b }, .{ .count = tokens, .x_stride = k, .y_stride = width, .prerotated = true });
        ops.encodeCombine(e, &ops_p, view(routed_out), view(shared_out), view(f.route_ids), view(want), .{ .rows = tokens, .width = width, .routed_row_stride = picks * width, .slot_stride = width, .shared_stride = width, .y_stride = width, .id_stride = picks });
        e.barrier();
        const joined = ex.Rows{ .count = routed + tokens, .x_stride = k, .y_stride = width, .routes = .{ .ids = .{ .buffer = f.route_ids }, .repeat = 1 }, .input_rows = routed + tokens, .prerotated = true, .no_epilogue = true };
        try ex.encodeIndexed(&f.p, e, .{ .descriptors = .{ .buffer = f.down_shared }, .count = moe.experts + 1, .k = k, .n = width }, .{ .buffer = rotated_f }, .{ .buffer = routed_out }, .{ .buffer = rotated_f }, joined);
        ex.encodeDownCombine(&f.p, e, ex.sumsView(.{ .buffer = rotated_f }, routed + tokens, k), .{ .buffer = f.route_ids }, .{ .buffer = f.down_shared }, .{ .buffer = got }, .{ .tokens = tokens, .picks = picks, .width = width, .experts = moe.experts, .y_stride = width, .shared_row = routed });
        try done(cb, e);
        try std.testing.expectEqualSlices(u16, want.slice(u16, tokens * width), got.slice(u16, tokens * width));
    };
}
