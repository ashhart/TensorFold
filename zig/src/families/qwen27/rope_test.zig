//! q27_rope on this GPU: M-RoPE rows with t = h = w keep plain rope's bits; an image row turns pair d by its axis.
const std = @import("std");
const mtl = @import("metal");
const source = @import("qwen_runtime_sources");
const glue = @import("glue.zig");
const axis = @import("core").qwen_vl.rope.axis;

const rows = 4;
const heads = 2;
const width = 256;
const plane = 8; // the position planes' stride, as the frame's capacity

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    x: mtl.Buffer,
    pos: mtl.Buffer,
    y: mtl.Buffer,

    /// Y for X at the planes' positions; h_end = w_end = 0 reads plane 0 alone.
    fn run(r: *Rig, planes: [3][rows]i32, h_end: u32, w_end: u32, rope: @import("yarn.zig").Rope) ![rows * heads * width]u16 {
        const p = r.pos.slice(i32, 3 * plane);
        for (planes, 0..) |row, k| @memcpy(p[k * plane ..][0..rows], &row);
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(r.pipe);
        e.setBuffer(r.x, 0, 0);
        e.setBuffer(r.pos, 0, 1);
        e.setValue(rope.apply(.{ .width = width, .heads = heads, .rows = rows, .rotary = 64, .activation = .bf16, .theta = 1e7, .h_end = h_end, .w_end = w_end, .plane = plane }), 2);
        e.setBuffer(r.y, 0, 3);
        e.dispatchThreads(mtl.Size.of(width, heads, rows), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        var out: [rows * heads * width]u16 = undefined;
        @memcpy(&out, r.y.slice(u16, out.len));
        return out;
    }
};

test "q27_rope: t = h = w is plain rope bit for bit, and an image row's pairs turn by their axis's position" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var r: Rig = .{ .device = device, .queue = try device.queue(), .lib = undefined, .pipe = undefined, .x = undefined, .pos = undefined, .y = undefined };
    defer r.queue.deinit();
    r.lib = try mtl.Library.fromSource(device, source.glue, mtl.CompileOptions.mlx());
    defer r.lib.deinit();
    r.pipe = try mtl.Pipeline.init(device, r.lib, "q27_rope", false);
    defer r.pipe.deinit();
    const n = rows * heads * width;
    r.x = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer r.x.deinit();
    r.pos = try device.buffer(3 * plane * 4, mtl.ResourceOptions.shared);
    defer r.pos.deinit();
    r.y = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer r.y.deinit();
    var rng = std.Random.DefaultPrng.init(0x5eed);
    for (r.x.slice(u16, n)) |*v| v.* = @truncate(@as(u32, @bitCast(rng.random().floatNorm(f32))) >> 16);
    const text = [rows]i32{ 5, 900, 70000, 262143 };
    const plain = try r.run(.{ text, @splat(0), @splat(0) }, 0, 0, .plain);
    const mrope = try r.run(.{ text, text, text }, 33, 30, .plain);
    try std.testing.expectEqualSlices(u16, &plain, &mrope);
    // each row an image row at (t, h, w); a pair's two values equal plain rope's at its axis's position
    const t = [rows]i32{ 10, 300, 4000, 9 };
    const h = [rows]i32{ 13, 302, 4010, 9 };
    const w = [rows]i32{ 17, 301, 4003, 40 };
    const image = try r.run(.{ t, h, w }, 33, 30, .plain);
    const at = [3][rows * heads * width]u16{ try r.run(.{ t, @splat(0), @splat(0) }, 0, 0, .plain), try r.run(.{ h, @splat(0), @splat(0) }, 0, 0, .plain), try r.run(.{ w, @splat(0), @splat(0) }, 0, 0, .plain) };
    for (0..rows * heads) |rh| for (0..width) |x| {
        const i = rh * width + x;
        // past the rotary dims every plane copies
        const k: usize = if (x < 64) axis(@intCast(x % 32), .{ 11, 11, 10 }) else 0;
        try std.testing.expectEqual(at[k][i], image[i]);
    };

    // Static YaRN up to 512K: rotated dims use yarn_freq and yarn_scale; dims >= 64 stay raw passthrough.
    const y_tbl = @import("core").rope_yarn.yarn(1e7, 2.0, 262144);
    const yarn_rope: @import("yarn.zig").Rope = .{ .yarn = .{ .factor = 2.0, .trained = 262144, .scale = y_tbl.scale, .freq = y_tbl.freq } };
    const y_pos = [rows]i32{ 5, 900, 70000, 450000 };
    const y_gpu = try r.run(.{ y_pos, @splat(0), @splat(0) }, 0, 0, yarn_rope);
    const x_host = r.x.slice(u16, n);
    for (0..rows) |row| for (0..heads) |hd| for (0..width) |x| {
        const idx = (row * heads + hd) * width + x;
        if (x >= 64) {
            try std.testing.expectEqual(x_host[idx], y_gpu[idx]);
            continue;
        }
        const second = x >= 32;
        const d = if (second) x - 32 else x;
        const base_idx = (row * heads + hd) * width;
        const v: f32 = @bitCast(@as(u32, x_host[idx]) << 16);
        const partner_u16 = x_host[base_idx + if (second) d else d + 32];
        const pval: f32 = @bitCast(@as(u32, partner_u16) << 16);
        const angle = @as(f32, @floatFromInt(y_pos[row])) * y_tbl.freq[d];
        const cs = @cos(angle) * y_tbl.scale;
        const sn = @sin(angle) * y_tbl.scale;
        const ref_f = if (second) v * cs + pval * sn else v * cs - pval * sn;
        const ref_b: u16 = @truncate((@as(u32, @bitCast(ref_f)) + 0x7fff + ((@as(u32, @bitCast(ref_f)) >> 16) & 1)) >> 16);
        const ulp_ok = ((ref_b & 0x7fff) == 0 and (y_gpu[idx] & 0x7fff) == 0) or
            ((ref_b ^ y_gpu[idx]) < 0x8000 and @abs(@as(i32, ref_b) - @as(i32, y_gpu[idx])) <= 1);
        try std.testing.expect(ulp_ok);
    };
}
