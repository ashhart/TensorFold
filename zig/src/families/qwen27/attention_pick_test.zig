//! q27_attn_prompt_scan and _pick on this GPU: the masked prompt kernel iff a value some row can't see isn't finite.
const std = @import("std");
const mtl = @import("metal");
const attn = @import("attention.zig");
const simd_attention = @import("simd_attention.zig");

const kv_heads = 4;
const cap = 48;
const dim = 256;
const grid = [3]u32{ 7, 8, 9 };

const Rig = struct {
    queue: mtl.Queue,
    pipes: [2]mtl.Pipeline,
    values: mtl.Buffer,
    caches: mtl.Buffer,
    pick: mtl.Buffer,

    /// One launch of `rows` rows at `start`, every value 1 but `plant` (kv head, key, dim, bits); true: masked.
    fn run(r: *Rig, start: i32, rows: i32, plant: ?[4]u32) !bool {
        const v = r.values.slice(u16, kv_heads * cap * dim);
        @memset(v, 0x3f80);
        if (plant) |p| v[(p[0] * cap + p[1]) * dim + p[2]] = @intCast(p[3]);
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        e.useResource(r.values, mtl.ResourceUsage.read); // read through the caches table, as Attention.encode declares it
        const args = attn.PromptArgs{ .start = start, .rows = rows, .chunks = 1, .rp = 16, .key_stride = cap * dim, .value_stride = cap * dim, .scale2 = 0, .kv_heads = kv_heads };
        attn.Attention.encodePick(e, r.pipes, .{ .buffer = r.caches }, args, r.pick, grid);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        const w = r.pick.slice(u32, 10);
        try std.testing.expectEqual(@as(u32, 0), w[0]); // cleared for the next launch
        const masked = w[7] != 0;
        try std.testing.expectEqualSlices(u32, if (masked) &[3]u32{ 0, 0, 0 } else &grid, w[4..7]);
        try std.testing.expectEqualSlices(u32, if (masked) &grid else &[3]u32{ 0, 0, 0 }, w[7..10]);
        return masked;
    }
};

test "prompt pick: one non-finite value at every key, head and kind; masked iff start < key < start + rows" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const a = std.testing.allocator;
    const text = try simd_attention.grouped(a, device.tensorUnits(), true, 6, kv_heads, false);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var r: Rig = .{ .queue = try device.queue(), .pipes = undefined, .values = undefined, .caches = undefined, .pick = undefined };
    defer r.queue.deinit();
    r.pipes = .{ try mtl.Pipeline.init(device, lib, "q27_attn_prompt_scan", false), try mtl.Pipeline.init(device, lib, "q27_attn_prompt_pick", false) };
    defer for (r.pipes) |p| p.deinit();
    r.values = try device.buffer(kv_heads * cap * dim * 2, mtl.ResourceOptions.shared);
    defer r.values.deinit();
    r.caches = try device.buffer(16, mtl.ResourceOptions.shared);
    defer r.caches.deinit();
    r.caches.slice(u64, 2)[0] = r.values.gpuAddress(); // keys: not read
    r.caches.slice(u64, 2)[1] = r.values.gpuAddress();
    r.pick = try device.buffer(48, mtl.ResourceOptions.shared);
    defer r.pick.deinit();
    @memset(r.pick.slice(u32, 12), 0);
    const kinds = [_]u32{ 0x7f80, 0xff80, 0x7fc0, 0xffc1 }; // +Inf, -Inf, NaN, a negative NaN
    var masked: usize = 0;
    // launches run back to back on one flag, so a launch that fails to clear it shows in the next finite one
    for ([_]i32{ 1, 2, 3, 8, 33, 40 }) |rows| for ([_]i32{ 0, 5 }) |start| {
        if (start + rows > cap) continue;
        try std.testing.expect(!try r.run(start, rows, null));
        var key: i32 = @max(start - 1, 0);
        while (key <= @min(start + rows, cap - 1)) : (key += 1) for (0..kv_heads) |h| {
            const k: u32 = @intCast(key);
            const want = start < key and key < start + rows;
            const got = try r.run(start, rows, .{ @intCast(h), k, (k * 37 + @as(u32, @intCast(h)) * 11) % dim, kinds[(k + h) % kinds.len] });
            if (got != want) std.debug.print("rows {d} start {d} key {d} head {d}: masked {}\n", .{ rows, start, key, h, got });
            try std.testing.expectEqual(want, got);
            if (got) masked += 1;
            try std.testing.expect(!try r.run(start, rows, null));
        };
    };
    try std.testing.expect(masked > 0);
}
