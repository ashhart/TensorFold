//! A full draft block can extend past the target's last 4,096-key attention chunk.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const block = @import("dspark_block.zig");
const quant = @import("dspark_quant.zig");
const base = @import("dspark_hold_test.zig");

test "lfm2 dspark: attention past maximum target context preserves the next scratch row and includes the tail" {
    var r = try base.Rig.init();
    defer r.deinit();
    // The anchor is 131,068 and the configured block has nine rows, even for a one-token grant.
    const keys = st.max_context + 5;
    const cap = keys + 32;
    const width = c.query_heads * c.head_dim;
    const kv_words = c.kv_heads * cap * c.head_dim;
    const guard: u32 = 0x7fc12345;
    const q = try r.buffer(u16, 8 * width);
    defer q.deinit();
    const k = try r.buffer(u16, kv_words);
    defer k.deinit();
    const v = try r.buffer(u16, kv_words);
    defer v.deinit();
    @memset(k.slice(u16, kv_words), quant.toBf16(1));
    // Zero Q makes attention uniform. Only the extra chunk has nonzero V, so omitting it changes output.
    for (0..c.kv_heads) |h| {
        const value = quant.toBf16(@as(f32, @floatFromInt((h + 1) * 4096)));
        @memset(v.slice(u16, kv_words)[(h * cap + st.max_context) * c.head_dim ..][0 .. 5 * c.head_dim], value);
    }
    const pm = try r.buffer(u32, 2 * block.part_row);
    defer pm.deinit();
    const pl = try r.buffer(u32, 2 * block.part_row);
    defer pl.deinit();
    const po = try r.buffer(u32, 2 * block.part_row * c.head_dim);
    defer po.deinit();
    for ([_]mtl.Buffer{ pm, pl }) |b| @memset(b.slice(u32, 2 * block.part_row), guard);
    @memset(po.slice(u32, 2 * block.part_row * c.head_dim), guard);
    const y = try r.buffer(u16, 2 * width);
    defer y.deinit();
    @memset(y.slice(u16, 2 * width), 0x7fc1);
    const Args = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };
    const args: Args = .{ .len = keys - 1, .cap = cap, .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .subs = st.subs };
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(r.k.get("lfm2_attn_block"));
    for ([_]mtl.Buffer{ q, k, v }, 0..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(args, 3);
    for ([_]mtl.Buffer{ pm, pl, po }, 4..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(@as(u32, 1), 7);
    const used = (keys + st.chunk_keys - 1) / st.chunk_keys * st.subs;
    e.dispatchGroups(mtl.Size.of(1, c.kv_heads, used), mtl.Size.of(128, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    // Check before merging: the old stride overwrites this allocated guard row, without a buffer overrun.
    for ([_]mtl.Buffer{ pm, pl }) |b| for (b.slice(u32, 2 * block.part_row)[block.part_row..]) |word| {
        if (word != guard) {
            std.debug.print("DraftAttentionPartialOverwritesNextRow: guard {x}, got {x}\n", .{ guard, word });
            return error.DraftAttentionPartialOverwritesNextRow;
        }
    };
    for (po.slice(u32, 2 * block.part_row * c.head_dim)[block.part_row * c.head_dim ..]) |word| {
        if (word != guard) {
            std.debug.print("DraftAttentionPartialOverwritesNextRow: guard {x}, got {x}\n", .{ guard, word });
            return error.DraftAttentionPartialOverwritesNextRow;
        }
    }
    const merged = r.queue.commandBuffer();
    const me = merged.compute(.serial);
    me.setPipeline(r.k.get("lfm2_attn_block_merge"));
    for ([_]mtl.Buffer{ pm, pl, po }, 0..) |b, i| me.setBuffer(b, 0, i);
    me.setValue(args, 3);
    me.setBuffer(y, 0, 4);
    me.dispatchGroups(mtl.Size.of(c.query_heads, 1, 1), mtl.Size.of(32, 1, 1));
    me.end();
    merged.commit();
    merged.wait();
    if (merged.failure()) |_| return error.GpuFailed;
    for (0..c.query_heads) |h| {
        const kv = h / (c.query_heads / c.kv_heads);
        const mean = @as(f32, @floatFromInt(5 * (kv + 1) * 4096)) / @as(f32, keys);
        const expected = quant.toBf16(mean);
        for (y.slice(u16, 2 * width)[h * c.head_dim ..][0..c.head_dim]) |word| {
            try std.testing.expectEqual(expected, word);
        }
    }
    for (y.slice(u16, 2 * width)[width..]) |word| try std.testing.expectEqual(@as(u16, 0x7fc1), word);
}
