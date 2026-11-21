//! The batched decode window (attention_rows.zig) against the per-slot chain it replaces: bit-identical, slot by slot.
const std = @import("std");
const mtl = @import("metal");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const View = @import("view.zig").View;

const slots = 4;
const positions = [slots]u32{ 5, 130, 700, 16700 }; // before the ring fills, wrapped, past the pick budget, a long prompt
const main_rows = 8400;
const ring_at = 2 * 128 * 528; // the layer's ring inside a slot's live buffer
const width = 512;

fn view(b: mtl.Buffer) View {
    return .{ .buffer = b };
}
fn at(b: mtl.Buffer, offset: usize) View {
    return .{ .buffer = b, .offset = offset };
}
fn submit(cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}
fn draw(s: *u64) u32 {
    s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
    return @intCast(s.* >> 32);
}
fn bf(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn fill(b: mtl.Buffer, len: usize, state: *u64) void {
    for (b.contents()[0..len]) |*x| x.* = @intCast(draw(state) & 0xff);
}
fn fillBf(b: mtl.Buffer, count: usize, state: *u64) void {
    for (b.slice(u16, count)) |*x| x.* = bf((@as(f32, @floatFromInt(draw(state) % 2000)) - 1000.0) / 995.0);
}
// A key row's scale bytes stay small: e8m0 near 2^0 (base 124) for the window, e4 near 2^-3 (0x20) for compressed.
fn finishPacked(b: mtl.Buffer, rows: usize, stride: usize, body: usize, base: u32, state: *u64) void {
    if (body == 512) for (0..rows) |r| {
        for (b.contents()[r * stride ..][0..body]) |*x|
            if ((x.* & 0x7f) == 0x7f) {
                x.* = (x.* & 0x80) | 0x7e;
            };
    };
    for (0..rows) |r| for (b.contents()[r * stride + body .. (r + 1) * stride]) |*x| {
        x.* = @intCast(base + draw(state) % 4);
    };
}

fn decodeWindows(comptime heads: usize) !void {
    const q_stride = heads * width;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var sparse = try csa2.Pipelines.load(device);
    defer sparse.deinit();
    var cache = try kv.Pipelines.load(device);
    defer cache.deinit();
    var rot = try rope.Pipelines.load(device, 17000, 64, .{ .theta = 10000 }, .{ .theta = 160000 });
    defer rot.deinit();
    if (!sparse.rowsReady(heads, width)) return error.SkipZigTest;
    const opts = mtl.ResourceOptions.shared;
    var state: u64 = 0x9e3779b97f4a7c15;

    var live: [slots]mtl.Buffer = undefined;
    var main: [slots]mtl.Buffer = undefined;
    var sel: [slots]mtl.Buffer = undefined;
    for (0..slots) |s| {
        live[s] = try device.buffer(3 * 128 * 528, opts);
        main[s] = try device.buffer(main_rows * 288, opts);
        sel[s] = try device.buffer(512 * 4, opts);
        fill(live[s], 3 * 128 * 528, &state);
        finishPacked(live[s], 3 * 128, 528, 512, 124, &state);
        fill(main[s], main_rows * 288, &state);
        finishPacked(main[s], main_rows, 288, 256, 0x20, &state);
        const comp: u32 = (positions[s] + 1) / 2;
        for (sel[s].slice(i32, 512), 0..) |*x, j| {
            const pick: i32 = @intCast(draw(&state) % comp);
            x.* = if (j >= comp) -1 else if (j % 11 == 5) -1 else pick;
        }
    }
    defer for (0..slots) |s| {
        live[s].deinit();
        main[s].deinit();
        sel[s].deinit();
    };
    const chunk = try device.buffer(slots * 528, opts);
    defer chunk.deinit();
    fill(chunk, slots * 528, &state);
    finishPacked(chunk, slots, 528, 512, 124, &state);
    const raw = try device.buffer(slots * width * 2, opts);
    defer raw.deinit();
    fillBf(raw, slots * width, &state);
    const norm_w = try device.buffer(width * 2, opts);
    defer norm_w.deinit();
    fillBf(norm_w, width, &state);
    const q_a = try device.buffer(slots * q_stride * 2, opts);
    defer q_a.deinit();
    const q_b = try device.buffer(slots * q_stride * 2, opts);
    defer q_b.deinit();
    fillBf(q_a, slots * q_stride, &state);
    @memcpy(q_b.contents()[0 .. slots * q_stride * 2], q_a.contents()[0 .. slots * q_stride * 2]);
    const sink = try device.buffer(heads * 4, opts);
    defer sink.deinit();
    for (sink.slice(f32, heads)) |*x| x.* = @as(f32, @floatFromInt(draw(&state) % 100)) / 100.0;

    // The per-forward table, laid out as attention_common.RowsScratch.
    const tab = try device.buffer(544, opts);
    defer tab.deinit();
    const pos_t = tab.slice(u32, 8);
    const addr_t = @as([*]u64, @ptrCast(@alignCast(tab.contents() + 32)))[0..32];
    const mains_t = @as([*]u64, @ptrCast(@alignCast(tab.contents() + 288)))[0..32];
    @memset(tab.contents()[0..544], 0);
    for (0..slots) |s| {
        pos_t[s] = positions[s];
        addr_t[4 * s] = live[s].gpuAddress();
        addr_t[4 * s + 2] = sel[s].gpuAddress();
        mains_t[4 * s + 2] = main[s].gpuAddress();
    }
    const pos_v = at(tab, 0);
    const addr_v = at(tab, 32);
    const mains_v = at(tab, 288);

    // -- the sparse scan, merge ----------------------------------------------------------------------------------
    const dense_one = try device.buffer((128 + 1 + main_rows) * width * 2, opts);
    defer dense_one.deinit();
    const dense_all = try device.buffer(slots * csa2.rows_dense * width * 2, opts);
    defer dense_all.deinit();
    const parts = try device.buffer(slots * 8 * (heads / 16) * 4 * 32 * 68 * 4, opts);
    defer parts.deinit();
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..slots) |s| {
            const comp: u32 = (positions[s] + 1) / 2;
            try sparse.encode(enc, at(q_a, s * q_stride * 2), at(live[s], ring_at), at(chunk, s * 528), view(main[s]), view(sel[s]), view(sink), at(q_a, s * q_stride * 2), view(dense_one), view(parts), .{
                .rows = 1,
                .heads = heads,
                .width = width,
                .q_stride = q_stride,
                .position = positions[s],
                .chunk_rows = 1,
                .ratio = 2,
                .compressed_rows = comp,
                .topk = @min(512, comp),
                .splits = 8,
            });
        }
        try submit(cb, enc);
    }
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        sparse.encodeRows(enc, view(q_b), view(sink), view(chunk), view(dense_all), view(parts), pos_v, addr_v, mains_v, .{
            .rows = slots,
            .heads = heads,
            .width = width,
            .q_stride = q_stride,
            .ring_off = ring_at,
            .owner = 2,
            .ratio = 2,
            .splits = 8,
        });
        try submit(cb, enc);
    }
    try std.testing.expectEqualSlices(u8, q_a.contents()[0 .. slots * q_stride * 2], q_b.contents()[0 .. slots * q_stride * 2]);
    for (q_a.slice(u16, slots * q_stride)) |bits|
        try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(@as(u32, bits) << 16))));

    // -- the rotation, both directions -----------------------------------------------------------------------------
    for (0..slots) |s| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        inline for (.{ false, true }) |inverse| {
            try rot.encode(enc, at(q_a, s * q_stride * 2), .{ .rows = 1, .heads = heads, .width = width, .rotary = 64, .stride = q_stride, .position = positions[s], .inverse = @intFromBool(inverse), .spectrum = .compressed });
        }
        try submit(cb, enc);
    }
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        inline for (.{ 0, 1 }) |inverse| {
            rot.encodeRows(enc, view(q_b), pos_v, .{ .rows = slots, .heads = heads, .width = width, .rotary = 64, .stride = q_stride, .inverse = inverse }, .compressed);
        }
        try submit(cb, enc);
    }
    try std.testing.expectEqualSlices(u8, q_a.contents()[0 .. slots * q_stride * 2], q_b.contents()[0 .. slots * q_stride * 2]);

    // -- the window row's norm + rope + pack, and its publish into each slot's ring --------------------------------
    const chunk_a = try device.buffer(slots * 528, opts);
    defer chunk_a.deinit();
    const chunk_b = try device.buffer(slots * 528, opts);
    defer chunk_b.deinit();
    @memset(chunk_a.contents()[0 .. slots * 528], 0);
    @memset(chunk_b.contents()[0 .. slots * 528], 0);
    const glue: kv.Glue = .{ .rows = 1, .width = width, .stride = width, .packed_stride = 528, .block = 32, .position = 0, .rotary = 64, .write_y = 0, .eps = 1e-6 };
    const factors: View = .{ .buffer = rot.factors, .offset = @as(usize, rot.capacity) * rot.rotary * @sizeOf(f32) };
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..slots) |s| {
            var g = glue;
            g.position = positions[s];
            try cache.encodeGlue(enc, at(raw, s * width * 2), view(norm_w), factors, at(chunk_a, s * 528), at(chunk_a, s * 528), g);
        }
        var g = glue;
        g.rows = slots;
        try cache.encodeGlueRows(enc, view(raw), view(norm_w), factors, view(chunk_b), pos_v, g);
        try submit(cb, enc);
    }
    try std.testing.expectEqualSlices(u8, chunk_a.contents()[0 .. slots * 528], chunk_b.contents()[0 .. slots * 528]);

    var ring_ref: [slots]mtl.Buffer = undefined;
    for (0..slots) |s| {
        ring_ref[s] = try device.buffer(3 * 128 * 528, opts);
        @memcpy(ring_ref[s].contents()[0 .. 3 * 128 * 528], live[s].contents()[0 .. 3 * 128 * 528]);
    }
    defer for (0..slots) |s| ring_ref[s].deinit();
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..slots) |s| try cache.publishWindow(enc, at(chunk_b, s * 528), at(ring_ref[s], ring_at), 1, positions[s]);
        cache.publishRows(enc, view(chunk_b), addr_v, pos_v, ring_at, slots);
        try submit(cb, enc);
    }
    for (0..slots) |s| try std.testing.expectEqualSlices(u8, ring_ref[s].contents()[0 .. 3 * 128 * 528], live[s].contents()[0 .. 3 * 128 * 528]);
    std.debug.print("[batched decode windows] {d} slots bit-exact: scan+merge, rope both ways, glue, publish\n", .{slots});
}

test "batched decode windows match the per-slot chain bit for bit" {
    inline for (.{ 16, 32, 64 }) |heads| try decodeWindows(heads);
}
