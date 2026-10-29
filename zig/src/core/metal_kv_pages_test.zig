//! The paged arena on the GPU: rows written through two streams' tables into scattered pages read back bit-equal.
const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv_pool.zig");
const pages = @import("metal_kv_pages.zig");

const words = 16; // u32 words a row
const row_bytes = words * 4;

/// Scatter rows into pages and gather them back, both through tf_page_offset.
const source = pages.address_msl ++
    \\kernel void tf_kvp_scatter(device const uint* src [[buffer(0)]], device uchar* arena [[buffer(1)]],
    \\    device const uint* table [[buffer(2)]], constant uint& rows [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    \\  if (g.y >= rows) return;
    \\  device uint* row = (device uint*)(arena + tf_page_offset(table, g.y, 256 * 64, 64));
    \\  row[g.x] = src[g.y * 16 + g.x];
    \\}
    \\kernel void tf_kvp_gather(device const uchar* arena [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    device const uint* table [[buffer(2)]], constant uint& rows [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    \\  if (g.y >= rows) return;
    \\  device const uint* row = (device const uint*)(arena + tf_page_offset(table, g.y, 256 * 64, 64));
    \\  out[g.y * 16 + g.x] = row[g.x];
    \\}
;

const Pipes = struct { scatter: mtl.Pipeline, gather: mtl.Pipeline };

fn run(queue: mtl.Queue, pipe: mtl.Pipeline, a: mtl.Buffer, b: mtl.Buffer, p: *pages.Pages, lane: usize, rows: u32) !void {
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    enc.setBuffer(a, 0, 0);
    enc.setBuffer(b, 0, 1);
    enc.setBuffer(p.tables.buf.?, p.tables.offset(lane), 2);
    enc.setValue(rows, 3);
    enc.dispatchThreads(mtl.Size.of(words, rows, 1), mtl.Size.of(words, 4, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

test "rows through two streams' tables into scattered pages: read back, gathered and blitted bit-equal" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest; // a recording Metal stand-in has no device
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = row_bytes, .page_tokens = pages.page_tokens, .snap_bytes = 0 };
    const p = try pages.Pages.init(gpa, device, l, 3, 12, 6 * pages.page_tokens, mtl.ResourceOptions.shared);
    defer p.deinit();
    // streams 0 and 1 take pages in turn, stream 0 ends, and stream 2 takes its freed, scattered ids
    try p.admit(0, 5);
    try p.admit(1, 6);
    for (1..6) |n| {
        try p.cover(0, n * pages.page_tokens);
        try p.cover(1, n * pages.page_tokens);
    }
    p.finish(0);
    const rows: [2]u32 = .{ 1000, 1500 }; // neither a whole number of pages
    const lanes: [2]usize = .{ 2, 1 };
    try p.admit(2, l.pages(rows[0]));
    try p.cover(2, rows[0]);
    try p.cover(1, rows[1]);
    const ids2 = p.ids(2);
    try std.testing.expect(ids2[1] != ids2[0] + 1); // not one contiguous span
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipes: Pipes = .{ .scatter = try mtl.Pipeline.init(device, lib, "tf_kvp_scatter", false), .gather = try mtl.Pipeline.init(device, lib, "tf_kvp_gather", false) };
    defer pipes.scatter.deinit();
    defer pipes.gather.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    for (lanes, rows) |lane, n| {
        const src = try device.buffer(n * row_bytes, mtl.ResourceOptions.shared);
        defer src.deinit();
        prng.random().bytes(src.slice(u8, n * row_bytes));
        try run(queue, pipes.scatter, src, p.arena.buf.?, p, lane, n);
        // each row sits at its page's id x page bytes + its row there
        const arena = p.arena.buf.?.slice(u8, p.arena.pages * l.pageBytes());
        const want = src.slice(u8, n * row_bytes);
        for (0..n) |r| {
            const at = p.arena.offset(p.ids(lane)[r / pages.page_tokens]) + (r % pages.page_tokens) * row_bytes;
            try std.testing.expectEqualSlices(u8, want[r * row_bytes ..][0..row_bytes], arena[at..][0..row_bytes]);
        }
        // a gather kernel through the table rebuilds the contiguous cache
        const out = try device.buffer(n * row_bytes, mtl.ResourceOptions.shared);
        defer out.deinit();
        try run(queue, pipes.gather, p.arena.buf.?, out, p, lane, n);
        try std.testing.expectEqualSlices(u8, want, out.slice(u8, n * row_bytes));
        // and a blit a page in table order: the contiguous layout is the pages back to back
        const flat = try device.buffer(l.pages(n) * l.pageBytes(), mtl.ResourceOptions.shared);
        defer flat.deinit();
        const cb = queue.commandBuffer();
        const blit = cb.blit();
        for (p.ids(lane), 0..) |id, i| blit.copy(p.arena.buf.?, @intCast(p.arena.offset(id)), flat, i * l.pageBytes(), @intCast(l.pageBytes()));
        blit.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        try std.testing.expectEqualSlices(u8, want, flat.slice(u8, n * row_bytes));
    }
    // stream 1's rows, written after stream 2's, left stream 2's pages alone: gather stream 2 again
    {
        const n = rows[0];
        const out = try device.buffer(n * row_bytes, mtl.ResourceOptions.shared);
        defer out.deinit();
        try run(queue, pipes.gather, p.arena.buf.?, out, p, 2, n);
        var check = std.Random.DefaultPrng.init(11);
        const want = try gpa.alloc(u8, n * row_bytes);
        defer gpa.free(want);
        check.random().bytes(want);
        try std.testing.expectEqualSlices(u8, want, out.slice(u8, n * row_bytes));
    }
    p.finish(1);
    p.finish(2);
    try std.testing.expect(p.idle());
}

test "an arena past the device's longest buffer is refused before it is made" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const most = device.maxBufferLength();
    try std.testing.expectError(error.ArenaTooLarge, pages.Arena.init(device, 2, most / 2 + 1, mtl.ResourceOptions.shared));
}

test "pages over a family's own buffers leave them to the family" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = row_bytes, .page_tokens = pages.page_tokens, .snap_bytes = 0 };
    const sized = pages.Pages.bytes(l, 2, 4, 2 * pages.page_tokens);
    const arena = try device.buffer(sized.arena, mtl.ResourceOptions.shared);
    defer arena.deinit();
    const tables = try device.buffer(sized.tables, mtl.ResourceOptions.shared);
    defer tables.deinit();
    const small = try device.buffer(sized.arena - 1, mtl.ResourceOptions.shared);
    defer small.deinit();
    try std.testing.expectError(error.ArenaTooSmall, pages.Pages.initIn(std.testing.allocator, l, 2, 4, 2 * pages.page_tokens, small, tables));
    const p = try pages.Pages.initIn(std.testing.allocator, l, 2, 4, 2 * pages.page_tokens, arena, tables);
    try p.admit(1, 2);
    try p.cover(1, pages.page_tokens + 1);
    try std.testing.expectEqualSlices(u32, p.ids(1), tables.slice(u32, 4)[2..4]); // lane 1's table at its offset
    p.finish(1);
    try std.testing.expect(p.idle());
    p.deinit();
    @memset(arena.slice(u8, sized.arena), 1); // still the family's to use
    try std.testing.expectEqual(sized.arena, arena.length());
}
