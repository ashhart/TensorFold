//! Qwen3.5-2B's paged keys and attention against the contiguous kernels on one GPU: the same bits, pages shuffled.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

const D = 256;
const kv_heads = 2;
const query_heads = 8;
const page = 256; // TF_PAGE_TOKENS
const layers = 6; // attention layers a page holds
const page_bytes: u64 = layers * 2 * kv_heads * page * D * 2;
const layer = 3; // a layer past the first, so its part's offset counts

fn part(values: bool) usize {
    return (layer * 2 + @as(usize, @intFromBool(values))) * kv_heads * page * D * 2;
}

fn pipe(device: mtl.Device, source: []const u8, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

fn attnPartial() struct { source: []const u8, function: [:0]const u8 } {
    inline for (sources.qwen35.all) |k| if (comptime std.mem.eql(u8, k.key, "attn_partial")) return .{ .source = k.source, .function = k.function };
    unreachable;
}

fn bf16s(rnd: std.Random, out: []u16) void {
    for (out) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

test "paged keys and attention give the contiguous kernels' bits through shuffled pages" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const opts = mtl.ResourceOptions.shared;
    const ap = attnPartial();
    const keys = try pipe(device, sources.qwen35_layout, "qwen35_keys");
    defer keys.deinit();
    const keys_paged = try pipe(device, sources.qwen35_paged, "qwen35_keys_paged");
    defer keys_paged.deinit();
    const attn = try pipe(device, ap.source, ap.function);
    defer attn.deinit();
    const attn_paged = try pipe(device, sources.qwen35_paged, "qwen35_attn_partial_paged");
    defer attn_paged.deinit();
    var prng = std.Random.DefaultPrng.init(35);
    const rnd = prng.random();
    // a stream at 700 rows takes a window of 9 (past a page's end and a chunk's), its cache 3 rows longer
    for ([_][2]u32{ .{ 700, 9 }, .{ 255, 2 }, .{ 0, 16 }, .{ 1024, 1 } }) |shape| {
        const len = shape[0];
        const rows = shape[1];
        const cap = len + rows + 3;
        const npages = (cap + page - 1) / page;
        const arena_pages = npages + 3;
        // the stream's table: its pages in shuffled arena ids
        var ids: [16]u32 = undefined;
        for (ids[0..arena_pages], 0..) |*id, i| id.* = @intCast(i);
        rnd.shuffle(u32, ids[0..arena_pages]);
        const table = try device.buffer(npages * 4, opts);
        defer table.deinit();
        @memcpy(table.slice(u32, npages), ids[0..npages]);
        const arena = try device.buffer(arena_pages * page_bytes, opts);
        defer arena.deinit();
        rnd.bytes(arena.slice(u8, arena_pages * page_bytes)); // stale bytes outside the stream's rows
        const kc = try device.buffer(kv_heads * cap * D * 2, opts);
        defer kc.deinit();
        const vc = try device.buffer(kv_heads * cap * D * 2, opts);
        defer vc.deinit();
        // the rows before `len` in both layouts alike
        bf16s(rnd, kc.slice(u16, kv_heads * cap * D));
        bf16s(rnd, vc.slice(u16, kv_heads * cap * D));
        const a8 = arena.slice(u8, arena_pages * page_bytes);
        for (0..len) |pos| for (0..kv_heads) |h| for ([_]bool{ false, true }, [_]mtl.Buffer{ kc, vc }) |values, src| {
            const at = ids[pos / page] * page_bytes + part(values) + (h * page + pos % page) * D * 2;
            @memcpy(a8[at..][0 .. D * 2], src.slice(u8, kv_heads * cap * D * 2)[(h * cap + pos) * D * 2 ..][0 .. D * 2]);
        };
        // the window's keys and values written by each kernel from the same inputs
        const kin = try device.buffer(rows * kv_heads * D * 2, opts);
        defer kin.deinit();
        const vin = try device.buffer(rows * kv_heads * D * 2, opts);
        defer vin.deinit();
        bf16s(rnd, kin.slice(u16, rows * kv_heads * D));
        bf16s(rnd, vin.slice(u16, rows * kv_heads * D));
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            const dims = [4]u32{ len, cap, 0, 0 };
            e.setPipeline(keys);
            e.setBuffer(kin, 0, 0);
            e.setBuffer(vin, 0, 1);
            e.setBuffer(kc, 0, 2);
            e.setBuffer(vc, 0, 3);
            e.setValue(dims, 4);
            e.dispatchThreads(mtl.Size.of(D, kv_heads, rows), mtl.Size.of(256, 1, 1));
            e.setPipeline(keys_paged);
            e.setBuffer(kin, 0, 0);
            e.setBuffer(vin, 0, 1);
            e.setBuffer(arena, part(false), 2);
            e.setBuffer(arena, part(true), 3);
            e.setValue(dims, 4);
            e.setBuffer(table, 0, 5);
            e.setValue(page_bytes, 6);
            e.dispatchThreads(mtl.Size.of(D, kv_heads, rows), mtl.Size.of(256, 1, 1));
            e.end();
            try finish(cb);
        }
        for (len..len + rows) |pos| for (0..kv_heads) |h| for ([_]bool{ false, true }, [_]mtl.Buffer{ kc, vc }) |values, src| {
            const at = ids[pos / page] * page_bytes + part(values) + (h * page + pos % page) * D * 2;
            try std.testing.expectEqualSlices(u8, src.slice(u8, kv_heads * cap * D * 2)[(h * cap + pos) * D * 2 ..][0 .. D * 2], a8[at..][0 .. D * 2]);
        };
        // attention over every row so far, each window row seeing its own and those before it
        const nch = (len + rows + 127) / 128;
        const q = try device.buffer(query_heads * rows * D * 2, opts);
        defer q.deinit();
        bf16s(rnd, q.slice(u16, query_heads * rows * D));
        const dims = [8]i32{ @intCast(len), @intCast(rows), @intCast(cap), @intCast(nch), 0, @intCast(rows), 0, 0 };
        const slots = query_heads * rows * nch;
        var outs: [2][3]mtl.Buffer = undefined;
        for (&outs) |*o| o.* = .{ try device.buffer(slots * 4, opts), try device.buffer(slots * 4, opts), try device.buffer(slots * D * 4, opts) };
        defer for (outs) |o| for (o) |b| b.deinit();
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            for (outs, 0..) |o, paged| {
                e.setPipeline(if (paged == 1) attn_paged else attn);
                e.setBuffer(q, 0, 0);
                e.setBuffer(if (paged == 1) arena else kc, if (paged == 1) part(false) else 0, 1);
                e.setBuffer(if (paged == 1) arena else vc, if (paged == 1) part(true) else 0, 2);
                e.setValue(@as(f32, 1.0 / 16.0), 3);
                e.setBytes(std.mem.asBytes(&dims), 4);
                for (o, 5..) |b, slot| e.setBuffer(b, 0, slot);
                if (paged == 1) {
                    e.setBuffer(table, 0, 8);
                    e.setValue(page_bytes, 9);
                }
                e.dispatchThreads(mtl.Size.of(512, nch, kv_heads * rows), mtl.Size.of(512, 1, 1));
            }
            e.end();
            try finish(cb);
        }
        for (outs[0], outs[1]) |want, got| try std.testing.expectEqualSlices(u8, want.slice(u8, want.length()), got.slice(u8, got.length()));
    }
}
