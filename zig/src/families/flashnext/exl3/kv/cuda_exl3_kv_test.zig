//! Host tests of the KV placement: shared allocation geometry, paging refusals, policy bytes and kernel smem.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const kvpage = @import("cuda_exl3_kvpage.zig");
const hub = @import("cuda_exl3_kv.zig");
const int8k = @import("cuda_exl3_kv8.zig");

const kv_heads = wt.kv_heads;
const index_row: usize = wt.index_dim * 2;

const EngineKv = hub.EngineKv;
const entryOf = hub.entryOf;
const forget = hub.forget;
const kvRowBytes = hub.kvRowBytes;
const seqBytes = hub.seqBytes;
const sequenceLayout = hub.sequenceLayout;
const snapRows = hub.snapRows;
const Kv8 = int8k.Kv8;
const chunkSmem = int8k.chunkSmem;

test "exl3: bf16 paging and host indexer placement share allocation and budget geometry" {
    const cap: usize = 200_016;
    const cfg: kvpage.Config = .{};
    for ([_]bool{ false, true }) |host_index| {
        const placement: EngineKv = .{ .ikc_host = host_index };
        const resident = try sequenceLayout(placement, cap, 1024, null);
        const paged = try sequenceLayout(placement, cap, 4096, cfg);
        try std.testing.expectEqual(cap, resident.cache_rows);
        try std.testing.expectEqual(@as(usize, 1), paged.cache_rows);
        try std.testing.expectEqual(@as(usize, 4096), paged.paging.?.rows);
        try std.testing.expectEqual(snapRows(false, kv_heads), paged.row);
        const placeholders = wt.attn_n * 2 * paged.row.kv;
        try std.testing.expectEqual(try kvpage.deviceBytesFor(cap, cfg.forChunk(4096)) + placeholders, paged.bytes.kv);
        try std.testing.expectEqual(resident.bytes.host + resident.bytes.kv, paged.bytes.host);
        try std.testing.expectEqual(resident.bytes.device() - resident.bytes.kv + paged.bytes.kv, paged.bytes.device());
        try std.testing.expectEqual(resident.bytes.ikc, paged.bytes.ikc);
        try std.testing.expectEqual(@as(u64, if (host_index) 0 else wt.attn_n * cap * index_row), paged.bytes.ikc);
        const small = try sequenceLayout(placement, cap, 1024, cfg);
        try std.testing.expectEqual(@as(u64, 2 * wt.attn_n * (4096 - cfg.rows) * kvpage.row_bytes), paged.bytes.device() - small.bytes.device());
    }
}

test "exl3: paging refuses incompatible cache layouts before allocating" {
    const cfg: kvpage.Config = .{};
    var k8: Kv8 = undefined;
    try std.testing.expectError(error.KvPagePlacement, sequenceLayout(.{ .k8 = &k8 }, 4096, 1024, cfg));
    try std.testing.expectError(error.KvPagePlacement, sequenceLayout(.{ .heads = 1 }, 4096, 1024, cfg));
    try std.testing.expectError(error.KvPageConfig, sequenceLayout(.{}, 4096, 1024, .{ .slots = 0 }));
    try std.testing.expectError(error.KvPageConfig, sequenceLayout(.{}, 4096, 1024, .{ .window = wt.dense_keys - 1 }));
}

test "exl3: newSeq with host indexer keys enters the paging allocator" {
    if (cuda.kernels.available) return error.SkipZigTest;
    const old_force = kvpage.force;
    const old_off = kvpage.force_off;
    defer {
        kvpage.force = old_force;
        kvpage.force_off = old_off;
    }
    kvpage.force = .{};
    kvpage.force_off = false;
    var driver: cuda.Driver = undefined;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: @import("../../cuda_prompt.zig").Engine = undefined;
    engine.gpa = std.testing.allocator;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 4096;
    engine.round_graph = .{};
    engine.page_config = kvpage.force;
    (try entryOf(&engine)).ikc_host = true;
    defer forget(engine.gpa, &engine);
    try std.testing.expectError(error.BuiltWithoutKernels, engine.newSeq(8192));
}

test "exl3: kv the default policy is upstream's seqBytes, split" {
    const prompt = @import("../../cuda_prompt.zig");
    for ([_]usize{ 1, 5, 4096, 200_016, 262_144 }) |cap| {
        const s = seqBytes(cap, .{});
        try std.testing.expectEqual(@as(u64, prompt.seqBytes(cap)), s.device());
        try std.testing.expectEqual(@as(u64, 0), s.host);
    }
}

test "exl3: kv int8 is 34 bytes a 32-group, int4 18; a head split halves K/V, not the indexer" {
    try std.testing.expectEqual(@as(usize, 2048), kvRowBytes(.bf16, 2));
    try std.testing.expectEqual(@as(usize, 1088), kvRowBytes(.int8, 2));
    try std.testing.expectEqual(@as(usize, 576), kvRowBytes(.int4, 2));
    try std.testing.expectEqual(@as(usize, 544), kvRowBytes(.int8, 1));
    const cap = 200_000;
    const a = seqBytes(cap, .{ .kv = .int8 });
    const b = seqBytes(cap, .{ .kv = .int8, .heads = 1 });
    try std.testing.expectEqual(a.kv, 2 * b.kv);
    try std.testing.expectEqual(a.ikc, b.ikc);
    const h = seqBytes(cap, .{ .kv = .int8, .kv_host = true, .ikc_host = true });
    try std.testing.expectEqual(a.kv + a.ikc, h.host);
    try std.testing.expectEqual(a.device() - a.kv - a.ikc, h.device());
}

test "exl3: kv chunk smem fits the 99 KiB opt-in on sm_120" {
    try std.testing.expect(chunkSmem(2) <= 99 * 1024);
    try std.testing.expectEqual(@as(u32, 60928), chunkSmem(1));
    try std.testing.expectEqual(@as(u32, 78336), chunkSmem(2));
}
