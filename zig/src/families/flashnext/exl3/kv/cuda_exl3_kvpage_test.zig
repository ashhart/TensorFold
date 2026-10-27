//! Host tests of the KV paging: tables, bounds, schedules and the attention dispatch.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");
const hub = @import("cuda_exl3_kvpage.zig");
const configs = @import("cuda_exl3_kvpage_config.zig");
const snaps = @import("cuda_exl3_kvpage_snapshot.zig");
const paged = @import("cuda_exl3_kvpage_paged.zig");

const head_dim = wt.head_dim;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const att_rows = parts.att_rows;
const attn_n = wt.attn_n;
const work_n: usize = att_rows * top;

const select = hub.select;
const Config = configs.Config;
const cdiv = configs.cdiv;
const deviceBytesFor = configs.deviceBytesFor;
const pagingRows = configs.pagingRows;
const row_bytes = configs.row_bytes;
const windowPlan = configs.windowPlan;
const SnapshotLayout = snaps.SnapshotLayout;
const Paged = paged.Paged;

test "exl3: kvpage tables and device bytes" {
    const cfg: Config = .{};
    try std.testing.expect(cfg.window >= wt.dense_keys);
    // 200k tokens: the device side of the default shape against upstream's K/V for the same sequence
    const wr = cfg.window + cfg.rows + ratio;
    const dev = 2 * (attn_n * wr + ratio * cfg.slots) * row_bytes;
    const full = attn_n * 2 * 200_016 * row_bytes;
    try std.testing.expect(dev * 8 < full);
    try std.testing.expectEqual(@as(usize, 1024), row_bytes);
    // 200k: the default shape's device side is about half a GiB (upstream's K/V: 4.58 GiB)
    try std.testing.expect(try deviceBytesFor(200_016, cfg) < 5 << 27);
}

test "exl3: kvpage allocation and admission reject invalid checked sizes before CUDA access" {
    const cfg: Config = .{};
    const layout = try SnapshotLayout.init(200_016, cfg, 0);
    try std.testing.expectEqual(layout.device + layout.tables + work_n * 4 * 4, try deviceBytesFor(200_016, cfg));
    try std.testing.expectError(error.Overflow, deviceBytesFor(std.math.maxInt(usize), cfg));
    try std.testing.expectError(error.KvPageRows, deviceBytesFor(0, cfg));
    var driver: cuda.Driver = undefined;
    try std.testing.expectError(error.Overflow, Paged.init(std.testing.allocator, &driver, std.math.maxInt(usize), cfg));
    var oversized = cfg;
    oversized.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, Paged.init(std.testing.allocator, &driver, 32, oversized));
}

test "exl3: kvpage window boundaries and conservative capacity" {
    const cfg: Config = .{ .window = 2052, .rows = 256, .slots = 512 };
    const wr = cfg.window + cfg.rows + ratio;
    try std.testing.expectEqual(@as(usize, 0), try windowPlan(cfg, wr, 0, wr - 1, 1));
    try std.testing.expectEqual(@as(usize, 260), try windowPlan(cfg, wr, 0, wr, 1));
    try std.testing.expectEqual(@as(usize, 0), try windowPlan(cfg, wr, 260, 0, 256));
    try std.testing.expectError(error.KvPageRewind, windowPlan(cfg, wr, 260, 259, 1));
    try std.testing.expectError(error.KvPageRows, windowPlan(cfg, wr, 0, wr + 1, 1));
    try std.testing.expectEqual(@as(usize, 8192), try windowPlan(cfg, wr, 8192, 8192, 256));
    try std.testing.expectEqual(@as(usize, 8192), try windowPlan(cfg, wr, 8192, 8192, 1));
}

test "exl3: kvpage capacity subdivision boundaries and checked admission" {
    var cfg: Config = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    try std.testing.expectEqual(@as(usize, 256), try pagingRows(cfg, 65536, 256));
    try std.testing.expectEqual(@as(usize, 32), try pagingRows(cfg, 65540, 256));
    try std.testing.expectEqual(@as(usize, 31), try pagingRows(cfg, 65540, 31));
    try std.testing.expectEqual(@as(usize, 32), try pagingRows(cfg, 65540, 33));
    cfg.slots = 16383;
    try std.testing.expectEqual(@as(usize, 31), try pagingRows(cfg, 65536, 256));
    cfg.slots = top - 1;
    try std.testing.expectEqual(@as(usize, 256), try pagingRows(cfg, (top - 1) * ratio, 256));
    try std.testing.expectError(error.KvPageCapacity, pagingRows(cfg, top * ratio, 1));
    const wr = cfg.window + cfg.rows + ratio;
    try std.testing.expectError(error.KvPageCapacity, windowPlan(cfg, wr, top * ratio, top * ratio, 256));
    try std.testing.expectError(error.KvPageRows, pagingRows(cfg, 0, 0));
    try std.testing.expectError(error.KvPageRows, pagingRows(cfg, 0, att_rows + 1));
    try std.testing.expectError(error.KvPageWindow, pagingRows(cfg, 1, 1));
    try std.testing.expectError(error.Overflow, windowPlan(cfg, std.math.maxInt(usize), 0, std.math.maxInt(usize), 1));
    cfg.slots = 0;
    try std.testing.expectError(error.KvPageConfig, windowPlan(cfg, wr, 0, 0, 1));
}

test "exl3: kvpage adversarial disjoint old blocks fit every subdivision" {
    for ([_]usize{ 512, 513, 1023, 1024, 16383, 16384 }) |slots| {
        const cfg: Config = .{ .slots = slots };
        const origin = (slots + 1) * ratio;
        for ([_]usize{ 1, 2, 15, 16, 17, 30, 31, 32, 33, 127, 255, 256 }) |rows| {
            const width = try pagingRows(cfg, origin, rows);
            var first: usize = 0;
            var calls: usize = 0;
            while (first < rows) {
                const count = @min(width, rows - first);
                var selected = std.AutoHashMap(usize, void).init(std.testing.allocator);
                defer selected.deinit();
                for (first..first + count) |row| for (0..top) |pick| {
                    const block = (row * top + pick) % (origin / ratio);
                    try selected.put(block, {});
                };
                try std.testing.expect(selected.count() <= slots);
                try std.testing.expectEqual(count * top, selected.count());
                first += count;
                calls += 1;
            }
            try std.testing.expectEqual(rows, first);
            try std.testing.expectEqual(cdiv(rows, width), calls);
            if (width < rows) try std.testing.expect(origin > wt.dense_keys);
        }
    }
    const cfg: Config = .{ .slots = 511 };
    try std.testing.expectEqual(att_rows, try pagingRows(cfg, 511 * ratio, att_rows));
}

test "exl3: kvpage 18k and 200k schedules use identical admission and paging bounds" {
    const cfg: Config = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    const wr = cfg.window + cfg.rows + ratio;
    for ([_]usize{ 18432, 18448, 200000, 200016 }) |cap| {
        var origin: usize = 0;
        var position: usize = 0;
        var split_calls: usize = 0;
        while (position < cap) {
            const rows = @min(cfg.rows, cap - position);
            origin = try windowPlan(cfg, wr, origin, position, rows);
            var first: usize = 0;
            while (first < rows) : (first += att_rows) {
                const count = @min(att_rows, rows - first);
                const width = try pagingRows(cfg, origin, count);
                try std.testing.expect(@min(width * top, origin / ratio) <= cfg.slots);
                if (origin / ratio <= cfg.slots) try std.testing.expectEqual(count, width);
                if (width < count) split_calls += 1;
            }
            position += rows;
        }
        try std.testing.expectEqual(cap > 65536 + cfg.window, split_calls > 0);
        try std.testing.expectEqual(@as(usize, 1), try pagingRows(cfg, origin, 1));
        try std.testing.expectEqual(@as(usize, 16), try pagingRows(cfg, origin, 16));
    }
}

test "exl3: kvpage real attention dispatch preserves selection and slices sparse rows" {
    const Fake = struct {
        const ids_base: u64 = 0x100000;
        const nk_base: u64 = 0x200000;
        const spr_base: u64 = 0x300000;
        const query_base: u64 = 0x400000;
        const output_base: u64 = 0x500000;
        const pos_base: u64 = 0x600000;
        var selected: usize = 0;
        var consumed: usize = 0;
        var planned: usize = 0;
        var width: usize = 0;
        var launches: usize = 0;

        fn value(parameters: ?[*]?*anyopaque, index: usize, comptime Value: type) Value {
            const pointer: *const Value = @ptrCast(@alignCast(parameters.?[index].?));
            return pointer.*;
        }

        fn launch(kernel: cuda.abi.Function, grid_x: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            launches += 1;
            switch (@intFromPtr(kernel.?)) {
                1 => {
                    std.debug.assert(value(parameters, 1, u64) == pos_base + selected * 16);
                    std.debug.assert(value(parameters, 2, u64) == ids_base);
                    std.debug.assert(grid_x == @min(att_rows, 531 - consumed));
                    selected += 1;
                },
                2 => {
                    const offset = consumed % att_rows;
                    std.debug.assert(value(parameters, 0, u64) == ids_base + offset * idw * 4);
                    std.debug.assert(value(parameters, 1, u64) == nk_base + offset * 4);
                    std.debug.assert(value(parameters, 2, u64) == spr_base + offset * 4);
                    planned = @intCast(value(parameters, 3, c_int));
                    std.debug.assert(planned == @min(width, @min(att_rows - offset, 531 - consumed)));
                },
                3 => {},
                4 => std.debug.assert(grid_x == planned),
                else => unreachable,
            }
            return cuda.abi.success;
        }

        pub fn attnPrep(_: @This(), _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: u64, _: f32, _: usize, _: usize, _: @import("../../cuda_glue.zig").Pg) !void {}
        pub fn qsaPool(_: @This(), _: u64, _: u64, _: u64, _: u64, _: u64, _: f32, _: usize, _: @import("../../cuda_glue.zig").Pg) !void {}
        pub fn qsaScores(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: @import("../../cuda_glue.zig").Pg) !void {}
        pub fn attnChunks(_: @This(), query: u64, _: u64, _: u64, pos: u64, _: u64, _: u64, _: u64, ids: u64, nk: u64, spr: u64, rows: usize, _: usize, chunks: usize, qsa: bool, _: @import("../../cuda_glue.zig").Pg) !void {
            try std.testing.expectEqual(query_base + consumed * wt.q_heads * head_dim * 2, query);
            try std.testing.expectEqual(pos_base + (selected - 1) * 16, pos);
            try std.testing.expectEqual(ids_base + (consumed % att_rows) * idw * 4, ids);
            try std.testing.expectEqual(nk_base + (consumed % att_rows) * 4, nk);
            try std.testing.expectEqual(spr_base + (consumed % att_rows) * 4, spr);
            try std.testing.expectEqual(planned, rows);
            try std.testing.expectEqual(@as(usize, 5), chunks);
            try std.testing.expect(qsa);
        }
        pub fn attnMerge(_: @This(), _: u64, _: u64, _: u64, pos: u64, output: u64, _: u64, _: u64, rows: usize, _: usize, qsa: bool) !void {
            try std.testing.expectEqual(pos_base + (selected - 1) * 16, pos);
            try std.testing.expectEqual(output_base + consumed * wt.out_k * 2, output);
            try std.testing.expect(qsa);
            consumed += rows;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const function: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var pg: Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 4096, .rows = 2048, .slots = 16384 };
    pg.cap = 200016;
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.nblocks = cdiv(pg.cap, ratio) + 1;
    pg.dev.ptr = 0x700000;
    pg.tabs.ptr = 0x800000;
    pg.work.ptr = 0x900000;
    pg.host_dev = 0xa00000;
    pg.k.plan = .{ .d = &driver, .handle = @ptrFromInt(2) };
    pg.k.fetch = .{ .d = &driver, .handle = @ptrFromInt(3) };
    pg.k.remap = .{ .d = &driver, .handle = @ptrFromInt(4) };
    const Buffer = struct { ptr: u64 = 0 };
    const Seq = struct { cap: usize, ikc: [attn_n]Buffer = @splat(.{}), pooled: [attn_n]Buffer = @splat(.{}), scores: u64 = 0 };
    const State = struct { inv: u64 = 0, dummy: u64 = 0, qids: u64 = Fake.ids_base, qnk: u64 = Fake.nk_base, qspr: u64 = Fake.spr_base, po: u64 = 0, pm: u64 = 0, pl: u64 = 0 };
    const Selection = struct { select: cuda.Function };
    const Pass = struct { seq: *Seq, s: *State, stream: cuda.Stream, tri: Fake = .{}, k: *const Selection, pos0: usize, lanes: []const u8 = &.{} };
    var seq: Seq = .{ .cap = pg.cap };
    var state: State = .{};
    const kernels: Selection = .{ .select = function };
    const layer = .{ .q_scale = Buffer{}, .k_scale = Buffer{}, .i_scale = Buffer{}, .ik_scale = Buffer{} };
    const at = .{ .pos = Fake.pos_base, .q = Fake.query_base, .iq = @as(u64, 0), .ao = Fake.output_base };
    for ([_]usize{ 14336, 65536, 65540, 194560 }) |origin| {
        pg.win0 = @splat(origin);
        Fake.selected = 0;
        Fake.consumed = 0;
        Fake.planned = 0;
        Fake.launches = 0;
        Fake.width = try pagingRows(pg.cfg, origin, att_rows);
        var pass: Pass = .{ .seq = &seq, .s = &state, .stream = stream, .k = &kernels, .pos0 = origin + pg.cfg.window };
        try pg.attnRows(&pass, layer, 0, 0, 531, at);
        try std.testing.expectEqual(@as(usize, 3), Fake.selected);
        try std.testing.expectEqual(@as(usize, 531), Fake.consumed);
        const launches = Fake.launches;
        pg.cfg.slots = top - 1;
        try std.testing.expectError(error.KvPageCapacity, pg.attnRows(&pass, layer, 0, 0, 531, at));
        try std.testing.expectEqual(launches, Fake.launches);
        pg.cfg.slots = 16384;
    }
}

test "exl3: kvpage host preparation writes back and compacts once before replay" {
    const Fake = struct {
        var copies: usize = 0;
        fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(dst + bytes <= src or src + bytes <= dst);
            const target: [*]u8 = @ptrFromInt(dst);
            const source: [*]const u8 = @ptrFromInt(src);
            @memcpy(target[0..bytes], source[0..bytes]);
            copies += 1;
            return cuda.abi.success;
        }
    };
    Fake.copies = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var pg: Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 2052, .rows = 256, .slots = 512 };
    pg.cap = 8192;
    pg.wr = pg.cfg.window + pg.cfg.rows + ratio;
    pg.win0 = @splat(0);
    pg.rebases = @splat(0);
    pg.written = @splat(0);
    const device = try std.testing.allocator.alloc(u8, pg.devBytes());
    defer std.testing.allocator.free(device);
    pg.dev.ptr = @intFromPtr(device.ptr);
    const host = try std.testing.allocator.alloc(u8, 2 * pg.cap * row_bytes);
    defer std.testing.allocator.free(host);
    @memset(host, 0);
    pg.host_dev = @intFromPtr(host.ptr);
    for (0..2) |side| {
        const start = if (side == 0) pg.devK(0) else pg.devV(0);
        const words: [*]u32 = @ptrFromInt(start);
        for (0..pg.wr) |position| @memset(words[position * (row_bytes / 4) ..][0 .. row_bytes / 4], @as(u32, @intCast(position + side * pg.cap)));
    }
    const before = pg.win0[0];
    const pos0 = pg.wr;
    try pg.prepareLayer(stream, 0, pos0, 1);
    const moved = pg.win0[0];
    try std.testing.expect(moved > before);
    try std.testing.expectEqual(@as(u64, 1), pg.rebases[0]);
    try std.testing.expectEqual(@as(u64, moved), pg.written[0]);
    for (0..2) |side| {
        const written: [*]const u32 = @ptrFromInt(pg.host_dev + side * pg.cap * row_bytes);
        const kept: [*]const u32 = @ptrFromInt(if (side == 0) pg.devK(0) else pg.devV(0));
        for (0..moved) |position| for (written[position * (row_bytes / 4) ..][0 .. row_bytes / 4]) |word| try std.testing.expectEqual(@as(u32, @intCast(position + side * pg.cap)), word);
        for (moved..pos0) |position| for (kept[(position - moved) * (row_bytes / 4) ..][0 .. row_bytes / 4]) |word| try std.testing.expectEqual(@as(u32, @intCast(position + side * pg.cap)), word);
    }
    const copies = Fake.copies;
    try pg.prepareLayer(stream, 0, pos0, 1);
    try std.testing.expectEqual(copies, Fake.copies);
    try std.testing.expectEqual(@as(u64, 1), pg.rebases[0]);
}
