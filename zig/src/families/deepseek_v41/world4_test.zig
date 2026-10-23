const std = @import("std");
const admission = @import("block_admission.zig");
const Config = @import("config.zig").Config;
const Options = @import("block.zig").Options;
const partition = @import("partition.zig");

test "world four: the default expert width is admitted" {
    const c: Config = .{};
    try admission.options(&c, Options{ .world = 4 }, &.{0}, 1);
    try std.testing.expectEqual(@as(u32, 2304), c.expert_width);
}

test "world four: block shares cover each width once in order" {
    for ([_]u32{ 128, 256, 512, 2304, 7168, 129280 }) |width| {
        for ([_]u32{ 1, 2, 4 }) |world| {
            if (width / 128 < world) continue;
            var cursor: u32 = 0;
            var previous: u32 = std.math.maxInt(u32);
            for (0..world) |rank| {
                const p = try partition.share(width, rank, world);
                try std.testing.expectEqual(cursor, p.lo);
                try std.testing.expect(p.len % 128 == 0 and p.len > 0);
                try std.testing.expect(p.len <= previous and previous - p.len <= 128 or rank == 0);
                previous = p.len;
                cursor += p.len;
            }
            try std.testing.expectEqual(width, cursor);
        }
    }
    try std.testing.expectError(error.SplitNotOnBlocks, partition.share(2304, 0, 0));
    try std.testing.expectError(error.SplitNotOnBlocks, partition.share(2305, 0, 4));
    try std.testing.expectError(error.SplitNotOnBlocks, partition.share(128, 0, 4));
    try std.testing.expectError(error.SplitNotOnBlocks, partition.share(2304, 4, 4));
}

test "world four: default expert and vocabulary shares keep whole blocks" {
    const cfg: Config = .{};
    for ([_]u32{ 640, 640, 512, 512 }, [_]u32{ 32384, 32384, 32256, 32256 }, 0..) |expert, vocab, rank| {
        try std.testing.expectEqual(expert, (try partition.share(cfg.expert_width, rank, 4)).len);
        try std.testing.expectEqual(vocab, (try partition.share(cfg.vocab, rank, 4)).len);
        try admission.options(&cfg, Options{ .world = 4, .rank = @intCast(rank) }, &.{0}, 1);
    }
}

test "world four: trellis and scale spans retain all blocks and legacy parts" {
    const Group = @import("core").exl3_layout.Group;
    for ([_]Group{ .{ .k_tiles = 448, .n_tiles = 144, .words = 32 }, .{ .k_tiles = 144, .n_tiles = 448, .words = 48 }, .{ .k_tiles = 448, .n_tiles = 8080, .words = 96 } }) |g| {
        for ([_]@import("core").exl3_layout.Split{ .row, .col }) |split| {
            for ([_]u32{ 1, 2 }) |world| for (0..world) |rank| {
                try std.testing.expectEqualDeep(try g.part(split, @intCast(rank), world), try partition.group(g, split, @intCast(rank), world));
            };
            var bytes: u64 = 0;
            var scale: u64 = 0;
            for (0..4) |rank| {
                const p = try partition.group(g, split, @intCast(rank), 4);
                try std.testing.expectEqual(bytes, p.trellis.offset);
                try std.testing.expectEqual(scale, if (split == .row) p.suh.offset else p.svh.offset);
                bytes += p.trellis.len;
                scale += if (split == .row) p.suh.len else p.svh.len;
            }
            try std.testing.expectEqual(if (split == .row) g.trellisBytes() else @as(u64, g.n_tiles) * g.tileBytes(), bytes);
            try std.testing.expectEqual(2 * @as(u64, if (split == .row) g.k() else g.n()), scale);
        }
    }
}

test "world four: decode and padded prefill widths are local" {
    const block = @import("block.zig");
    const cfg: Config = .{};
    for (0..4) |rank| for ([_]bool{ false, true }) |prefill| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var widths: block.Widths = .{};
        try @import("widths_listing.zig").parseDense(a, &widths, @embedFile("fixtures/dense-k2.txt"));
        try @import("widths_listing.zig").parseExperts(a, &widths, @embedFile("fixtures/expert-k2.txt"));
        for ([_]u32{ 0, 40 }) |L| try widths.gm.put(a, L, .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
        const o: Options = .{ .world = 4, .rank = @intCast(rank) };
        const calls = if (prefill) try @import("block_prefill.zig").emitPrefill(a, &cfg, &widths, o, &.{ 0, 40 }, 17, 0, true) else try block.emit(a, &cfg, &widths, o, &.{ 0, 40 }, 1, 0, true);
        const expert = try partition.share(cfg.expert_width, rank, 4);
        const vocab = try partition.share(cfg.vocab, rank, 4);
        var seen_expert = false;
        var seen_logits = false;
        for (calls) |call| for (call.args) |arg| {
            if (arg.arg != .t or arg.arg.t.role != .buf) continue;
            const t = arg.arg.t;
            if (std.mem.endsWith(u8, t.role.buf, "svh_g")) {
                try std.testing.expectEqual(@as(i64, expert.len), t.shape[1]);
                seen_expert = true;
            }
            if (std.mem.eql(u8, t.role.buf, "w.logits")) {
                try std.testing.expectEqual(@as(i64, vocab.len), t.shape[1]);
                seen_logits = true;
            }
        };
        try std.testing.expect(seen_logits and (prefill or seen_expert));
    };
    try admission.nativePrefill(&cfg, 1);
    try admission.nativePrefill(&cfg, 2);
    try admission.nativePrefill(&cfg, 4);
    try std.testing.expectError(error.UnsupportedPrefillHeads, admission.nativePrefill(&cfg, 8));
}
