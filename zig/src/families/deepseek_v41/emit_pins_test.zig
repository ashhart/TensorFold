//! The calls and prefetch sources of worlds one, two and four, pinned on main so a head split can't move them.

const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const ds = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;
const widths = @import("prefill_heads_test.zig").widths;
const Sha256 = std.crypto.hash.sha2.Sha256;

const even_worlds = [_]u32{ 1, 2, 4 };

fn add(h: *Sha256, a: std.mem.Allocator, value: anytype) !void {
    var text: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(value, .{}, &text.writer);
    h.update(text.written());
}

fn expectDigest(want: []const u8, h: *Sha256) !void {
    var digest: [32]u8 = undefined;
    h.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    try t.expectEqualStrings(want, &hex);
}

test "worlds one, two and four: decode calls keep their main fingerprint" {
    const cfg: Config = .{};
    var h = Sha256.init(.{});
    for (even_worlds) |world| for (0..world) |rank| for ([_]i64{ 1, 17, 64 }) |rows| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = world, .rank = @intCast(rank) };
        try add(&h, a, try block.emit(a, &cfg, &w, o, &.{ 0, 2, 20, 24 }, rows, 2048, true));
        try add(&h, a, try block.emit(a, &cfg, &w, o, &.{40}, rows, 2048, false));
    };
    try expectDigest("742375bc2d734325b352265dc161b2afd470dcde20da39a7ad3798a00354c9d9", &h);
}

test "worlds one, two and four: DSpark calls keep their main fingerprint" {
    const cfg: Config = .{};
    var h = Sha256.init(.{});
    for (even_worlds) |world| for (0..world) |rank| for ([_]i64{ 1, 17, 64 }) |rows| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var w = try widths(a);
        try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
        const o: block.Options = .{ .world = world, .rank = @intCast(rank) };
        try add(&h, a, try ds.emitIngest(a, &cfg, &w, o, rows, 0));
        try add(&h, a, try ds.emitPass(a, &cfg, &w, o, cfg.dspark_block));
    };
    try expectDigest("d82f43080a0b8260f2e20fefee04e25d90e36c7cc8f7b0e3141faac4c08f4da5", &h);
}

test "worlds one, two and four: prefill calls keep their main fingerprint" {
    const cfg: Config = .{};
    var h = Sha256.init(.{});
    for (even_worlds) |world| for (0..world) |rank| for ([_]i64{ 16, 17, 129 }) |rows| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = world, .rank = @intCast(rank) };
        try add(&h, a, try bp.emitPrefill(a, &cfg, &w, o, &.{ 0, 2, 20, 24, 40 }, rows, 2048, true));
    };
    try expectDigest("589725eceeb07b8ea0b6c7be9488ce59a7d4e72dac64442987131ad326746e09", &h);
}

test "worlds one, two and four: prefetch sources keep their main fingerprint" {
    const cfg: Config = .{};
    var h = Sha256.init(.{});
    for (even_worlds) |world| for (0..world) |rank| for ([_]u32{ 0, 2 }) |layer| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        var e: block.Emitter = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .world = world, .rank = @intCast(rank) }, .n = 1, .start = 0 };
        try add(&h, a, try e.l2pfSources(layer, .o));
        try add(&h, a, try e.l2pfSources(layer, .x));
    };
    try expectDigest("7c3f40b37335ccc7a773ee24f6b608f9648d9ce9306a5abfc1a0421e77de054b", &h);
}
