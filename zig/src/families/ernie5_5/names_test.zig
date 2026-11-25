//! The weight map against the release's safetensors headers: the folded table, edited copies and the real shards.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const names = @import("names.zig");

const t = std.testing;
const folded = @embedFile("fixtures/release_header.txt");

/// The tag safetensors writes for the dtypes the release holds.
fn dtypeTag(d: st.DType) []const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f32 => "F32",
        .f8_e4m3 => "F8_E4M3",
        .f8_e8m0 => "F8_E8M0",
        .i8 => "I8",
        else => unreachable,
    };
}

fn byName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Each line over its layer ranges and, for `{E}` lines, its expert count; the head holds the count and the sha256.
pub fn unfold(arena: std.mem.Allocator, text: []const u8) !st.Header {
    var out: st.Header = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next().?, ' ');
    if (!std.mem.eql(u8, head.next().?, "ernie55-release")) return error.BadFixture;
    const count = try std.fmt.parseInt(usize, head.next().?, 10);
    const digest = head.next().?;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pat = f.next().?;
        var e: st.Entry = .{ .dtype = st.DType.parse(f.next().?).?, .rank = 0, .shape = @splat(1), .begin = 0, .end = 0 };
        var dims = std.mem.tokenizeScalar(u8, f.next().?, 'x');
        while (dims.next()) |d| : (e.rank += 1) e.shape[e.rank] = try std.fmt.parseInt(usize, d, 10);
        const layers = f.next() orelse {
            try out.put(arena, pat, e);
            continue;
        };
        const experts = if (f.next()) |x| try std.fmt.parseInt(usize, x, 10) else 1;
        var spans = std.mem.tokenizeScalar(u8, layers, ',');
        while (spans.next()) |span| {
            const dash = std.mem.indexOfScalar(u8, span, '-');
            const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
            for (lo..hi + 1) |l| for (0..experts) |x| {
                var lb: [8]u8 = undefined;
                var eb: [8]u8 = undefined;
                const by_layer = try std.mem.replaceOwned(u8, arena, pat, "{L}", try std.fmt.bufPrint(&lb, "{d}", .{l}));
                try out.put(arena, try std.mem.replaceOwned(u8, arena, by_layer, "{E}", try std.fmt.bufPrint(&eb, "{d}", .{x})), e);
            };
        }
    }
    if (out.count() != count) return error.BadFixture;
    const sorted = try arena.dupe([]const u8, out.keys());
    std.mem.sort([]const u8, sorted, {}, byName);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (sorted, 0..) |n, i| {
        const v = out.get(n).?;
        if (i > 0) h.update("\n");
        h.update(n);
        h.update(" ");
        h.update(dtypeTag(v.dtype));
        h.update(" ");
        for (v.shape[0..v.rank], 0..) |d, k| {
            var b: [24]u8 = undefined;
            if (k > 0) h.update("x");
            h.update(try std.fmt.bufPrint(&b, "{d}", .{d}));
        }
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(d, .lower), digest)) return error.BadFixture;
    return out;
}

fn release() !config.Config {
    var p: config.Problem = .{};
    return config.parse(t.allocator, config_test.release_json, &p);
}

test "the map matches the release's 140,457 tensors by name, dtype and shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try release();
    const r = try names.check(a, &c, try unfold(a, folded));
    try t.expect(r.clean());
    try t.expectEqual(@as(usize, 140457), r.expected);
}

test "a missing, retyped, reshaped or extra tensor is reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try release();
    var h = try unfold(a, folded);
    _ = h.orderedRemove("model.layers.7.self_attn.core_attention.indexer.wk.weight");
    h.getPtr("model.layers.3.block_sparse_moe.experts.511.w2.weight_scale_inv").?.dtype = .u8;
    h.getPtr("model.mtp.2.self_attn.linear_o_group_proj").?.shape[1] = 4095;
    try h.put(a, "model.layers.1.self_attn.core_attention.indexer.wk.weight", h.get("model.layers.14.self_attn.core_attention.indexer.wk.weight").?);
    const r = try names.check(a, &c, h);
    try t.expectEqualStrings("model.layers.7.self_attn.core_attention.indexer.wk.weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqual(@as(usize, 2), r.wrong.items.len);
    try t.expectEqualStrings("model.layers.3.block_sparse_moe.experts.511.w2.weight_scale_inv", r.wrong.items[0]);
    try t.expectEqualStrings("model.mtp.2.self_attn.linear_o_group_proj", r.wrong.items[1]);
    try t.expectEqual(@as(usize, 1), r.unknown.items.len);
    try t.expectEqualStrings("model.layers.1.self_attn.core_attention.indexer.wk.weight", r.unknown.items[0]);
}

test "a config that moves a DSA layer is caught by the release's headers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ratios: std.ArrayList(u8) = .empty;
    try ratios.appendSlice(a, "[-2");
    for (1..43) |i| try ratios.appendSlice(a, if (i % 7 == 0 and i != 7) ", -2" else ", 128");
    try ratios.appendSlice(a, "]");
    var p: config.Problem = .{};
    const c = try config.parse(t.allocator, try config_test.edited(a, &.{.{ "compress_ratios", ratios.items }}), &p);
    const r = try names.check(a, &c, try unfold(a, folded));
    try t.expect(!r.clean());
    try t.expect(r.missing.items.len > 0 and r.unknown.items.len > 0);
}

/// Every model shard's header from the directory TF_ERNIE55_HEADERS names (the shards, or copies of their headers).
fn realHeaders(a: std.mem.Allocator) !st.Header {
    const dir = std.mem.span(std.c.getenv("TF_ERNIE55_HEADERS") orelse return error.SkipZigTest);
    var all: st.Header = .empty;
    for (1..129) |i| {
        var buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "/model-{d:0>5}-of-00128.safetensors", .{i});
        var file = try std.Io.Dir.cwd().openFile(t.io, try std.mem.concat(a, u8, &.{ dir, path }), .{});
        defer file.close(t.io);
        var reader = file.reader(t.io, &.{});
        var len: [8]u8 = undefined;
        try reader.interface.readSliceAll(&len);
        const json = try a.alloc(u8, std.mem.readInt(u64, &len, .little));
        try reader.interface.readSliceAll(json);
        try names.merge(a, &all, try st.parseHeader(a, json, std.math.maxInt(usize)));
    }
    return all;
}

test "the map matches the release's own headers (TF_ERNIE55_HEADERS)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try release();
    const real = try realHeaders(a);
    const r = try names.check(a, &c, real);
    try t.expect(r.clean());
    const table = try unfold(a, folded);
    try t.expectEqual(table.count(), real.count());
    for (table.keys(), table.values()) |n, want| {
        const got = real.get(n).?;
        try t.expectEqual(want.dtype, got.dtype);
        try t.expectEqualSlices(usize, want.shape[0..want.rank], got.shape[0..got.rank]);
    }
}
