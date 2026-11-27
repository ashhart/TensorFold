//! The weight map against the releases' safetensors headers: the folded tables, edited copies and the real shards.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const names = @import("names.zig");

const t = std.testing;
pub const header_37b = @embedFile("fixtures/header_37b_fp8.txt");
pub const header_708b = @embedFile("fixtures/header_708b_fp8.txt");
pub const header_37b_bf16 = @embedFile("fixtures/header_37b_bf16.txt");

/// The tag safetensors writes for the dtypes the releases hold.
fn dtypeTag(d: st.DType) ![]const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f32 => "F32",
        .f8_e4m3 => "F8_E4M3",
        else => error.BadFixture,
    };
}

fn byName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Inclusive layer spans "0-2,4" of a folded line, each layer's names put with `{E}` over `experts` numbers.
fn expand(arena: std.mem.Allocator, out: *st.Header, pattern: []const u8, layers: []const u8, experts: usize, e: st.Entry) !void {
    var spans = std.mem.tokenizeScalar(u8, layers, ',');
    while (spans.next()) |span| {
        const dash = std.mem.indexOfScalar(u8, span, '-');
        const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
        const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
        for (lo..hi + 1) |l| for (0..experts) |x| {
            var lb: [12]u8 = undefined;
            var eb: [12]u8 = undefined;
            const by_layer = try std.mem.replaceOwned(u8, arena, pattern, "{L}", try std.fmt.bufPrint(&lb, "{d}", .{l}));
            const full = try std.mem.replaceOwned(u8, arena, by_layer, "{E}", try std.fmt.bufPrint(&eb, "{d}", .{x}));
            const slot = try out.getOrPut(arena, full);
            if (slot.found_existing) return error.BadFixture;
            slot.value_ptr.* = e;
        };
    }
}

/// A folded table back to its header entries; the head line's count and sha256 must match what it unfolds to.
pub fn unfold(arena: std.mem.Allocator, text: []const u8) !st.Header {
    var out: st.Header = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next() orelse return error.BadFixture, ' ');
    if (!std.mem.startsWith(u8, head.next() orelse "", "gigachat4-")) return error.BadFixture;
    const count = try std.fmt.parseInt(usize, head.next() orelse "", 10);
    const digest = head.next() orelse return error.BadFixture;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pattern = f.next().?;
        var e: st.Entry = .{ .dtype = st.DType.parse(f.next() orelse "") orelse return error.BadFixture, .rank = 0, .shape = @splat(1), .begin = 0, .end = 0 };
        var dims = std.mem.tokenizeScalar(u8, f.next() orelse return error.BadFixture, 'x');
        while (dims.next()) |d| : (e.rank += 1) e.shape[e.rank] = try std.fmt.parseInt(usize, d, 10);
        const layers = f.next() orelse {
            try out.put(arena, pattern, e);
            continue;
        };
        const experts = if (f.next()) |x| try std.fmt.parseInt(usize, x, 10) else 1;
        try expand(arena, &out, pattern, layers, experts, e);
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
        h.update(try dtypeTag(v.dtype));
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

pub fn parsed(json: []const u8) !config.Config {
    var p: config.Problem = .{};
    return config.parse(t.allocator, json, &p);
}

test "the map matches each release's tensors by name, dtype and shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, []const u8, usize }{
        .{ try config_test.json37b(a), header_37b, 36700 },
        .{ config_test.json_708b, header_708b, 95616 },
        .{ try config_test.json37bBf16(a), header_37b_bf16, 18742 },
    };
    for (cases) |c| {
        const cfg = try parsed(c[0]);
        const r = try names.check(a, &cfg, try unfold(a, c[1]));
        try t.expect(r.clean());
        try t.expectEqual(c[2], r.expected);
    }
}

test "a missing, retyped, reshaped or unknown tensor is reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parsed(try config_test.json37b(a));
    var h = try unfold(a, header_37b);
    _ = h.orderedRemove("model.layers.7.self_attn.indexer.wk.weight");
    h.getPtr("model.layers.5.mlp.experts.191.down_proj.weight_scale_inv").?.dtype = .bf16;
    h.getPtr("model.layers.11.self_attn.kv_a_proj_with_mqa.weight_scale_inv").?.shape[0] = 4;
    h.getPtr("model.layers.2.mlp.gate.weight").?.dtype = .f8_e4m3;
    try h.put(a, "model.layers.0.self_attn.indexer.wk.weight", h.get("model.layers.3.self_attn.indexer.wk.weight").?);
    const r = try names.check(a, &c, h);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqualStrings("model.layers.7.self_attn.indexer.wk.weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 3), r.wrong.items.len);
    try t.expectEqualStrings("model.layers.2.mlp.gate.weight", r.wrong.items[0]);
    try t.expectEqualStrings("model.layers.5.mlp.experts.191.down_proj.weight_scale_inv", r.wrong.items[1]);
    try t.expectEqualStrings("model.layers.11.self_attn.kv_a_proj_with_mqa.weight_scale_inv", r.wrong.items[2]);
    try t.expectEqual(@as(usize, 1), r.unknown.items.len);
    try t.expectEqualStrings("model.layers.0.self_attn.indexer.wk.weight", r.unknown.items[0]);
}

test "a stray scale beside a BF16 weight, or BF16 headers under the FP8 config, are refused" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bf16 = try parsed(try config_test.json37bBf16(a));
    var h = try unfold(a, header_37b_bf16);
    const scale = h.get("model.layers.3.self_attn.indexer.k_norm.weight").?;
    try h.put(a, "model.layers.3.self_attn.o_proj.weight_scale_inv", scale);
    const r = try names.check(a, &bf16, h);
    try t.expectEqual(@as(usize, 0), r.missing.items.len + r.wrong.items.len);
    try t.expectEqualStrings("model.layers.3.self_attn.o_proj.weight_scale_inv", r.unknown.items[0]);
    const fp8 = try parsed(try config_test.json37b(a));
    const crossed = try names.check(a, &fp8, try unfold(a, header_37b_bf16));
    try t.expect(!crossed.clean());
    try t.expectEqual(@as(usize, 36700 - 18742), crossed.missing.items.len);
    try t.expectEqual(@as(usize, 36700 - 18742), crossed.wrong.items.len);
}

test "a config that moves an attention layer is caught by the release's headers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const moved = try config_test.edited(a, try config_test.json37b(a), &.{.{ "full_attention_layers", "[2, 7, 11, 15, 19, 23, 27, 31]" }});
    const moved_cfg = try parsed(moved);
    const r = try names.check(a, &moved_cfg, try unfold(a, header_37b));
    try t.expect(r.missing.items.len > 0 and r.unknown.items.len > 0);
    const dense = try config_test.edited(a, try config_test.json37b(a), &.{.{ "dense_attention_layers", "[3]" }});
    const dense_cfg = try parsed(dense);
    const d = try names.check(a, &dense_cfg, try unfold(a, header_37b));
    try t.expectEqual(@as(usize, 5), d.unknown.items.len);
    try t.expectEqualStrings("model.layers.3.self_attn.indexer.k_norm.weight", d.unknown.items[0]);
}

test "a folded table whose count or digest is off is refused" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cut = header_37b[0 .. std.mem.lastIndexOfScalar(u8, header_37b[0 .. header_37b.len - 1], '\n').? + 1];
    try t.expectError(error.BadFixture, unfold(a, cut));
    const retagged = try std.mem.replaceOwned(u8, a, header_37b, "lm_head.weight BF16", "lm_head.weight F32");
    try t.expectError(error.BadFixture, unfold(a, retagged));
}

/// Every model shard's header from the directory TF_GIGACHAT4_HEADERS names (the 37B FP8 shards, or their headers).
fn realHeaders(a: std.mem.Allocator) !st.Header {
    const dir = std.mem.span(std.c.getenv("TF_GIGACHAT4_HEADERS") orelse return error.SkipZigTest);
    var all: st.Header = .empty;
    for (1..4) |i| {
        var buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "/model-{d:0>5}-of-00003.safetensors", .{i});
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

test "the map matches the 37B release's own headers (TF_GIGACHAT4_HEADERS)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const real = try realHeaders(a);
    const c = try parsed(try config_test.json37b(a));
    const r = try names.check(a, &c, real);
    try t.expect(r.clean());
    const table = try unfold(a, header_37b);
    try t.expectEqual(table.count(), real.count());
}
