//! The weight map against the checkpoint's index: fixtures/names.txt folds its 214,025 names.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");

const t = std.testing;
const real = @embedFile("fixtures/config.json");
const folded = @embedFile("fixtures/names.txt");

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, real, &why);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Each `{L}` line over its layer ranges, each `{E}` line over its expert count; the header holds count and sha256.
fn unfold(arena: std.mem.Allocator) ![]const []const u8 {
    return unfoldOf(arena, folded, "ling31-names");
}

/// `unfold` for any folded fixture: `tag` is the header's first word (tiny_test.zig folds Ling-3.0-tiny's index).
pub fn unfoldOf(arena: std.mem.Allocator, text: []const u8, tag: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next().?, ' ');
    if (!std.mem.eql(u8, head.next().?, tag)) return error.BadFixture;
    const count = try std.fmt.parseInt(usize, head.next().?, 10);
    const digest = head.next().?;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pat = f.next().?;
        const layers = f.next() orelse {
            try out.append(arena, pat);
            continue;
        };
        const experts = if (f.next()) |e| try std.fmt.parseInt(u32, e, 10) else 1;
        var spans = std.mem.tokenizeScalar(u8, layers, ',');
        while (spans.next()) |span| {
            const dash = std.mem.indexOfScalar(u8, span, '-');
            const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
            for (lo..hi + 1) |l| for (0..experts) |e| {
                var lb: [8]u8 = undefined;
                var eb: [8]u8 = undefined;
                const a = try std.mem.replaceOwned(u8, arena, pat, "{L}", try std.fmt.bufPrint(&lb, "{d}", .{l}));
                try out.append(arena, try std.mem.replaceOwned(u8, arena, a, "{E}", try std.fmt.bufPrint(&eb, "{d}", .{e})));
            };
        }
    }
    std.mem.sort([]const u8, out.items, {}, lessThan);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (out.items, 0..) |n, i| {
        if (i > 0) h.update("\n");
        h.update(n);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    if (out.items.len != count or !std.mem.eql(u8, &std.fmt.bytesToHex(d, .lower), digest)) return error.FixtureNotIndex;
    return out.items;
}

test "the folded fixture unfolds to the index's exact name list (count and sha256 of the sorted names)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(usize, 214025), (try unfold(arena.allocator())).len);
}

test "the map accounts for every index name and finds each it expects: 210,925 backbone, 3,100 MTP layer 72" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try load();
    const r = try names.check(arena.allocator(), &c, try unfold(arena.allocator()));
    for (r.missing.items) |n| std.debug.print("missing {s}\n", .{n});
    for (r.unknown.items) |n| std.debug.print("unaccounted {s}\n", .{n});
    try t.expectEqual(@as(usize, 0), r.missing.items.len);
    try t.expectEqual(@as(usize, 0), r.unknown.items.len);
    try t.expectEqual(@as(usize, 214025), r.expected);
    try t.expectEqual(@as(usize, 3100), r.mtp);
}

const Count = struct {
    n: usize = 0,
    scales: usize = 0,
    roles: std.EnumSet(names.Role) = .empty,

    pub fn put(k: *Count, name: []const u8, x: names.Tensor) !void {
        _ = name;
        k.n += 1;
        k.scales += @intFromBool(x.scale);
        k.roles.insert(x.role);
    }
};

test "per layer: KDA dense 0, KDA MoE 4, MLA MoE 7, the MTP layer 72" {
    const c = try load();
    var k0: Count = .{};
    try names.layer(&c, 0, &k0);
    try t.expectEqual(@as(usize, 2 + 13 + 6 + 6), k0.n); // norms, KDA parts, their 6 FP8 scales (b_proj is bf16), dense MLP
    try t.expect(k0.roles.contains(.kda_a_log) and k0.roles.contains(.dense_down) and !k0.roles.contains(.router));
    var k4: Count = .{};
    try names.layer(&c, 4, &k4);
    try t.expectEqual(@as(usize, 2 + 19 + 1 + 1 + 6 + 512 * 6), k4.n); // the router is bf16 and has a bias
    var k7: Count = .{};
    try names.layer(&c, 7, &k7);
    try t.expectEqual(@as(usize, 2 + 3 + 2 + 4 + 3 + 1 + 2 + 6 + 512 * 6), k7.n); // g_proj of 7 is bf16
    try t.expect(k7.roles.contains(.mla_gate) and !k7.roles.contains(.kda_q));
    var k72: Count = .{};
    try names.layer(&c, 72, &k72);
    try t.expectEqual(@as(usize, 3100), k72.n);
    try t.expect(k72.roles.contains(.mtp_eh) and k72.roles.contains(.mtp_final_norm));
}

test "a config that disagrees with the index is caught: the router taken as FP8 misses 69 scales" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const json = try std.mem.replaceOwned(u8, t.allocator, real, "\"mlp.gate\",", "\"mlp.gate_x\",");
    defer t.allocator.free(json);
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, json, &why);
    const r = try names.check(arena.allocator(), &c, try unfold(arena.allocator()));
    try t.expectEqual(@as(usize, 69), r.missing.items.len);
    try t.expectEqualStrings("model.layers.4.mlp.gate.weight_scale_inv", r.missing.items[0]);
    try t.expectEqual(@as(usize, 0), r.unknown.items.len);
}

test "tied embeddings drop lm_head, bf16 drops every scale, no head gate drops MLA's g_proj" {
    var c = try load();
    c.tie = true;
    c.quant = .bf16;
    c.head_gate = false;
    var g: Count = .{};
    try names.globals(&c, &g);
    try t.expectEqual(@as(usize, 2), g.n);
    var k7: Count = .{};
    try names.layer(&c, 7, &k7);
    try t.expectEqual(@as(usize, 0), k7.scales);
    try t.expect(!k7.roles.contains(.mla_gate));
}

test "the real index (TF_LING_INDEX=<model.safetensors.index.json>) against the map" {
    const path = std.mem.span(std.c.getenv("TF_LING_INDEX") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .unlimited);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const map = doc.object.get("weight_map").?.object;
    const c = try load();
    const r = try names.check(a, &c, map.keys());
    try t.expectEqual(@as(usize, 0), r.missing.items.len + r.unknown.items.len);
    try t.expectEqual(@as(usize, map.count()), r.expected);
}

test "NVFP4 linears store weight, weight_scale, weight_scale_2 and input_scale; the module names the loader reads" {
    var c = try load();
    c.quant = .nvfp4;
    var k0: Count = .{};
    try names.layer(&c, 0, &k0);
    try t.expectEqual(@as(usize, 2 + 13 + 6 * 3 + 3 + 3 * 3), k0.n); // norms, KDA, 3 side tensors a quantized linear, dense
    var b: [192]u8 = undefined;
    try t.expectEqualStrings("model.layers.4.mlp.experts.511.down_proj", try names.moduleName(&b, 4, .expert_down, 511));
    try t.expectEqualStrings("model.layers.7.attention.dense", try names.moduleName(&b, 7, .mla_o, null));
    try t.expectEqualStrings("model.layers.0.attention.A_log", try names.moduleName(&b, 0, .kda_a_log, null));
    try t.expectEqualStrings("model.layers.72.eh_proj", try names.moduleName(&b, 72, .mtp_eh, null));
    try t.expectEqualStrings("lm_head", try names.moduleName(&b, 0, .lm_head, null));
}
