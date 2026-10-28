//! Ling-3.0-tiny@9a98e35f's config.json and index names (fixtures/tiny_*) against the loader, where they differ.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const ling = @import("cuda.zig");
const names_test = @import("names_test.zig");

const t = std.testing;
const tiny = @embedFile("fixtures/tiny_config.json");
const folded = @embedFile("fixtures/tiny_names.txt");

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, tiny, &why);
}

fn refused(from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    if (std.mem.count(u8, tiny, from) != 1) return error.EditAnchor;
    const json = try std.mem.replaceOwned(u8, t.allocator, tiny, from, to);
    defer t.allocator.free(json);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, json, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "tiny's config.json parses: 24 layers in groups of 4, 16 heads, q_lora_rank 256, 128 experts, no MTP, no limits" {
    const c = try load();
    try t.expectEqual(@as(u32, 1536), c.hidden);
    try t.expectEqual(@as(u32, 24), c.layers);
    try t.expectEqual(@as(u32, 4), c.group);
    try t.expectEqual(@as(u32, 1), c.dense_layers);
    try t.expectEqual(@as(u32, 4608), c.dense_inter);
    try t.expectEqual(@as(u32, 16), c.kda_heads);
    try t.expectEqual(@as(u32, 16), c.mla_heads);
    try t.expectEqual(@as(u32, 256), c.q_lora);
    try t.expectEqual(@as(u32, 512), c.kv_lora);
    try t.expectEqual(@as(u32, 192), c.qkDim());
    try t.expect(c.head_gate);
    try t.expectEqual(@as(u32, 128), c.experts);
    try t.expectEqual(@as(u32, 8), c.topk);
    try t.expectEqual(@as(u32, 8), c.n_group);
    try t.expectEqual(@as(u32, 4), c.topk_group);
    try t.expectEqual(@as(u32, 512), c.moe_inter);
    try t.expectEqual(@as(u32, 512), c.shared_inter);
    try t.expectEqual(@as(u32, 0), c.mtp);
    try t.expectEqual(config.Quant.bf16, c.quant);
    for (0..c.layers) |i| {
        const l = ling.limits(&c, @intCast(i));
        try t.expectEqual(@as(f32, 0), l.routed + l.shared + l.dense); // "expert_swiglu_limit_list": null
    }
}

test "tiny's schedule: MLA on layers 3, 7, 11, 15, 19 and 23, KDA on the other 18, layer 0 dense, the rest MoE" {
    const c = try load();
    try t.expectEqual(@as(u32, 6), c.countKind(.mla));
    try t.expectEqual(@as(u32, 18), c.countKind(.kda));
    for (0..c.layers) |i| {
        const mla = i % 4 == 3;
        try t.expectEqual(if (mla) config.Kind.mla else config.Kind.kda, c.kind(@intCast(i)));
        try t.expectEqual(i != 0, c.isMoe(@intCast(i)));
    }
}

test "tiny's routing as glue.select takes it: 8 groups of 16 experts (not whole lane columns) keeping 4" {
    const c = try load();
    const g = ling.groups(&c);
    try t.expectEqual(@as(usize, 8), g.n);
    try t.expectEqual(@as(usize, 4), g.keep);
    try t.expect(c.experts / g.n == 16 and (c.experts / g.n) % 32 != 0); // the division branch of keep_groups
    try t.expect(c.topk <= g.keep * (c.experts / g.n));
}

test "tiny's index accounts for all 9,283 names, none of the MTP layer's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try load();
    const all = try names_test.unfoldOf(arena.allocator(), folded, "ling30tiny-names");
    try t.expectEqual(@as(usize, 9283), all.len);
    const r = try names.check(arena.allocator(), &c, all);
    for (r.missing.items) |n| std.debug.print("missing {s}\n", .{n});
    for (r.unknown.items) |n| std.debug.print("unaccounted {s}\n", .{n});
    try t.expectEqual(@as(usize, 0), r.missing.items.len + r.unknown.items.len);
    try t.expectEqual(@as(usize, 9283), r.expected);
    try t.expectEqual(@as(usize, 0), r.mtp);
}

test "what tiny shares with flash is still refused when it changes: other q_lora ranks, limit lists of the wrong length, MTP" {
    try refused("\"q_lora_rank\": 256", "\"q_lora_rank\": 384", error.MlaGeometry, "q_lora_rank 384");
    try refused("\"q_lora_rank\": 256", "\"q_lora_rank\": 128", error.MlaGeometry, "q_lora_rank 128");
    try refused("\"expert_swiglu_limit_list\": null", "\"expert_swiglu_limit_list\": [0, 10]", error.LimitList, "2 entries for 24 layers");
    try refused("\"hidden_size\": 1536", "\"hidden_size\": 1024", error.Width, "hidden_size 1024");
    try refused("\"num_attention_heads\": 16", "\"num_attention_heads\": 8", error.KdaGeometry, "key/value heads");
    try refused("\"num_experts\": 128", "\"num_experts\": 130", error.Routing, "past glue.select's bounds");
}
