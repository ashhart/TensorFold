//! The layer plans of both releases, a moved or dense attention layer, and the norm kinds by site.
const std = @import("std");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const norms = @import("norms.zig");
const plan = @import("plan.zig");

const t = std.testing;

fn parsed(json: []const u8) !config.Config {
    var p: config.Problem = .{};
    return config.parse(t.allocator, json, &p);
}

test "37B: 24 Gated DeltaNet and 8 DSA layers, one dense MLP first, the MTP layer at 32" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try parsed(try config_test.json37b(arena.allocator()));
    const p = plan.build(&c);
    try t.expectEqual(@as(u32, 24), p.count(.gated_deltanet));
    try t.expectEqual(@as(u32, 8), p.count(.mla_dsa));
    try t.expectEqual(@as(u32, 0), p.count(.mla_dense));
    try t.expectEqual(@as(u32, 1), p.dense());
    try t.expectEqual(plan.Layer{ .mixer = .gated_deltanet, .ffn = .dense }, p.layers[0]);
    try t.expectEqual(plan.Layer{ .mixer = .gated_deltanet, .ffn = .moe }, p.layers[1]);
    try t.expectEqual(plan.Layer{ .mixer = .mla_dsa, .ffn = .moe }, p.layers[3]);
    try t.expectEqual(plan.Layer{ .mixer = .gated_deltanet, .ffn = .moe }, p.layers[30]);
    try t.expectEqual(plan.Layer{ .mixer = .mla_dsa, .ffn = .moe }, p.layers[31]);
    try t.expectEqual(plan.Mtp{ .index = 32, .sparse = false }, p.mtp);
    for (0..32) |i| try t.expectEqual(i % 4 == 3, p.layers[i].mixer == .mla_dsa);
}

test "708B: 48 Gated DeltaNet and 16 DSA layers, three dense MLPs first, the MTP layer at 64" {
    const c = try parsed(config_test.json_708b);
    const p = plan.build(&c);
    try t.expectEqual(@as(u32, 48), p.count(.gated_deltanet));
    try t.expectEqual(@as(u32, 16), p.count(.mla_dsa));
    try t.expectEqual(@as(u32, 3), p.dense());
    try t.expectEqual(plan.Layer{ .mixer = .gated_deltanet, .ffn = .dense }, p.layers[2]);
    try t.expectEqual(plan.Layer{ .mixer = .mla_dsa, .ffn = .moe }, p.layers[3]);
    try t.expectEqual(plan.Layer{ .mixer = .mla_dsa, .ffn = .moe }, p.layers[63]);
    try t.expectEqual(@as(u32, 64), p.mtp.index);
}

test "dense_attention_layers turns a DSA layer into MLA without an indexer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parsed(try config_test.edited(a, config_test.json_708b, &.{.{ "dense_attention_layers", "[7]" }}));
    const p = plan.build(&c);
    try t.expectEqual(plan.Mixer.mla_dsa, p.layers[3].mixer);
    try t.expectEqual(plan.Mixer.mla_dense, p.layers[7].mixer);
    try t.expectEqual(@as(u32, 15), p.count(.mla_dsa));
}

test "every norm site has its kind, width, epsilon and gate constant" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try parsed(try config_test.json37b(arena.allocator()));
    try t.expectEqual(norms.Kind.gated, norms.kind(.input));
    try t.expectEqual(norms.Kind.gated, norms.kind(.kv_a));
    try t.expectEqual(norms.Kind.gated, norms.kind(.mtp_head));
    try t.expectEqual(norms.Kind.rms, norms.kind(.index_key));
    try t.expectEqual(norms.Kind.gated_rms_sigmoid, norms.kind(.delta_output));
    try t.expectEqual(@as(u32, 2560), norms.width(&c, .post_feedforward));
    try t.expectEqual(@as(u32, 1536), norms.width(&c, .q_a));
    try t.expectEqual(@as(u32, 512), norms.width(&c, .kv_a));
    try t.expectEqual(@as(u32, 128), norms.width(&c, .index_key));
    try t.expectEqual(@as(u32, 128), norms.width(&c, .delta_output));
    try t.expectEqual(@as(f64, 1e-6), norms.eps(&c, .final));
    try t.expectEqual(@as(f64, 2), norms.gateConstant(&c, .input));
    try t.expectEqual(@as(f64, 2), norms.gateConstant(&c, .delta_output));
    try t.expectEqual(@as(f64, 1), norms.gateConstant(&c, .index_key));
}
