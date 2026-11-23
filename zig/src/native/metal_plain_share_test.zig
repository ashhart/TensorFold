//! Nemotron without an MTP head: a shared plain window, not a widened one.
const std = @import("std");
const tf = @import("tensorfold");
const nemotron = tf.nemotron;
const lanes = tf.lanes;

test "plain sharing: Nemotron without a head shares streams without widening its window" {
    var b: nemotron.backend.Metal = undefined;
    b.o = .{ .capacity = 4096, .streams = 4, .batch_rows = 32, .drafts = false };
    b.head = null;
    b.costs = .{};
    const facts = b.facts();
    try std.testing.expect(facts.plain_share and facts.hidden_rows);
    try std.testing.expectEqual(@as(u32, 1), facts.exact_width);
    try std.testing.expect(!facts.mtp and !facts.speculate and !facts.allocate_shared_drafts);
    var config = try lanes.Config.init(std.testing.allocator, facts, nemotron.backend.max_window, nemotron.backend.max_window - 1);
    defer config.deinit(std.testing.allocator);
    try std.testing.expect(config.family_streams and config.pipelined);
    try std.testing.expectEqual(@as(u32, 4), config.batch_streams);
    try std.testing.expectEqual(@as(u32, 1), config.family_width);
    try std.testing.expectEqual(@as(u32, 0), config.max_copy);
    try std.testing.expectEqual(@as(u32, 0), config.most_drafts);
    try std.testing.expect(!config.family_mtp);
}
