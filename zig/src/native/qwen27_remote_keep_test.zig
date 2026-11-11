//! A reply's end on a pulled prefix is kept only with TF_PREFILL_KEEP, and then marked remote.
const std = @import("std");
const api = @import("engine_api");
const pc = api.prompt_cache;
const tf = @import("tensorfold");
const lanes = tf.lanes;

/// `keepEnd` is the served host's reply keep; `FakeStates` its tests' fake family (bytes grow with the rows kept).
pub fn exercise(keepEnd: anytype, FakeStates: type) !void {
    const gpa = std.testing.allocator;
    const at: u32 = 5000; // a history mark at 5000 tokens, then a 7-token generation prompt and a 100-token reply
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, .{ .warm = .on }, (4 * at + 200) << 16);
    defer store.deinit();
    var tokens: [at + 107]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    const prompt = tokens[0 .. at + 7];
    try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
    var s = try lanes.Stream.init(gpa, .{ .id = "reply", .prompt = prompt, .max_new = 100 });
    defer s.deinit(gpa);
    try s.context.appendSlice(gpa, tokens[at + 7 ..]);
    keepEnd(&store, false, gpa, &s, tokens.len); // an own reply is kept as the engine's
    try std.testing.expectEqual(@as(u64, 2), store.counts.kept);
    try std.testing.expect(!store.entries.items[store.entries.items.len - 1].remote);
    s.reuse.remote = true; // a reply on a pulled prefix (#1594 step 2b): with TF_PREFILL_KEEP off, never kept
    keepEnd(&store, false, gpa, &s, tokens.len - 1);
    try std.testing.expectEqual(@as(u64, 2), store.counts.kept);
    keepEnd(&store, true, gpa, &s, tokens.len - 1); // with it on, kept and marked remote
    try std.testing.expectEqual(@as(u64, 3), store.counts.kept);
    try std.testing.expect(store.entries.items[store.entries.items.len - 1].remote and !store.keep_remote);
}
