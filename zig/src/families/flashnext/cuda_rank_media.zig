//! Control carries bounded image spans. NCCL broadcasts encoded BF16 rows from the only tower, on rank zero.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const wire = @import("cuda_rank_wire.zig");
const Session = @import("cuda_rank_engine.zig").Session;
const shared = @import("qwen_vision");
pub fn send(s: *Session, q: *prompt.Seq) !void {
    const m = &(q.media orelse return error.BadImageCount);
    var raw: [64 * 12]u8 = undefined;
    const runs = m.image.plan.runs;
    for (runs, 0..) |r, i| {
        std.mem.writeInt(u32, raw[i * 12 ..][0..4], r.at, .little);
        std.mem.writeInt(u32, raw[i * 12 + 4 ..][0..4], r.gh, .little);
        std.mem.writeInt(u32, raw[i * 12 + 8 ..][0..4], r.gw, .little);
    }
    const bytes = raw[0 .. runs.len * 12];
    const h = try s.command(.{ .kind = .media, .object = q.rank_id, .arg = m.image.plan.src.len, .bytes = @intCast(bytes.len) }, bytes);
    // after the poison: a broadcast that may be enqueued drains before the caller can free the features
    errdefer shared.completion.drain(s.engine.driver, s.engine.stream.*);
    errdefer s.poison();
    try s.collective.broadcast(s.engine.stream.*, m.image.features.ptr, m.image.features.len);
    try s.collective.finish(s.engine.stream.*);
    try s.ack(h);
}
pub fn receive(s: *Session, q: *prompt.Seq, h: wire.Header, payload: []const u8) !void {
    if (q.media != null or q.fed != 0 or h.arg > q.cap) return error.BadRankFrame;
    const v = s.engine.image_factory orelse return error.VisionNotEnabled;
    var runs: [64]shared.media.Run = undefined;
    const n = try decode(payload, &runs);
    q.media = try v.receive(@intCast(h.arg), runs[0..n]);
    const m = &q.media.?;
    // q keeps the features; a failed transfer drains the stream before any teardown can free them
    errdefer shared.completion.drain(s.engine.driver, s.engine.stream.*);
    try s.collective.broadcast(s.engine.stream.*, m.image.features.ptr, m.image.features.len);
    try s.collective.finish(s.engine.stream.*);
}
pub fn decode(payload: []const u8, out: []shared.media.Run) !usize {
    if (payload.len == 0 or payload.len % 12 != 0 or payload.len / 12 > out.len or payload.len / 12 > 64)
        return error.BadRankFrame;
    for (out[0 .. payload.len / 12], 0..) |*r, i| r.* = .{
        .at = std.mem.readInt(u32, payload[i * 12 ..][0..4], .little),
        .gh = std.mem.readInt(u32, payload[i * 12 + 4 ..][0..4], .little),
        .gw = std.mem.readInt(u32, payload[i * 12 + 8 ..][0..4], .little),
    };
    return payload.len / 12;
}
test "rank media frames are bounded and rebuild the same image plan" {
    var raw: [24]u8 = @splat(0);
    for ([_]u32{ 2, 4, 6, 10, 2, 2 }, 0..) |x, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], x, .little);
    var runs: [64]shared.media.Run = undefined;
    const n = try decode(&raw, &runs);
    var p = try shared.media.Plan.init(std.testing.allocator, 14, runs[0..n], .{});
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 7), p.rows);
    try std.testing.expectEqual(@as(i32, 6), p.src[10]);
    try std.testing.expectError(error.BadRankFrame, decode(raw[0..23], &runs));
    try std.testing.expectError(error.BadRankFrame, decode(&raw, runs[0..1]));
}

test {
    if (@import("builtin").os.tag != .windows) _ = @import("cuda_rank_media_test.zig"); // over a Unix socket pair
}
