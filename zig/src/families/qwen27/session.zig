//! One greedy stream without drafts over rendered ids: prompts in chunks of up to 128 rows, windows of 16.
const std = @import("std");
const q = @import("decode_round.zig");
const Round = @import("round_plan.zig").Round;

pub const Session = struct {
    runner: *q.Runner,
    reference_tree: bool = false,
    slot: u32 = 0, // the runner slot whose stream this feeds
    pub fn prefill(s: Session, ids: []const u32, chunk: usize) !void {
        if (ids.len == 0 or chunk == 0 or chunk > s.runner.model.frame.capacity or ids.len > s.runner.capacity - s.runner.offsets[s.slot]) return error.BadPrompt;
        s.runner.model.kernels.prompt = true;
        defer s.runner.model.kernels.prompt = false;
        var first: usize = 0;
        while (first < ids.len) {
            const end = @min(ids.len, first + chunk);
            try s.feed(ids[first..end], if (end == ids.len) .last else .none);
            first = end;
        }
    }
    pub fn promptChunk(s: Session, ids: []const u32, last: bool) !void {
        if (ids.len == 0 or ids.len > s.runner.model.frame.capacity) return error.BadPrompt;
        s.runner.model.kernels.prompt = true;
        defer s.runner.model.kernels.prompt = false;
        try s.feed(ids, if (last) .last else .none);
    }

    /// Prompt rows with decoded rows' arithmetic (an earlier reply's tokens): the bits decoding them gave.
    pub fn decodeChunk(s: Session, ids: []const u32, last: bool) !void {
        if (ids.len == 0 or ids.len > s.runner.model.frame.capacity) return error.BadPrompt;
        try s.feed(ids, if (last) .last else .none);
    }

    pub fn step(s: Session, token: u32) !void {
        try s.feed(&.{token}, .last);
    }
    pub fn greedy(s: Session) !u32 {
        if (s.runner.failed or s.runner.active != null or s.runner.offsets[s.slot] == 0) return error.RoundNotReady;
        const values = s.runner.model.frame.get(.logits).buffer.slice(u16, s.runner.model.config.vocab);
        return argmax(values);
    }
    /// Hash committed recurrent bytes, visible KV prefixes and the last row's logits (not unused capacity).
    pub fn fingerprint(s: Session) ![32]u8 {
        const r = s.runner;
        if (r.active != null or r.failed or r.offsets[0] == 0) return error.RoundNotReady;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(std.mem.asBytes(&r.offsets[0]));
        for (0..r.gdn.budget.layers) |layer| {
            hash.update(try r.gdn.committed(layer, 0, true));
            hash.update(try r.gdn.committed(layer, 0, false));
        }
        for (0..r.caches.len / r.slots) |layer| {
            const cache = r.caches[layer * r.slots];
            for ([_]@import("projection.zig").Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }) |ref, stride| {
                const head_stride = if (stride == 0) cache.capacity * 256 else stride;
                for (0..r.model.config.kv_heads) |head| hash.update(ref.buffer.contents()[ref.offset + head * @as(usize, head_stride) * 2 ..][0 .. @as(usize, r.offsets[0]) * 256 * 2]);
            }
        }
        hash.update(r.model.frame.get(.logits).buffer.contents()[0 .. r.model.config.vocab * 2]);
        var out: [32]u8 = undefined;
        hash.final(&out);
        return out;
    }

    fn feed(s: Session, ids: []const u32, head: @import("forward.zig").Head) !void {
        var round = try Round.init(s.runner.allocator, &.{.{ .slot = s.slot, .start = s.runner.offsets[s.slot], .capacity = s.runner.capacity, .ids = ids }}, @intCast(s.runner.model.config.conv_kernel), s.runner.slots, @intCast(s.runner.model.config.vocab));
        defer round.deinit();
        if (!s.reference_tree) {
            try s.runner.advance(&round, head);
            return;
        }
        try s.runner.verifyHead(&round, head);
        var path: [128]u32 = undefined;
        for (path[0..ids.len], 0..) |*v, i| v.* = @intCast(i);
        try s.runner.keep(&round, &.{path[0..ids.len]});
    }
};

/// BF16 head matches the Python row backend; ties choose the lowest token ID.
pub fn argmax(values: []const u16) !u32 {
    if (values.len == 0 or values.len > std.math.maxInt(u32)) return error.BadLogits;
    var best: f32 = -std.math.inf(f32);
    var id: u32 = 0;
    for (values, 0..) |bits, i| {
        const v: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        if (v > best) {
            best = v;
            id = @intCast(i);
        }
    }
    return id;
}

test "greedy selection handles negative logits and chooses the first tie" {
    try std.testing.expectEqual(@as(u32, 1), try argmax(&.{ 0xc000, 0x3f80, 0x3f80, 0xbf80 }));
    try std.testing.expectEqual(@as(u32, 0), try argmax(&.{ 0xbf80, 0xc000 }));
    try std.testing.expectError(error.NonfiniteLogits, argmax(&.{0x7fc0}));
    try std.testing.expectError(error.BadLogits, argmax(&.{}));
}

pub const TopTwo = struct { ids: [2]u32, logits: [2]f32 };
pub fn topTwo(values: []const u16) !TopTwo {
    if (values.len < 2 or values.len > std.math.maxInt(u32)) return error.BadLogits;
    var result = TopTwo{ .ids = .{ 0, 1 }, .logits = .{ -std.math.inf(f32), -std.math.inf(f32) } };
    for (values, 0..) |bits, i| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        if (value > result.logits[0]) {
            result.ids[1] = result.ids[0];
            result.logits[1] = result.logits[0];
            result.ids[0] = @intCast(i);
            result.logits[0] = value;
        } else if (value > result.logits[1]) {
            result.ids[1] = @intCast(i);
            result.logits[1] = value;
        }
    }
    return result;
}
test "top-two margins retain first-ID ties and reject nonfinite scores" {
    const top = try topTwo(&.{ 0x3f80, 0x4000, 0x4000, 0xbf80 });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, &top.ids);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2 }, &top.logits);
    try std.testing.expectError(error.NonfiniteLogits, topTwo(&.{ 0x3f80, 0x7fc0 }));
}
