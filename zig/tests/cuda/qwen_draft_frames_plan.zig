//! A fixed full-block draft schedule and canonical records, independent of a frame's stream order.
const std = @import("std");
pub const streams = 8;
pub const rounds = 6;
pub const block = 16;
pub const depth = block - 1;
pub const word_bytes = depth * 43 * 4;
pub const lengths = [streams]usize{ 37, 64, 129, 200, 300, 511, 700, 1000 };
pub const Mode = enum { alone, together, mixed_a, mixed_b, prompt_control, swap_control };
pub const Partition = struct { mode: Mode, groups: []const []const usize };
pub const partitions = [_]Partition{
    .{ .mode = .alone, .groups = &.{ &.{0}, &.{1}, &.{2}, &.{3}, &.{4}, &.{5}, &.{6}, &.{7} } },
    .{ .mode = .together, .groups = &.{&.{ 0, 1, 2, 3, 4, 5, 6, 7 }} },
    .{ .mode = .mixed_a, .groups = &.{ &.{ 3, 0, 7, 1 }, &.{ 5, 2 }, &.{ 6, 4 } } },
    .{ .mode = .mixed_b, .groups = &.{ &.{6}, &.{ 1, 4, 0, 2, 5, 3, 7 } } },
};

pub fn validate(p: Partition) !void {
    var seen: [streams]bool = @splat(false);
    for (p.groups) |group| {
        if (group.len == 0 or group.len > streams) return error.BadFramePartition;
        for (group) |id| {
            if (id >= streams or seen[id]) return error.BadFramePartition;
            seen[id] = true;
        }
    }
    for (seen) |present| if (!present) return error.BadFramePartition;
}

/// No cost rule or acceptance-dependent scheduling: each stream visits one full-block frame in every round.
pub fn schedule(ops: anytype, p: Partition) !void {
    try validate(p);
    for (0..rounds) |round| for (p.groups) |group| try ops.frame(round, group);
}

pub const Difference = struct { round: usize, stream: usize, byte: usize };
pub const Capture = struct {
    a: std.mem.Allocator,
    rank: usize,
    stride: usize,
    bytes: []u8,
    seen: [streams * rounds]bool = @splat(false),

    pub fn init(a: std.mem.Allocator, rank: usize) !Capture {
        if (rank == 0) return error.BadSelectorRank;
        const selector = try std.math.mul(usize, try std.math.mul(usize, depth, rank), 2);
        const stride = try std.math.add(usize, depth * 4 + word_bytes + 4, selector);
        const bytes = try a.alloc(u8, try std.math.mul(usize, stride, streams * rounds));
        @memset(bytes, 0);
        return .{ .a = a, .rank = rank, .stride = stride, .bytes = bytes };
    }

    pub fn deinit(c: *Capture) void {
        c.a.free(c.bytes);
    }

    pub fn slot(c: Capture, round: usize, id: usize) ![]u8 {
        if (round >= rounds or id >= streams) return error.BadFrameRecord;
        return c.bytes[(round * streams + id) * c.stride ..][0..c.stride];
    }

    pub fn write(c: *Capture, round: usize, id: usize, proposals: []const u32, words: []const u8, selector: []const u8, landed: usize) !void {
        const dst = try c.slot(round, id);
        if (c.seen[round * streams + id] or proposals.len != depth or words.len != word_bytes or selector.len != depth * c.rank * 2 or landed > depth) return error.BadFrameRecord;
        for (proposals, 0..) |token, i| std.mem.writeInt(u32, dst[4 * i ..][0..4], token, .little);
        @memcpy(dst[depth * 4 ..][0..word_bytes], words);
        @memcpy(dst[depth * 4 + word_bytes ..][0..selector.len], selector);
        std.mem.writeInt(u32, dst[dst.len - 4 ..][0..4], @intCast(landed), .little);
        c.seen[round * streams + id] = true;
    }

    pub fn complete(c: Capture) !void {
        for (c.seen) |present| if (!present) return error.MissingFrameRecord;
    }

    /// Byte arrays stay numeric in JSON, including valid UTF-8 and non-UTF-8 payloads alike.
    pub fn json(c: Capture, out: *std.Io.Writer, mode: Mode) !void {
        for (0..rounds) |round| for (0..streams) |id| {
            if (!c.seen[round * streams + id]) continue;
            const bytes = try c.slot(round, id);
            var proposals: [depth]u32 = undefined;
            for (&proposals, 0..) |*token, i| token.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
            try std.json.Stringify.value(.{ .mode = mode, .stream = id, .round = round, .proposals = proposals, .accepted = std.mem.readInt(u32, bytes[bytes.len - 4 ..][0..4], .little), .words = bytes[depth * 4 ..][0..word_bytes], .selector = bytes[depth * 4 + word_bytes .. bytes.len - 4] }, .{ .emit_strings_as_arrays = true }, out);
            try out.writeByte('\n');
        };
    }

    pub fn difference(c: Capture, other: Capture) !?Difference {
        try c.complete();
        try other.complete();
        if (c.rank != other.rank or c.stride != other.stride) return error.BadFrameRecord;
        for (0..rounds) |round| for (0..streams) |id| {
            if (std.mem.indexOfDiff(u8, try c.slot(round, id), try other.slot(round, id))) |byte|
                return .{ .round = round, .stream = id, .byte = byte };
        };
        return null;
    }

    /// The misassignment control moves real payloads, not stream labels or comparison keys.
    pub fn swapStreams(c: *Capture, first: usize, second: usize) !void {
        try c.complete();
        if (first == second) return error.BadFrameRecord;
        for (0..rounds) |round| {
            const x = try c.slot(round, first);
            const y = try c.slot(round, second);
            for (x, y) |*a, *b| std.mem.swap(u8, a, b);
        }
    }
};

/// Compare all payload bytes, and require a negative control to produce an actual difference.
pub fn compare(base: Capture, candidate: Capture, control: bool) !?Difference {
    const diff = try base.difference(candidate);
    if (control and diff == null) return error.FrameControlNotDetected;
    if (!control and diff != null) return error.DraftFrameDependent;
    return diff;
}

pub fn accepted(proposals: []const u32, sampled: []const u32) !usize {
    if (proposals.len != depth or sampled.len != block) return error.BadFrameRecord;
    var n: usize = 0;
    while (n < depth and proposals[n] == sampled[n]) : (n += 1) {}
    return n;
}
