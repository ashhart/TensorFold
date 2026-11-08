//! The pair transport's types, the seam ABI the engine calls, and the window one exchange may carry.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;

/// The pair's transport type: the row-split seam the engine exports, so this file names no other family's type.
pub const Pair = tf.pair_rows.Pair;

/// The command set rank 1 replays (glm_dsa's `follow.zig` shape); `.tree`/`.commit` carry a draft round.
pub const Kind = enum(u32) { begin = 1, chunk, step, end, tree, commit };

/// Rank 0 puts each round on the wire before running it; rank 1 replays them in order here. Owned by the host.
pub const Link = struct {
    pair: *Pair,
    gpa: std.mem.Allocator,
    words: std.ArrayList(u32) = .empty,

    pub fn deinit(l: *Link) void {
        l.words.deinit(l.gpa);
    }

    fn put(l: *Link, kind: Kind) !void {
        try l.pair.send(@backingInt(kind), l.words.items);
        l.words.clearRetainingCapacity();
    }

    /// A prompt's chunk boundaries are rank 0's memory policy's, so the send carries them: rank 1 cannot infer them.
    pub fn begin(l: *Link, slot: u32) !void {
        l.words.clearRetainingCapacity();
        try l.words.append(l.gpa, slot);
        try l.put(.begin);
    }

    pub fn chunk(l: *Link, slot: u32, ids: []const u32, last: bool, decoded: bool) !void {
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, @intCast(ids.len), @intFromBool(last), @intFromBool(decoded) });
        try l.words.appendSlice(l.gpa, ids);
        try l.put(.chunk);
    }

    pub fn step(l: *Link, slot: u32, token: u32) !void {
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, token });
        try l.put(.step);
    }

    /// A draft round's verify: the rows (ids) with their parents; rank 1 replays the same verify round.
    pub fn tree(l: *Link, slot: u32, ids: []const u32, parents: []const i32) !void {
        std.debug.assert(ids.len == parents.len);
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, @intCast(ids.len) });
        try l.words.appendSlice(l.gpa, ids);
        for (parents) |p| try l.words.append(l.gpa, @bitCast(p));
        try l.put(.tree);
    }

    /// The accepted path rank 0 read from its picks: rank 1 keeps the same rows so both caches agree.
    pub fn commit(l: *Link, slot: u32, path: []const u32) !void {
        l.words.clearRetainingCapacity();
        try l.words.appendSlice(l.gpa, &.{ slot, @intCast(path.len) });
        try l.words.appendSlice(l.gpa, path);
        try l.put(.commit);
    }

    pub fn end(l: *Link, slot: u32) void {
        l.words.clearRetainingCapacity();
        l.words.append(l.gpa, slot) catch return;
        l.put(.end) catch {};
    }
};

/// Rank 1 replays rank 0's rounds: refuse at the head of `prefill`, its first backend call (GLM's `beginPrompt`).
pub fn refuseFollower(link: ?*Link) !void {
    if (link) |l| if (l.pair.rank != 0) {
        std.log.info("q27 rank1 follower refusal: queued words={d} failed_seam={d}", .{ l.words.items.len, l.pair.failedSeam() });
        return error.FollowsPeer;
    };
}

/// A pair's token is the head seam's pick, so a sampled or grammar-constrained stream has no pair answer.
pub fn refuseSampling(link: ?*Link, s: *const lanes.Stream) !void {
    // refuse it at `prefill`'s head too, before either rank settles, drops or binds a round (no second argmax).
    if (link == null) return;
    if (s.sampling != null or s.grammar != null) return error.PairGreedyOnly;
}

/// The pair's seam ends in the engine's shape: the transport's `Ref` is the frame's, so the casts are field copies.
pub fn pairReduce(p: *Pair) q.forward.Reduce {
    return .{ .ptr = p, .part = pairPart, .stage = pairStage, .gather = pairGather, .failed = pairFailed, .argmax = pairArgmax };
}

fn pairPart(ptr: *anyopaque) q.forward.Seam {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    const s = p.part();
    return .{ .x = s.x, .ref = .{ .buffer = s.ref.buf, .offset = s.ref.off } };
}

fn pairStage(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, src: q.forward.Ref, rows: u32, width: u32, stride: u32) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.stage(e, x, .{ .buf = src.buffer, .off = src.offset }, rows, width, stride);
}

fn pairGather(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, mine: q.forward.Ref, out: q.forward.Ref, rows: u32, width: u32, out_stride: u32) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.gather(e, x, .{ .buf = mine.buffer, .off = mine.offset }, .{ .buf = out.buffer, .off = out.offset }, rows, width, out_stride);
}

fn pairFailed(ptr: *anyopaque) bool {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    return p.failed();
}

fn pairArgmax(ptr: *anyopaque, e: *mtl.ComputeEncoder, x: u32, logits: q.forward.Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: q.forward.Ref) void {
    const p: *Pair = @ptrCast(@alignCast(ptr));
    p.argmaxBf16(e, x, .{ .buf = logits.buffer, .off = logits.offset }, stride, lo, n, rows, .{ .buf = picks.buffer, .off = picks.offset });
}

/// The window an exchange may carry: a round's rows of the widest seam payload -- this rank's `intermediate / 2` bf16.
pub fn pairBytes() usize {
    return seamBytes(batch_rows);
}

/// The bytes a round of `rows` stages at the widest seam: this rank's `intermediate / 2` bf16.
pub fn seamBytes(rows: u32) usize {
    return @as(usize, rows) * @divExact(17408, 2) * @sizeOf(u16);
}

/// A pair's prompt pass at most: a chunk of the window's rows, so its widest seam fits `pairBytes()`.
pub fn chunk(requested: u32) u32 {
    return @min(requested, batch_rows);
}

/// Whether slot 0 resumes in place: one Mac does, a pair never (rank 1 resets the slot at rank 0's `begin`).
pub fn resumesInPlace(pair: ?*const Pair) bool {
    return pair == null;
}

/// The in-place resume a head keeps: `init`'s switch `Metal.init` read from `TF_QWEN27_SLOT_RESUME`, never a pair's.
pub fn slotResume(init: bool, pair: ?*const Pair) bool {
    return init and resumesInPlace(pair);
}

/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;

test "the served pair's chunk times the widest seam fits the window" {
    // a served pass asks for 512 rows here, 2048 on generation 17 (`model.promptRows`); the pair caps it at the window
    const requested = q.model.promptRows(0);
    const served = chunk(requested);
    try std.testing.expect(served <= batch_rows);
    try std.testing.expect(seamBytes(served) <= pairBytes());
    // the uncapped pass stages a seam wider than the window, so the cap is what keeps the exchange legal
    try std.testing.expectEqual(@as(usize, batch_rows * @divExact(17408, 2) * @sizeOf(u16)), pairBytes());
    try std.testing.expect(seamBytes(512) > pairBytes());
}

/// This rank's four identity words for `Pair.ready`: the engine build, chip class, checkpoint digest and widths tag.
pub fn pairFingerprint(a: std.mem.Allocator, io: std.Io, model: *q.model.Model, dir: []const u8) tf.pair_rows.Fingerprint {
    const generation = blk: {
        const name = std.mem.span(model.device.name());
        var words_it = std.mem.tokenizeScalar(u8, name, ' ');
        _ = words_it.next() orelse break :blk 0; // "Apple"
        const m = words_it.next() orelse break :blk 0; // "M4"
        if (m.len < 2 or std.ascii.toLower(m[0]) != 'm') break :blk 0;
        break :blk std.fmt.parseInt(u32, m[1..], 10) catch 0;
    };
    const chip: u64 = if (generation >= 1) generation else 0;
    const checkpoint: u64 = blk: {
        const digest = q.checkpoint_identity.identity(a, io, dir) catch break :blk 0;
        break :blk std.mem.readInt(u64, digest[0..8], .little);
    };
    return .{
        .version = engineVersion(),
        .chip = chip,
        .checkpoint = checkpoint,
        .format = model.widths_tag,
    };
}

/// The engine build version's hash: 0 when this binary carries no version string, refused against any named one.
fn engineVersion() u64 {
    const v = @import("build_options").version;
    if (v.len == 0) return 0;
    var h = std.hash.Wyhash.init(0x6437);
    h.update(v);
    return h.final();
}
