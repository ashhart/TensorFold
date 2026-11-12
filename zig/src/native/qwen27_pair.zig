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

/// Raise when a command's words change shape (`encode` and `decode` below); `Kind` hashes in by itself.
const command_revision = 1;

/// The command set's word for `Pair.ready`: builds that send or replay different commands refuse each other by name.
pub const protocol: u64 = blk: {
    const kinds = @typeInfo(Kind).@"enum";
    var h = std.hash.Wyhash.init(command_revision);
    for (kinds.field_names, kinds.field_values) |name, value| {
        h.update(name);
        h.update(&std.mem.toBytes(@as(u32, value)));
    }
    break :blk h.final();
};

/// One command's fields: what rank 0's `Link` sends and rank 1's `follow` replays, each naming its slot first.
pub const Command = union(Kind) {
    begin: u32,
    chunk: struct { slot: u32, ids: []const u32, last: bool, decoded: bool },
    step: struct { slot: u32, token: u32 },
    end: u32,
    tree: struct { slot: u32, ids: []const u32, parents: []const u32 },
    commit: struct { slot: u32, path: []const u32 },

    pub fn slot(c: Command) u32 {
        return switch (c) {
            .begin, .end => |at| at,
            inline else => |f| f.slot,
        };
    }
};

/// A command's words into `out`, slot first: [slot], [slot n last decoded ids], [slot token] or [slot n rows...].
pub fn encode(out: *std.ArrayList(u32), gpa: std.mem.Allocator, c: Command) !void {
    out.clearRetainingCapacity();
    switch (c) {
        .begin, .end => |at| try out.append(gpa, at),
        .chunk => |f| {
            try out.appendSlice(gpa, &.{ f.slot, @intCast(f.ids.len), @intFromBool(f.last), @intFromBool(f.decoded) });
            try out.appendSlice(gpa, f.ids);
        },
        .step => |f| try out.appendSlice(gpa, &.{ f.slot, f.token }),
        .tree => |f| {
            std.debug.assert(f.ids.len == f.parents.len);
            try out.appendSlice(gpa, &.{ f.slot, @intCast(f.ids.len) });
            try out.appendSlice(gpa, f.ids);
            try out.appendSlice(gpa, f.parents);
        },
        .commit => |f| {
            try out.appendSlice(gpa, &.{ f.slot, @intCast(f.path.len) });
            try out.appendSlice(gpa, f.path);
        },
    }
}

/// `encode`'s words back, checked before any is read as a field: the length, the flags, then the slot against `slots`.
pub fn decode(kind: u32, w: []const u32, slots: u32) error{ UnknownCommand, CommandLength, CommandFlag, SlotOutOfRange }!Command {
    const c: Command = switch (std.enums.fromInt(Kind, kind) orelse return error.UnknownCommand) {
        .begin => if (w.len == 1) .{ .begin = w[0] } else return error.CommandLength,
        .end => if (w.len == 1) .{ .end = w[0] } else return error.CommandLength,
        .chunk => blk: {
            if (w.len < 4 or w.len - 4 != w[1]) return error.CommandLength;
            if (w[2] > 1 or w[3] > 1) return error.CommandFlag;
            break :blk .{ .chunk = .{ .slot = w[0], .ids = w[4..], .last = w[2] != 0, .decoded = w[3] != 0 } };
        },
        .step => if (w.len == 2) .{ .step = .{ .slot = w[0], .token = w[1] } } else return error.CommandLength,
        .tree => blk: {
            if (w.len < 2 or w.len - 2 != 2 * @as(usize, w[1])) return error.CommandLength;
            break :blk .{ .tree = .{ .slot = w[0], .ids = w[2..][0..w[1]], .parents = w[2 + w[1] ..] } };
        },
        .commit => blk: {
            if (w.len < 2 or w.len - 2 != w[1]) return error.CommandLength;
            break :blk .{ .commit = .{ .slot = w[0], .path = w[2..] } };
        },
    };
    return if (c.slot() < slots) c else error.SlotOutOfRange;
}

/// Rank 0 puts each round on the wire before running it; rank 1 replays them in order here. Owned by the host.
pub const Link = struct {
    pair: *Pair,
    gpa: std.mem.Allocator,
    words: std.ArrayList(u32) = .empty,

    pub fn deinit(l: *Link) void {
        l.words.deinit(l.gpa);
    }

    fn put(l: *Link, c: Command) !void {
        try encode(&l.words, l.gpa, c);
        try l.pair.send(@backingInt(std.meta.activeTag(c)), l.words.items);
        l.words.clearRetainingCapacity();
    }

    /// A prompt's chunk boundaries are rank 0's memory policy's, so the send carries them: rank 1 cannot infer them.
    pub fn begin(l: *Link, slot: u32) !void {
        try l.put(.{ .begin = slot });
    }

    pub fn chunk(l: *Link, slot: u32, ids: []const u32, last: bool, decoded: bool) !void {
        try l.put(.{ .chunk = .{ .slot = slot, .ids = ids, .last = last, .decoded = decoded } });
    }

    pub fn step(l: *Link, slot: u32, token: u32) !void {
        try l.put(.{ .step = .{ .slot = slot, .token = token } });
    }

    /// A draft round's verify: the rows (ids) with their parents; rank 1 replays the same verify round.
    pub fn tree(l: *Link, slot: u32, ids: []const u32, parents: []const i32) !void {
        try l.put(.{ .tree = .{ .slot = slot, .ids = ids, .parents = @ptrCast(parents) } });
    }

    /// The accepted path rank 0 read from its picks: rank 1 keeps the same rows so both caches agree.
    pub fn commit(l: *Link, slot: u32, path: []const u32) !void {
        try l.put(.{ .commit = .{ .slot = slot, .path = path } });
    }

    pub fn end(l: *Link, slot: u32) void {
        l.put(.{ .end = slot }) catch {};
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

/// Whether `measure` runs before the lane config copies its costs: one Mac's order (main's); a pair's seams need both.
pub fn measuresFirst(pair: ?*const Pair) bool {
    return pair == null;
}

/// A shared round's rows at most: the frame's.
pub const batch_rows = 128;

test "qwen27 pair: the served chunk times the widest seam fits the window" {
    // a served pass asks for 512 rows here, 2048 on generation 17 (`model.promptRows`); the pair caps it at the window
    const config = q.config.Config{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 124160, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
    const requested = q.model.promptRows(config, 0);
    const served = chunk(requested);
    try std.testing.expect(served <= batch_rows);
    try std.testing.expect(seamBytes(served) <= pairBytes());
    // the uncapped pass stages a seam wider than the window, so the cap is what keeps the exchange legal
    try std.testing.expectEqual(@as(usize, batch_rows * @divExact(17408, 2) * @sizeOf(u16)), pairBytes());
    try std.testing.expect(seamBytes(512) > pairBytes());
}

/// This rank's four identity words for `Pair.ready`: the engine build, chip class, checkpoint digest and widths tag.
pub fn pairFingerprint(a: std.mem.Allocator, io: std.Io, model: *q.model.Model, dir: []const u8) tf.pair_rows.Fingerprint {
    const chip = chipGeneration(std.mem.span(model.device.name()));
    const checkpoint: u64 = blk: {
        const digest = q.checkpoint_identity.identity(a, io, dir) catch break :blk 0;
        break :blk std.mem.readInt(u64, digest[0..8], .little);
    };
    return .{
        .version = engineVersion(),
        .chip = chip,
        .checkpoint = checkpoint,
        .format = model.widths_tag,
        .protocol = protocol,
    };
}

/// The Apple GPU generation a Metal device name carries ("Apple M4 Max" is 4), or 0 when it names none.
pub fn chipGeneration(name: []const u8) u64 {
    var words_it = std.mem.tokenizeScalar(u8, name, ' ');
    _ = words_it.next() orelse return 0; // "Apple"
    const m = words_it.next() orelse return 0; // "M4"
    if (m.len < 2 or std.ascii.toLower(m[0]) != 'm') return 0;
    return std.fmt.parseInt(u32, m[1..], 10) catch 0;
}

/// The engine build version's hash: 0 when this binary carries no version string, refused against any named one.
fn engineVersion() u64 {
    const v = @import("build_options").version;
    if (v.len == 0) return 0;
    var h = std.hash.Wyhash.init(0x6437);
    h.update(v);
    return h.final();
}
