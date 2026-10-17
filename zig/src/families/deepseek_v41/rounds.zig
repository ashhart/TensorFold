//! A round plan: what rank 0 decided for one round, run by both ranks in order, as strict int64 plan-link words.

const std = @import("std");

pub const magic: i64 = 0x0D541;
pub const version: i64 = 2;

/// The prompt mode of the round's pieces: exact prefill (`full`) or the CED encoder-decoder replay.
pub const Mode = enum(u8) { full, replay };
/// Where an admission's cached prefix comes from.
pub const Tier = enum(u8) { none, ram, disk };
/// A session entry's identity (sessions/prefix.zig Digest).
pub const Key = [16]u8;

pub const Commit = struct { slot: u32, accepted: u32, bonus: u32 };
pub const Finish = struct { slot: u32, save: bool, cancelled: bool };
pub const Piece = struct { slot: u32, start: u64, end: u64 };

/// A request's keyed sampling (lookup.noise_of): rank 1 rebuilds the same draws.
pub const Noise = struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 };
/// Top-p rows taken from the candidates.
pub const Nucleus = struct { temperature: f64, top_p: f64, min_p: f64 };

pub const Admit = struct {
    slot: u32,
    prompt: []const u32,
    max_tokens: u32,
    tier: Tier = .none,
    key: Key = @splat(0),
    cached: u32 = 0,
    quota: u32 = 0,
    /// the request's packed grammar (rank 1 compiles it), empty for none
    grammar: []const i64 = &.{},
    job: u64 = 0,
};

pub const Window = struct {
    slot: u32,
    start: u64,
    pending: u32,
    /// DSpark depth (both ranks draft); 0: `drafts` are the host's (lookup)
    depth: u32 = 0,
    drafts: []const u32 = &.{},
    noise: ?Noise = null,
    nucleus: ?Nucleus = null,
};

pub const Plan = struct {
    round: u64 = 0,
    mode: Mode = .full,
    /// candidates a verify row
    count: u32 = 1,
    commits: []const Commit = &.{},
    finishes: []const Finish = &.{},
    spills: []const Key = &.{},
    admits: []const Admit = &.{},
    pieces: []const Piece = &.{},
    saves: []const u32 = &.{},
    finals: []const u32 = &.{},
    windows: []const Window = &.{},
    stop: bool = false,
    /// the allocator's cached blocks back to the system before the pieces (both ranks)
    release: bool = false,

    pub fn empty(p: *const Plan) bool {
        return p.commits.len == 0 and p.finishes.len == 0 and p.spills.len == 0 and p.admits.len == 0 and p.pieces.len == 0 and
            p.saves.len == 0 and p.finals.len == 0 and p.windows.len == 0 and !p.stop;
    }

    /// The plan as int64 words, appended to `out`.
    pub fn encode(p: *const Plan, gpa: std.mem.Allocator, out: *std.ArrayList(i64)) !void {
        var w: Writer = .{ .gpa = gpa, .out = out };
        try w.all(&.{ magic, version, @bitCast(p.round), @backingInt(p.mode), p.count, @as(i64, @intFromBool(p.stop)) | @as(i64, @intFromBool(p.release)) << 1 });
        try w.int(p.commits.len);
        for (p.commits) |c| try w.all(&.{ c.slot, c.accepted, c.bonus });
        try w.int(p.finishes.len);
        for (p.finishes) |f| try w.all(&.{ f.slot, @intFromBool(f.save), @intFromBool(f.cancelled) });
        try w.int(p.spills.len);
        for (p.spills) |k| try w.key(k);
        try w.int(p.admits.len);
        for (p.admits) |a| {
            try w.all(&.{ a.slot, a.max_tokens, @backingInt(a.tier), a.cached, a.quota, @bitCast(a.job) });
            try w.key(a.key);
            try w.ids(a.prompt);
            try w.int(a.grammar.len);
            try w.out.appendSlice(gpa, a.grammar);
        }
        try w.int(p.pieces.len);
        for (p.pieces) |x| try w.all(&.{ x.slot, @bitCast(x.start), @bitCast(x.end) });
        try w.ids(p.saves);
        try w.ids(p.finals);
        try w.int(p.windows.len);
        for (p.windows) |x| {
            try w.all(&.{ x.slot, @bitCast(x.start), x.pending, x.depth });
            try w.ids(x.drafts);
            if (x.noise) |n| {
                try w.all(&.{ 1, @bitCast(n.seed), @bitCast(n.temperature), n.top_k, @bitCast(n.top_p), @bitCast(n.min_p) });
            } else try w.int(0);
            if (x.nucleus) |n| {
                try w.all(&.{ 1, @bitCast(n.temperature), @bitCast(n.top_p), @bitCast(n.min_p) });
            } else try w.int(0);
        }
        try w.int(magic); // the end mark: a frame cut short or overrun cannot decode
    }

    /// A plan from `words`; its slices live in `arena` (freed with it).
    pub fn decode(arena: std.mem.Allocator, words: []const i64) !Plan {
        var r: Reader = .{ .a = arena, .w = words };
        if (try r.next() != magic or try r.next() != version) return error.NotAPlan;
        var p: Plan = .{};
        p.round = @bitCast(try r.next());
        p.mode = std.enums.fromInt(Mode, try r.next()) orelse return error.BadPlan;
        p.count = try r.u32_();
        const flags = try r.next();
        if (flags & ~@as(i64, 3) != 0) return error.BadPlan;
        p.stop = flags & 1 != 0;
        p.release = flags & 2 != 0;
        const commits = try arena.alloc(Commit, try r.len(3));
        for (commits) |*c| c.* = .{ .slot = try r.u32_(), .accepted = try r.u32_(), .bonus = try r.u32_() };
        p.commits = commits;
        const finishes = try arena.alloc(Finish, try r.len(3));
        for (finishes) |*f| f.* = .{ .slot = try r.u32_(), .save = try r.flag(), .cancelled = try r.flag() };
        p.finishes = finishes;
        const spills = try arena.alloc(Key, try r.len(2));
        for (spills) |*k| k.* = try r.key();
        p.spills = spills;
        const admits = try arena.alloc(Admit, try r.len(10));
        for (admits) |*a| {
            a.* = .{ .slot = try r.u32_(), .max_tokens = try r.u32_(), .prompt = &.{} };
            a.tier = std.enums.fromInt(Tier, try r.next()) orelse return error.BadPlan;
            a.cached = try r.u32_();
            a.quota = try r.u32_();
            a.job = @bitCast(try r.next());
            a.key = try r.key();
            a.prompt = try r.ids();
            const g = try r.len(1);
            a.grammar = try arena.dupe(i64, try r.take(g));
        }
        p.admits = admits;
        const pieces = try arena.alloc(Piece, try r.len(3));
        for (pieces) |*x| {
            x.* = .{ .slot = try r.u32_(), .start = try r.u64_(), .end = try r.u64_() };
            if (x.end <= x.start) return error.BadPlan;
        }
        p.pieces = pieces;
        p.saves = try r.ids();
        p.finals = try r.ids();
        const windows = try arena.alloc(Window, try r.len(7));
        for (windows) |*x| {
            x.* = .{ .slot = try r.u32_(), .start = try r.u64_(), .pending = try r.u32_(), .depth = try r.u32_() };
            x.drafts = try r.ids();
            if (try r.flag()) x.noise = .{ .seed = @bitCast(try r.next()), .temperature = try r.f64_(), .top_k = try r.u32_(), .top_p = try r.f64_(), .min_p = try r.f64_() };
            if (try r.flag()) x.nucleus = .{ .temperature = try r.f64_(), .top_p = try r.f64_(), .min_p = try r.f64_() };
        }
        p.windows = windows;
        if (try r.next() != magic or r.i != words.len) return error.BadPlan;
        return p;
    }
};

const Writer = struct {
    gpa: std.mem.Allocator,
    out: *std.ArrayList(i64),

    fn int(w: *Writer, v: anytype) !void {
        try w.out.append(w.gpa, @intCast(v));
    }

    fn all(w: *Writer, vs: []const i64) !void {
        try w.out.appendSlice(w.gpa, vs);
    }

    fn key(w: *Writer, k: Key) !void {
        try w.all(&.{ @bitCast(std.mem.readInt(u64, k[0..8], .little)), @bitCast(std.mem.readInt(u64, k[8..16], .little)) });
    }

    /// A length, then the ids.
    fn ids(w: *Writer, v: []const u32) !void {
        try w.int(v.len);
        try w.out.ensureUnusedCapacity(w.gpa, v.len);
        for (v) |x| w.out.appendAssumeCapacity(x);
    }
};

const Reader = struct {
    a: std.mem.Allocator,
    w: []const i64,
    i: usize = 0,

    fn next(r: *Reader) !i64 {
        if (r.i >= r.w.len) return error.TruncatedPlan;
        r.i += 1;
        return r.w[r.i - 1];
    }

    fn take(r: *Reader, n: usize) ![]const i64 {
        if (n > r.w.len - r.i) return error.TruncatedPlan;
        r.i += n;
        return r.w[r.i - n .. r.i];
    }

    fn u32_(r: *Reader) !u32 {
        return std.math.cast(u32, try r.next()) orelse error.BadPlan;
    }

    fn u64_(r: *Reader) !u64 {
        return std.math.cast(u64, try r.next()) orelse error.BadPlan;
    }

    fn f64_(r: *Reader) !f64 {
        return @bitCast(try r.next());
    }

    fn flag(r: *Reader) !bool {
        const v = try r.next();
        if (v != 0 and v != 1) return error.BadPlan;
        return v == 1;
    }

    /// A count of records of at least `words` words each, checked against what is left (a garbage count fails here).
    fn len(r: *Reader, words: usize) !usize {
        const n = try r.u64_();
        if (n > (r.w.len - r.i) / words) return error.TruncatedPlan;
        return @intCast(n);
    }

    fn key(r: *Reader) !Key {
        var k: Key = undefined;
        std.mem.writeInt(u64, k[0..8], @bitCast(try r.next()), .little);
        std.mem.writeInt(u64, k[8..16], @bitCast(try r.next()), .little);
        return k;
    }

    fn ids(r: *Reader) ![]const u32 {
        const raw = try r.take(try r.len(1));
        const out = try r.a.alloc(u32, raw.len);
        for (raw, out) |x, *y| y.* = std.math.cast(u32, x) orelse return error.BadPlan;
        return out;
    }
};

/// The end of a slot's next prefill piece toward `target` within `budget` rows, on the grid, never empty.
pub fn pieceEnd(done: u64, target: u64, budget: u64, grid: u64) u64 {
    if (target - done <= budget) return target;
    var end = (done + budget) / grid * grid;
    if (end <= done) end = @min(target, (done / grid + 1) * grid);
    return end;
}

const testing = std.testing;

test "a round plan round-trips: every section, keyed sampling, nucleus rows, the stop flag" {
    const a = testing.allocator;
    var key: Key = undefined;
    for (&key, 0..) |*b, i| b.* = @truncate(i * 17 + 3);
    const plan: Plan = .{
        .round = 1 << 40,
        .mode = .replay,
        .count = 64,
        .commits = &.{ .{ .slot = 0, .accepted = 3, .bonus = 129 }, .{ .slot = 2, .accepted = 0, .bonus = 7 } },
        .finishes = &.{.{ .slot = 1, .save = true, .cancelled = false }},
        .spills = &.{key},
        .admits = &.{.{ .slot = 3, .prompt = &.{ 0, 128799, 5 }, .max_tokens = 32768, .tier = .disk, .key = key, .cached = 2048, .quota = 9, .grammar = &.{ -1, 1 << 50 }, .job = 77 }},
        .pieces = &.{.{ .slot = 3, .start = 2048, .end = 4096 }},
        .saves = &.{3},
        .finals = &.{ 3, 0 },
        .windows = &.{
            .{ .slot = 0, .start = 1_048_000, .pending = 42, .depth = 5 },
            .{ .slot = 2, .start = 9, .pending = 1, .drafts = &.{ 4, 5, 6 }, .noise = .{ .seed = std.math.maxInt(u64), .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0.05 }, .nucleus = .{ .temperature = 0.7, .top_p = 0.95, .min_p = 0.0 } },
        },
        .stop = true,
        .release = true,
    };
    var words: std.ArrayList(i64) = .empty;
    defer words.deinit(a);
    try plan.encode(a, &words);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const back = try Plan.decode(arena.allocator(), words.items);
    try testing.expectEqual(plan.round, back.round);
    try testing.expectEqual(Mode.replay, back.mode);
    try testing.expect(back.stop and back.release and !back.empty());
    try testing.expectEqualSlices(Commit, plan.commits, back.commits);
    try testing.expectEqualSlices(Finish, plan.finishes, back.finishes);
    try testing.expectEqualSlices(u8, &key, &back.spills[0]);
    const ad = back.admits[0];
    try testing.expectEqualSlices(u32, plan.admits[0].prompt, ad.prompt);
    try testing.expectEqualSlices(i64, plan.admits[0].grammar, ad.grammar);
    try testing.expect(ad.tier == .disk and ad.cached == 2048 and ad.job == 77 and std.mem.eql(u8, &ad.key, &key));
    try testing.expectEqualSlices(Piece, plan.pieces, back.pieces);
    try testing.expectEqualSlices(u32, plan.finals, back.finals);
    try testing.expectEqual(@as(?Noise, null), back.windows[0].noise);
    try testing.expectEqual(plan.windows[1].noise, back.windows[1].noise);
    try testing.expectEqual(plan.windows[1].nucleus, back.windows[1].nucleus);
    try testing.expectEqualSlices(u32, plan.windows[1].drafts, back.windows[1].drafts);
}

test "a corrupted frame is refused, never half decoded" {
    const a = testing.allocator;
    const plan: Plan = .{ .windows = &.{.{ .slot = 0, .start = 5, .pending = 9, .drafts = &.{ 1, 2 } }} };
    var words: std.ArrayList(i64) = .empty;
    defer words.deinit(a);
    try plan.encode(a, &words);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    try testing.expect((try Plan.decode(ar, words.items)).empty() == false);
    // cut short at every length, one word too many, a wrong magic, a count far past the frame, a negative id
    for (0..words.items.len) |n| try testing.expect(std.meta.isError(Plan.decode(ar, words.items[0..n])));
    try words.append(a, 0);
    try testing.expectError(error.BadPlan, Plan.decode(ar, words.items));
    _ = words.pop();
    words.items[0] = 7;
    try testing.expectError(error.NotAPlan, Plan.decode(ar, words.items));
    words.items[0] = magic;
    const windows_at = 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1;
    words.items[windows_at] = 1 << 40;
    try testing.expectError(error.TruncatedPlan, Plan.decode(ar, words.items));
    words.items[windows_at] = 1;
    words.items[windows_at + 6] = -1;
    try testing.expectError(error.BadPlan, Plan.decode(ar, words.items));
}

test "piece ends on the prefill grid (plan.piece_end)" {
    try testing.expectEqual(@as(u64, 3000), pieceEnd(1000, 3000, 2048, 16));
    try testing.expectEqual(@as(u64, 3040), pieceEnd(1000, 9000, 2048, 16));
    try testing.expectEqual(@as(u64, 16), pieceEnd(0, 100, 8, 16));
    try testing.expectEqual(@as(u64, 10), pieceEnd(0, 10, 4, 16));
}
