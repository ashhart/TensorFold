//! TF_Q27_MTP_GPU_PICK: the MTP head's draft pick from a GPU top-1 row, and its chance bit-equal to the CPU pick's.
const std = @import("std");
const topk = @import("bf16_topk");

/// A logits row's pick and its probability under the head's softmax.
pub const Pick = struct { token: u32, chance: f32 };

/// Rows one top-k dispatch takes: a wider pass keeps the CPU pick.
pub const max_rows = topk.max_rows;

/// The top-1 launch for `rows` logits rows of `vocab` values each.
pub fn params(rows: u32, vocab: u32) topk.Params {
    return .{ .rows = rows, .vocab = vocab, .k = 1, .stride = vocab };
}

/// Each BF16 value's softmax term against one top value, computed once: the sum's terms and order stay the CPU pick's.
pub const Memo = struct {
    terms: []f64,
    marks: []u32,
    stamp: u32 = 0,
    top: ?u16 = null,

    pub fn init(gpa: std.mem.Allocator) !Memo {
        const terms = try gpa.alloc(f64, 1 << 16);
        errdefer gpa.free(terms);
        const marks = try gpa.alloc(u32, 1 << 16);
        @memset(marks, 0);
        return .{ .terms = terms, .marks = marks };
    }

    pub fn deinit(m: *Memo, gpa: std.mem.Allocator) void {
        gpa.free(m.terms);
        gpa.free(m.marks);
    }

    fn aim(m: *Memo, top: u16) void {
        if (m.top == top) return;
        m.top = top;
        if (m.stamp == std.math.maxInt(u32)) {
            @memset(m.marks, 0);
            m.stamp = 0;
        }
        m.stamp += 1;
    }

    /// One over the sum of exp(v - top), every value in row order, as the CPU pick adds them.
    pub fn chance(m: *Memo, values: []const u16, top: u16) f32 {
        m.aim(top);
        const high: f32 = @bitCast(@as(u32, top) << 16);
        var sum: f64 = 0;
        for (values) |v| {
            if (m.marks[v] != m.stamp) {
                m.marks[v] = m.stamp;
                m.terms[v] = @exp(@as(f64, @as(f32, @bitCast(@as(u32, v) << 16)) - high));
            }
            sum += m.terms[v];
        }
        return @floatCast(1 / sum);
    }
};

/// The pick from a logits row's top-1 result; a nonfinite row fails as the CPU pick does.
pub fn fromRow(row: *const topk.Row, values: []const u16, memo: *Memo) !Pick {
    const p = params(1, std.math.cast(u32, values.len) orelse return error.BadLogits);
    const entries = topk.decode(row, p) catch |err| return if (err == error.NonfiniteTopK) error.NonfiniteLogits else err;
    const top: u16 = @intCast(@as(u32, @bitCast(entries[0].score)) >> 16);
    if (values[entries[0].token] != top) return error.BadTopKResult;
    return .{ .token = entries[0].token, .chance = memo.chance(values, top) };
}

/// One fused pass's order: rows in, the head, picks and saves, then a single commit.
pub fn fused(ctx: anytype, Out: type) !Out {
    try ctx.before();
    const out = try ctx.head();
    try ctx.after(out);
    try ctx.commit();
    return out;
}

const Recorder = struct {
    log: [8]u8 = undefined,
    n: usize = 0,
    fail: u8 = 0,

    fn note(r: *Recorder, c: u8) !void {
        if (r.fail == c) return error.Injected;
        r.log[r.n] = c;
        r.n += 1;
    }
    fn before(r: *Recorder) !void {
        try r.note('g');
    }
    fn head(r: *Recorder) !u32 {
        try r.note('h');
        return 7;
    }
    fn after(r: *Recorder, out: u32) !void {
        try std.testing.expectEqual(@as(u32, 7), out);
        try r.note('p');
    }
    fn commit(r: *Recorder) !void {
        try r.note('c');
    }
};

test "qwen27 MTP pick: a fused absorb gathers, runs the head, picks and saves, then commits once" {
    var r = Recorder{};
    try std.testing.expectEqual(@as(u32, 7), try fused(&r, u32));
    try std.testing.expectEqualStrings("ghpc", r.log[0..r.n]);
    for ("ghp") |at| {
        var bad = Recorder{ .fail = at };
        try std.testing.expectError(error.Injected, fused(&bad, u32));
        try std.testing.expect(std.mem.indexOfScalar(u8, bad.log[0..bad.n], 'c') == null);
    }
}

fn expectSame(values: []const u16, memo: *Memo) !void {
    const cpu = try @import("mtp.zig").pick(values);
    const row = try topk.oracle(values, 1);
    const gpu = try fromRow(&row, values, memo);
    try std.testing.expectEqual(cpu.token, gpu.token);
    try std.testing.expectEqual(@as(u32, @bitCast(cpu.chance)), @as(u32, @bitCast(gpu.chance)));
}

test "qwen27 MTP pick: the top-1 contract picks the CPU's token and chance, ties and signed zeros included" {
    const a = std.testing.allocator;
    var memo = try Memo.init(a);
    defer memo.deinit(a);
    const cases = [_][]const u16{
        &.{ 0xc000, 0x3f80, 0x3f80, 0xbf80 }, // the first of a tie
        &.{ 0x8000, 0x0000, 0x8000 }, // -0 and +0 tie, so the first wins
        &.{ 0x0000, 0x8000 },
        &.{ 0x0001, 0x0002, 0x0002 }, // subnormal values order by value
        &.{ 0x8002, 0x8001, 0x8001 },
        &.{0xbf80},
    };
    for (cases) |c| try expectSame(c, &memo);
    var random = std.Random.DefaultPrng.init(0x2716);
    const rng = random.random();
    var values: [4096]u16 = undefined;
    for (0..64) |round| {
        for (&values) |*v| {
            // few distinct values in the odd rounds, so ties at the top are common
            const raw = if (round % 2 == 1) rng.intRangeAtMost(u16, 0x3f80, 0x3f84) else rng.int(u16);
            v.* = if (raw & 0x7f80 == 0x7f80) raw & 0xbfff else raw;
        }
        try expectSame(&values, &memo);
    }
}

test "qwen27 MTP pick: a nonfinite row fails as the CPU pick does, and a result that misreads its row is refused" {
    const a = std.testing.allocator;
    var memo = try Memo.init(a);
    defer memo.deinit(a);
    for ([_][]const u16{ &.{ 0x3f80, 0x7fc1 }, &.{ 0x7f80, 0x3f80 }, &.{ 0xff80, 0x3f80 } }) |values| {
        try std.testing.expectError(error.NonfiniteLogits, @import("mtp.zig").pick(values));
        const row = try topk.oracle(values, 1);
        try std.testing.expectError(error.NonfiniteLogits, fromRow(&row, values, &memo));
    }
    const values = [_]u16{ 0x3f80, 0x4000 };
    var row = try topk.oracle(&values, 1);
    row.entries[0].token = 0;
    try std.testing.expectError(error.BadTopKResult, fromRow(&row, &values, &memo));
}

test "qwen27 MTP pick: a memo reused across tops never keeps a term from the last top" {
    const a = std.testing.allocator;
    var memo = try Memo.init(a);
    defer memo.deinit(a);
    try expectSame(&.{ 0x3f80, 0x4000 }, &memo);
    try expectSame(&.{ 0x3f80, 0x4040 }, &memo);
    try expectSame(&.{ 0x3f80, 0x4000 }, &memo);
    memo.stamp = std.math.maxInt(u32);
    try expectSame(&.{ 0x3f80, 0x4080 }, &memo);
    try std.testing.expectEqual(@as(u32, 1), memo.stamp);
}
