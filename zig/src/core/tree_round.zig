//! Deterministic BF16 picks and verified-tree result decoding.
const std = @import("std");
pub const max_rows = 16;
pub const invalid_token = std.math.maxInt(u32);
pub const Pick = extern struct { token: u32 = invalid_token, nonfinite: u32 = 0 };
pub const Argmax = extern struct { rows: u32, vocab: u32, stride: u32 };
pub const Match = extern struct { rows: u32, vocab: u32, budget: u32, eos_count: u32 = 0 };
pub const Status = enum(u32) { ok, nonfinite, bad_tree, bad_pick };
pub const Stop = enum(u32) { bonus, length, eos, invalid };
pub const Result = extern struct {
    status: u32 = 0,
    stop: u32 = 0,
    nonfinite: u32 = 0,
    consumed_count: u32 = 0,
    emitted_count: u32 = 0,
    matched_count: u32 = 0,
    bonus_emitted: u32 = 0,
    pending_valid: u32 = 0,
    pending_token: u32 = 0,
    reserved: u32 = 0,
    path: [max_rows]u32 = @splat(0),
    tokens: [max_rows]u32 = @splat(0),
};
pub const Decoded = struct {
    status: Status,
    stop: Stop,
    nonfinite: u32,
    path: []const u32,
    tokens: []const u32,
    matched: u32,
    bonus: ?u32,
    pending: ?u32,
};
pub fn checkArgmax(p: Argmax) !void {
    if (p.rows == 0 or p.rows > max_rows or p.vocab == 0 or p.stride < p.vocab) return error.BadRoundShape;
}
pub fn checkMatch(p: Match) !void {
    if (p.rows == 0 or p.rows > max_rows or p.vocab == 0 or p.eos_count > max_rows) return error.BadRoundShape;
}
pub fn rowArgmax(words: []const u16) Pick {
    var pick = Pick{};
    var best: f32 = -std.math.inf(f32);
    for (words, 0..) |bits, id| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) {
            pick.nonfinite |= if (std.math.isNan(value)) @as(u32, 1) else if (value > 0) @as(u32, 2) else @as(u32, 4);
        } else if (pick.token == invalid_token or value > best) {
            best = value;
            pick.token = @intCast(id);
        }
    }
    if (pick.nonfinite != 0) pick.token = invalid_token;
    return pick;
}
fn invalid(status: Status, mask: u32) Result {
    return .{ .status = @backingInt(status), .stop = @backingInt(Stop.invalid), .nonfinite = mask };
}
pub fn oracle(p: Match, tokens: []const u32, parents: []const i32, picks: []const Pick, eos: []const u32) !Result {
    try checkMatch(p);
    if (tokens.len != p.rows or parents.len != p.rows or picks.len != p.rows or eos.len != p.eos_count) return error.BadRoundShape;
    for (tokens, parents, 0..) |token, parent, row| {
        if (token >= p.vocab or (if (row == 0) parent != -1 else parent < 0 or parent >= row)) return invalid(.bad_tree, 0);
    }
    for (eos) |token| if (token >= p.vocab) return invalid(.bad_tree, 0);
    var mask: u32 = 0;
    var bad_pick = false;
    for (picks) |pick| {
        mask |= pick.nonfinite & 7;
        bad_pick = bad_pick or (pick.nonfinite & ~@as(u32, 7) != 0) or (pick.nonfinite == 0 and pick.token >= p.vocab);
    }
    if (bad_pick) return invalid(.bad_pick, mask);
    if (mask != 0) return invalid(.nonfinite, mask);
    var path: [max_rows]u32 = @splat(0);
    var count: usize = 1;
    while (count < p.rows) {
        const node = path[count - 1];
        var child: ?u32 = null;
        for (tokens, parents, 0..) |token, parent, row| {
            if (parent == node and token == picks[node].token) {
                child = @intCast(row);
                break;
            }
        }
        path[count] = child orelse break;
        count += 1;
    }
    var emitted: [max_rows]u32 = @splat(0);
    for (1..count) |i| emitted[i - 1] = tokens[path[i]];
    emitted[count - 1] = picks[path[count - 1]].token;
    var take: usize = @min(count, p.budget);
    var stop: Stop = if (take == p.budget) .length else .bonus;
    for (emitted[0..take], 0..) |token, i| {
        if (std.mem.indexOfScalar(u32, eos, token) != null) {
            take = i + 1;
            stop = .eos;
            break;
        }
    }
    var result = Result{ .stop = @backingInt(stop), .consumed_count = @intCast(take), .emitted_count = @intCast(take), .matched_count = @intCast(@min(take, count - 1)) };
    @memcpy(result.path[0..take], path[0..take]);
    @memcpy(result.tokens[0..take], emitted[0..take]);
    result.bonus_emitted = @intFromBool(take > 0 and take == count);
    result.pending_valid = @intFromBool(stop == .bonus);
    if (result.pending_valid != 0) result.pending_token = emitted[take - 1];
    return result;
}
pub fn decode(result: *const Result, p: Match) !Decoded {
    try checkMatch(p);
    const status = std.enums.fromInt(Status, result.status) orelse return error.BadRoundResult;
    const stop = std.enums.fromInt(Stop, result.stop) orelse return error.BadRoundResult;
    const n = result.emitted_count;
    if (result.reserved != 0 or result.nonfinite > 7 or n > max_rows or n > p.budget or result.consumed_count != n or result.consumed_count > p.rows or result.matched_count > n or result.bonus_emitted > 1 or result.pending_valid > 1) return error.BadRoundResult;
    if (status != .ok) {
        if (stop != .invalid or n != 0 or result.matched_count != 0 or result.bonus_emitted != 0 or result.pending_valid != 0 or result.pending_token != 0) return error.BadRoundResult;
        if ((status == .nonfinite) != (result.nonfinite != 0) and status != .bad_pick) return error.BadRoundResult;
    } else {
        if (stop == .invalid or result.nonfinite != 0 or result.bonus_emitted != n - result.matched_count) return error.BadRoundResult;
        if (stop == .length and n != p.budget) return error.BadRoundResult;
        if (stop == .eos and n == 0) return error.BadRoundResult;
        if ((stop == .bonus) != (result.pending_valid == 1)) return error.BadRoundResult;
        if (stop == .bonus and (result.bonus_emitted != 1 or n == 0)) return error.BadRoundResult;
        if (result.pending_valid == 1 and result.pending_token != result.tokens[n - 1]) return error.BadRoundResult;
        if (result.pending_valid == 0 and result.pending_token != 0) return error.BadRoundResult;
        if (n > 0 and result.path[0] != 0) return error.BadRoundResult;
        for (result.path[0..n], result.tokens[0..n], 0..) |row, token, i| {
            if (row >= p.rows or token >= p.vocab or (i > 0 and row <= result.path[i - 1])) return error.BadRoundResult;
        }
    }
    for (result.path[n..]) |row| if (row != 0) return error.BadRoundResult;
    for (result.tokens[n..]) |token| if (token != 0) return error.BadRoundResult;
    return .{ .status = status, .stop = stop, .nonfinite = result.nonfinite, .path = result.path[0..n], .tokens = result.tokens[0..n], .matched = result.matched_count, .bonus = if (result.bonus_emitted == 1) result.tokens[n - 1] else null, .pending = if (result.pending_valid == 1) result.pending_token else null };
}
test "terminal matched row stays unconsumed and malformed result is rejected" {
    const p = Match{ .rows = 3, .vocab = 10, .budget = 2 };
    var r = try oracle(p, &.{ 1, 2, 3 }, &.{ -1, 0, 1 }, &.{ .{ .token = 2 }, .{ .token = 3 }, .{ .token = 4 } }, &.{});
    const d = try decode(&r, p);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, d.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, d.tokens);
    try std.testing.expect(d.pending == null and d.bonus == null);
    r.consumed_count = 3;
    try std.testing.expectError(error.BadRoundResult, decode(&r, p));
}
test "finite ties choose first token and nonfinite kinds invalidate a row" {
    try std.testing.expectEqual(Pick{ .token = 0 }, rowArgmax(&.{ 0x8000, 0, 0x8000 }));
    try std.testing.expectEqual(Pick{ .token = 1 }, rowArgmax(&.{ 0xbf80, 0x3f80, 0x3f80 }));
    try std.testing.expectEqual(Pick{ .token = invalid_token, .nonfinite = 7 }, rowArgmax(&.{ 0x7fc1, 0x7f80, 0xff80, 0x3f80 }));
    try std.testing.expectError(error.BadRoundShape, checkArgmax(.{ .rows = 17, .vocab = 1, .stride = 1 }));
    try std.testing.expectError(error.BadRoundShape, checkArgmax(.{ .rows = 1, .vocab = 2, .stride = 1 }));
    try std.testing.expectError(error.BadRoundShape, checkMatch(.{ .rows = 0, .vocab = 1, .budget = 1 }));
}
test "bonus EOS and zero budget expose no continuation" {
    const tokens: []const u32 = &.{ 1, 2, 3 };
    const parents: []const i32 = &.{ -1, 0, 1 };
    const picks: []const Pick = &.{ .{ .token = 2 }, .{ .token = 3 }, .{ .token = 4 } };
    var p = Match{ .rows = 3, .vocab = 10, .budget = 20 };
    var r = try oracle(p, tokens, parents, picks, &.{});
    const d = try decode(&r, p);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, d.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4 }, d.tokens);
    try std.testing.expectEqual(@as(?u32, 4), d.pending);
    try std.testing.expectEqual(@as(?u32, 4), d.bonus);
    p.eos_count = 1;
    r = try oracle(p, tokens, parents, picks, &.{3});
    const terminal = try decode(&r, p);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, terminal.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, terminal.tokens);
    try std.testing.expect(terminal.pending == null and terminal.bonus == null);
    r = try oracle(p, tokens, parents, picks, &.{4});
    try std.testing.expectEqual(@as(?u32, 4), (try decode(&r, p)).bonus);
    try std.testing.expect((try decode(&r, p)).pending == null);
    p.budget = 0;
    r = try oracle(p, tokens, parents, picks, &.{4});
    try std.testing.expectEqual(@as(usize, 0), (try decode(&r, p)).path.len);
    try std.testing.expectEqual(@as(usize, 0), (try decode(&r, p)).tokens.len);
}
test "decoder rejects malformed flags counts tokens and unused storage" {
    const p = Match{ .rows = 1, .vocab = 10, .budget = 20 };
    const good = try oracle(p, &.{1}, &.{-1}, &.{.{ .token = 2 }}, &.{});
    var bad: [12]Result = @splat(good);
    bad[0].status = 99;
    bad[1].stop = 99;
    bad[2].nonfinite = 1;
    bad[3].emitted_count = 17;
    bad[4].matched_count = 2;
    bad[5].bonus_emitted = 0;
    bad[6].pending_valid = 0;
    bad[7].pending_token = 3;
    bad[8].path[0] = 1;
    bad[9].tokens[0] = 10;
    bad[10].path[1] = 1;
    bad[11].tokens[1] = 1;
    for (&bad) |*r| try std.testing.expectError(error.BadRoundResult, decode(r, p));
}
