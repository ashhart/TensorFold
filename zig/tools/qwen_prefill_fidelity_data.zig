//! Raw-token fixtures and exact BF16 logit ranking for the developer-only Qwen fidelity probe.
const std = @import("std");

pub const ExecutionModes = struct {
    graph: bool = false,
    chunked_gdn: bool = false,
    profiling: bool = false,

    pub fn validate(m: ExecutionModes) !void {
        if (m.graph) return error.FidelityPrefillGraphUnavailable;
        if (m.chunked_gdn) return error.FidelityChunkedGdnUnavailable;
        if (m.profiling) return error.FidelityWholeEngineProfilingUnavailable;
    }
};

pub const Case = struct { id: []const u8, tokens: []const u32 };
pub const Fixture = struct { cases: []const Case, steps: usize = 16, chunk_rows: usize = 4096, kv8: bool = true };
pub const Top = struct { token: u32, logit: f32, bits: u16 };
pub const Summary = struct {
    top: [8]Top,
    top_margin: f64,
    chosen: u32,
    chosen_logit: f32,
    chosen_margin: f64,
    chosen_rank: usize,
};
fn integer(value: std.json.Value) !u32 {
    if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(u32)) return error.InvalidTokenFixture;
    return @intCast(value.integer);
}
fn tokens(a: std.mem.Allocator, value: std.json.Value) ![]const u32 {
    if (value != .array) return error.InvalidTokenFixture;
    const out = try a.alloc(u32, value.array.items.len);
    for (value.array.items, out) |v, *id| id.* = try integer(v);
    return out;
}
fn readCase(a: std.mem.Allocator, value: std.json.Value, index: usize) !Case {
    if (value != .object) return error.InvalidTokenFixture;
    const id = if (value.object.get("id")) |v| blk: {
        if (v != .string) return error.InvalidTokenFixture;
        break :blk v.string;
    } else try std.fmt.allocPrint(a, "case-{d}", .{index});
    return .{ .id = id, .tokens = try tokens(a, value.object.get("tokens") orelse return error.InvalidTokenFixture) };
}
/// The caller supplies an arena; returned strings and arrays share its lifetime.
pub fn parse(a: std.mem.Allocator, text: []const u8) !Fixture {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{ .allocate = .alloc_always });
    var out: Fixture = .{ .cases = &.{} };
    if (root == .array and (root.array.items.len == 0 or root.array.items[0] == .integer)) {
        const cases = try a.alloc(Case, 1);
        cases[0] = .{ .id = "case-0", .tokens = try tokens(a, root) };
        out.cases = cases;
    } else {
        const values = if (root == .object) root.object.get("cases") orelse root else root;
        if (values == .array) {
            const cases = try a.alloc(Case, values.array.items.len);
            for (values.array.items, cases, 0..) |v, *c, i| c.* = try readCase(a, v, i);
            out.cases = cases;
        } else if (values == .object and values.object.contains("tokens")) {
            const cases = try a.alloc(Case, 1);
            cases[0] = try readCase(a, values, 0);
            out.cases = cases;
        } else return error.InvalidTokenFixture;
        if (root == .object) {
            if (root.object.get("steps")) |v| out.steps = try integer(v);
            if (root.object.get("chunk_rows")) |v| out.chunk_rows = try integer(v);
            if (root.object.get("kv8")) |v| {
                if (v != .bool) return error.InvalidTokenFixture;
                out.kv8 = v.bool;
            }
        }
    }
    if (out.cases.len == 0 or out.cases.len > 8 or out.steps == 0 or out.steps > 64 or
        out.chunk_rows < 64 or out.chunk_rows > 8192 or out.chunk_rows % 64 != 0) return error.InvalidTokenFixture;
    for (out.cases, 0..) |c, i| {
        if (c.id.len == 0 or c.id.len > 64 or c.tokens.len == 0 or c.tokens.len > 131072) return error.InvalidTokenFixture;
        for (out.cases[0..i]) |prior| if (std.mem.eql(u8, c.id, prior.id)) return error.DuplicateFixtureId;
    }
    return out;
}
pub fn digest(a: std.mem.Allocator, prompt: []const u32, continuation: []const u32) ![]const u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u32{ prompt, continuation }) |sequence| for (sequence) |token| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, token, .little);
        h.update(&bytes);
    };
    var hash: [32]u8 = undefined;
    h.final(&hash);
    const hex = std.fmt.bytesToHex(hash, .lower);
    return a.dupe(u8, &hex);
}
/// A full baseline report or {cases:[{id,continuation}]} supplies one fixed token per recorded position.
pub fn continuations(a: std.mem.Allocator, text: []const u8, fixture: Fixture) ![]const []const u32 {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{ .allocate = .alloc_always });
    const out = try a.alloc([]const u32, fixture.cases.len);
    if (root == .array and fixture.cases.len == 1 and (root.array.items.len == 0 or root.array.items[0] == .integer)) {
        out[0] = try tokens(a, root);
    } else {
        if (root != .object) return error.InvalidContinuationFixture;
        if (root.object.get("policy")) |policy| {
            if (policy != .object) return error.InvalidContinuationFixture;
            const chunk = policy.object.get("chunk_rows") orelse return error.InvalidContinuationFixture;
            const kv8 = policy.object.get("kv8") orelse return error.InvalidContinuationFixture;
            if (try integer(chunk) != fixture.chunk_rows or kv8 != .bool or kv8.bool != fixture.kv8)
                return error.ContinuationPolicyMismatch;
        }
        const cases = root.object.get("cases") orelse return error.InvalidContinuationFixture;
        if (cases != .array or cases.array.items.len != fixture.cases.len) return error.InvalidContinuationFixture;
        for (fixture.cases, out) |expected, *target| {
            var matched: ?std.json.Value = null;
            for (cases.array.items) |c| {
                if (c != .object) return error.InvalidContinuationFixture;
                const id = c.object.get("id") orelse return error.InvalidContinuationFixture;
                if (id != .string) return error.InvalidContinuationFixture;
                if (std.mem.eql(u8, id.string, expected.id)) {
                    if (matched != null) return error.DuplicateFixtureId;
                    matched = c;
                }
            }
            const found = matched orelse return error.InvalidContinuationFixture;
            if (found.object.get("prompt_sha256")) |hash| {
                if (hash != .string or !std.mem.eql(u8, hash.string, try digest(a, expected.tokens, &.{})))
                    return error.ContinuationPromptMismatch;
            }
            target.* = try tokens(a, found.object.get("continuation") orelse return error.InvalidContinuationFixture);
        }
    }
    for (out) |sequence| if (sequence.len != fixture.steps) return error.InvalidContinuationFixture;
    return out;
}
fn decodeBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
fn ahead(a: Top, b: Top) bool {
    return a.logit > b.logit or (a.logit == b.logit and a.token < b.token);
}
pub fn summarize(logits: []const u16, forced: ?u32) !Summary {
    if (logits.len < 8 or (forced != null and forced.? >= logits.len)) return error.InvalidLogitRow;
    var top: [8]Top = undefined;
    var count: usize = 0;
    for (logits, 0..) |bits, token| {
        const logit = decodeBf16(bits);
        if (!std.math.isFinite(logit)) return error.NonfiniteLogits;
        const next: Top = .{ .token = @intCast(token), .logit = logit, .bits = bits };
        var at: usize = 0;
        while (at < count and ahead(top[at], next)) : (at += 1) {}
        if (at == 8) continue;
        const last = @min(count, 7);
        var i = last;
        while (i > at) : (i -= 1) top[i] = top[i - 1];
        top[at] = next;
        count = @min(count + 1, 8);
    }
    const chosen = forced orelse top[0].token;
    const chosen_logit = decodeBf16(logits[chosen]);
    var rank: usize = 1;
    for (logits, 0..) |bits, token| {
        const v = decodeBf16(bits);
        if (v > chosen_logit or (v == chosen_logit and token < chosen)) rank += 1;
    }
    const other = if (top[0].token == chosen) top[1].logit else top[0].logit;
    return .{ .top = top, .top_margin = @as(f64, top[0].logit) - top[1].logit, .chosen = chosen, .chosen_logit = chosen_logit, .chosen_margin = @as(f64, chosen_logit) - other, .chosen_rank = rank };
}

test "fidelity data ranks BF16 logits with deterministic ties and signed fixed-token margins" {
    const logits = [_]u16{ 0x3f80, 0x4000, 0x4000, 0xbf80, 0, 0x3f00, 0x3e80, 0x3e00, 0xc000 };
    const greedy = try summarize(&logits, null);
    try std.testing.expectEqual(@as(u32, 1), greedy.chosen);
    try std.testing.expectEqual(@as(u32, 2), greedy.top[1].token);
    try std.testing.expectEqual(@as(f64, 0), greedy.top_margin);
    const replay = try summarize(&logits, 0);
    try std.testing.expectEqual(@as(usize, 3), replay.chosen_rank);
    try std.testing.expectEqual(@as(f64, -1), replay.chosen_margin);
    var bad = logits;
    bad[0] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteLogits, summarize(&bad, null));
}
test "fidelity data parses raw IDs and enforces fixed history length and prompt identity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = try parse(a, "{\"steps\":2,\"chunk_rows\":128,\"cases\":[{\"id\":\"coding\",\"tokens\":[1,2,3]}]}");
    const fixed = try continuations(a, "{\"cases\":[{\"id\":\"coding\",\"continuation\":[4,5]}]}", fixture);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, fixed[0]);
    try std.testing.expectError(error.InvalidContinuationFixture, continuations(a, "[4]", fixture));
    try std.testing.expectError(error.ContinuationPromptMismatch, continuations(a, "{\"cases\":[{\"id\":\"coding\",\"prompt_sha256\":\"wrong\",\"continuation\":[4,5]}]}", fixture));
    try std.testing.expectError(error.ContinuationPolicyMismatch, continuations(a, "{\"policy\":{\"chunk_rows\":128,\"kv8\":false},\"cases\":[{\"id\":\"coding\",\"continuation\":[4,5]}]}", fixture));
    const raw = try parse(a, "[1,2,3]");
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, raw.cases[0].tokens);
    try std.testing.expect(!std.mem.eql(u8, try digest(a, &.{1}, &.{2}), try digest(a, &.{1}, &.{3})));
}

test "fidelity modes refuse unexecuted whole-engine hooks before a report can attest them" {
    try (ExecutionModes{}).validate();
    try std.testing.expectError(error.FidelityPrefillGraphUnavailable, (ExecutionModes{ .graph = true }).validate());
    try std.testing.expectError(error.FidelityChunkedGdnUnavailable, (ExecutionModes{ .chunked_gdn = true }).validate());
    try std.testing.expectError(error.FidelityWholeEngineProfilingUnavailable, (ExecutionModes{ .profiling = true }).validate());
    try std.testing.expectError(error.FidelityPrefillGraphUnavailable, (ExecutionModes{ .graph = true, .chunked_gdn = true, .profiling = true }).validate());
}
