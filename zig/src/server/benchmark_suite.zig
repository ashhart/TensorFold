//! The versioned community suite: fixtures, digest, statuses, error codes, the sample record and per-cell medians.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const legacy_id = "community-v1";
pub const legacy_sha256_hex = "9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5be";
pub const id = "community-v2";
pub const sha256_hex = "348fc03357c61349b22288a4262a99216b5b23ff836378aeb0284aadc34d8317";
pub const seed_base: u32 = 1234;
pub const standard_reps = 5;

/// ``tensorfold benchmark suites``: the manifest and its digest as ``json.dumps(indent=2)`` writes them.
pub const suites_text =
    \\{
    \\  "id": "community-v2",
    \\  "fixtures": [
    \\    {
    \\      "id": "code",
    \\      "kind": "completion",
    \\      "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."
    \\    },
    \\    {
    \\      "id": "chat",
    \\      "kind": "chat",
    \\      "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."
    \\    }
    \\  ],
    \\  "max_tokens": 256,
    \\  "repetitions": 5,
    \\  "warmups_per_cell": 1,
    \\  "temperatures": [
    \\    1.0,
    \\    0.0
    \\  ],
    \\  "seed_base": 1234,
    \\  "sampled_top_k": 20,
    \\  "sampled_top_p": 0.95,
    \\  "thinking": false,
    \\  "ignore_eos": false,
    \\  "concurrency": 1,
    \\  "delivery_rate": "completion_tokens_minus_one_over_first_to_last_nonempty_sse_delivery",
    \\  "min_sampled_stop_tokens": 128,
    \\  "sha256": "348fc03357c61349b22288a4262a99216b5b23ff836378aeb0284aadc34d8317"
    \\}
++ "\n";

pub const Fixture = struct { id: []const u8, chat: bool, prompt: []const u8 };

pub const fixtures = [2]Fixture{
    .{ .id = "code", .chat = false, .prompt = "Write a short Python function that computes the Fibonacci sequence and explain it." },
    .{ .id = "chat", .chat = true, .prompt = "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example." },
};

pub const Status = enum { ok, early_eos, @"error", unmeasured };

pub const ErrorCode = enum {
    http_error,
    connection_error,
    timeout,
    cancelled,
    server_error,
    invalid_stream,
    truncated_stream,
    missing_usage,
    invalid_usage,
    empty_output,
    unexpected_finish,
    token_count_mismatch,
    unmeasured_delivery,
};

/// One attempt as the receipt holds it; nulls are written as null, never left out.
pub const Sample = struct {
    fixture_id: []const u8,
    temperature: f64,
    repeat: i32,
    seed: u32,
    status: Status = .@"error",
    prompt_tokens: ?u64 = null,
    completion_tokens: ?u64 = null,
    cached_tokens: ?u64 = null,
    delivery_seconds: ?f64 = null,
    delivery_tps: ?f64 = null,
    ttft_seconds: ?f64 = null,
    end_to_end_seconds: ?f64 = null,
    server_decode_tps: ?f64 = null,
    server_decode_seconds: ?f64 = null,
    prefill_seconds: ?f64 = null,
    token_sha: ?[]const u8 = null,
    error_code: ?ErrorCode = null,
    finish_reason: ?[]const u8 = null,
};

/// A cell's medians are set only when every declared repeat is ok.
pub const Cell = struct {
    fixture_id: []const u8,
    temperature: f64,
    complete: bool,
    protocol_complete: bool,
    protocol: []const u8 = id,
    delivery: ?f64,
    ttft: ?f64,
};

/// The cells of receipt samples in first-seen order, with medians only for complete cells.
pub fn cells(a: Allocator, samples: []const json.Value, expected_reps: i64) ![]Cell {
    var out: std.ArrayList(Cell) = .empty;
    var rows: std.ArrayList([]const json.Value) = .empty;
    for (samples) |s| {
        const f = json.strOr(s.get("fixture_id"));
        const t = number(s.get("temperature")) orelse 0;
        const at = for (out.items, 0..) |c, i| {
            if (std.mem.eql(u8, c.fixture_id, f) and c.temperature == t) break i;
        } else blk: {
            try out.append(a, .{ .fixture_id = f, .temperature = t, .complete = false, .protocol_complete = false, .delivery = null, .ttft = null });
            try rows.append(a, &.{});
            break :blk out.items.len - 1;
        };
        rows.items[at] = try std.mem.concat(a, json.Value, &.{ rows.items[at], &.{s} });
    }
    for (out.items, rows.items) |*cell, group| {
        cell.complete = completeCell(group, expected_reps);
        cell.protocol_complete = cell.complete and expected_reps == standard_reps;
        cell.delivery = try median(a, group, "delivery_tps", cell.complete);
        cell.ttft = try median(a, group, "ttft_seconds", cell.complete);
    }
    return out.items;
}

fn completeCell(group: []const json.Value, expected: i64) bool {
    if (group.len != expected) return false;
    var seen: [32]bool = @splat(false);
    for (group) |s| {
        const r = (s.get("repeat") orelse return false).int64() orelse return false;
        if (r < 0 or r >= expected or r >= seen.len or seen[@intCast(r)]) return false;
        seen[@intCast(r)] = true;
        if (!std.mem.eql(u8, json.strOr(s.get("status")), "ok")) return false;
    }
    return true;
}

fn median(a: Allocator, group: []const json.Value, key: []const u8, complete: bool) !?f64 {
    if (!complete) return null;
    var values: std.ArrayList(f64) = .empty;
    for (group) |s| {
        const r = (s.get("repeat") orelse continue).int64() orelse continue;
        if (r < 0) continue;
        const v = number(s.get(key)) orelse return null;
        if (!std.math.isFinite(v) or v < 0) return null;
        try values.append(a, v);
    }
    if (values.items.len == 0) return null;
    std.mem.sort(f64, values.items, {}, std.sort.asc(f64));
    const n = values.items.len;
    return if (n % 2 == 1) values.items[n / 2] else (values.items[n / 2 - 1] + values.items[n / 2]) / 2;
}

/// A JSON number (bools excluded), as Python's ``isinstance(v, (int, float))`` sees it.
pub fn number(v: ?json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .int, .float => x.float64(),
        else => null,
    };
}

/// Python's ``format(x, "g")``: six significant digits, trailing zeros dropped.
pub fn fmtG(buf: []u8, x: f64) []const u8 {
    if (x == 0) return if (std.math.signbit(x)) "-0" else "0";
    if (!std.math.isFinite(x)) return if (std.math.isNan(x)) "nan" else if (x > 0) "inf" else "-inf";
    var tmp: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&tmp, "{e:.5}", .{x}) catch return "?";
    const e_at = std.mem.indexOfScalar(u8, sci, 'e') orelse return "?";
    const exp = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch return "?";
    if (exp < -4 or exp >= 6) {
        const mantissa = trimZeros(sci[0..e_at]);
        const mag: u32 = @abs(exp);
        return std.fmt.bufPrint(buf, "{s}e{c}{d:0>2}", .{ mantissa, @as(u8, if (exp < 0) '-' else '+'), mag }) catch "?";
    }
    const decimals: usize = @intCast(5 - exp);
    const fixed = std.fmt.bufPrint(buf, "{d:.[1]}", .{ x, decimals }) catch return "?";
    return trimZeros(fixed);
}

fn trimZeros(s: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, s, '.') == null) return s;
    const t = std.mem.trimEnd(u8, s, "0");
    return if (t.len > 0 and t[t.len - 1] == '.') t[0 .. t.len - 1] else t;
}

/// Every object's keys in byte order, as ``sort_keys=True`` writes them.
pub fn sortKeys(v: json.Value) void {
    switch (v) {
        .object => |o| {
            const Ctx = struct {
                keys: []const []const u8,
                pub fn lessThan(c: @This(), x: usize, y: usize) bool {
                    return std.mem.lessThan(u8, c.keys[x], c.keys[y]);
                }
            };
            o.sort(Ctx{ .keys = o.keys() });
            for (o.values()) |item| sortKeys(item);
        },
        .array => |items| for (items) |item| sortKeys(item),
        else => {},
    }
}

/// The manifest digest: sorted keys, compact separators, non-ASCII kept.
fn manifestDigest(a: Allocator, manifest: json.Value) ![64]u8 {
    sortKeys(manifest);
    const bytes = try json.stringify(a, manifest, .{ .compact = true, .ascii = false });
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "the suites text re-derives the pinned manifest digest and its own sha256 field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = (try json.parse(a, suites_text)).ok;
    try std.testing.expectEqualStrings(sha256_hex, v.get("sha256").?.string);
    _ = v.object.orderedRemove("sha256");
    sortKeys(v);
    try std.testing.expectEqualStrings(sha256_hex, &try manifestDigest(a, v));
    for (fixtures, v.get("fixtures").?.array) |f, m| {
        try std.testing.expectEqualStrings(f.id, m.get("id").?.string);
        try std.testing.expectEqualStrings(f.prompt, m.get("prompt").?.string);
        try std.testing.expectEqual(f.chat, std.mem.eql(u8, m.get("kind").?.string, "chat"));
    }
}

test "negative control: integer temperatures give another manifest digest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = (try json.parse(a, suites_text)).ok;
    _ = v.object.orderedRemove("sha256");
    const temps = v.get("temperatures").?.array;
    temps[0] = try json.intValue(a, 1);
    temps[1] = try json.intValue(a, 0);
    try std.testing.expect(!std.mem.eql(u8, sha256_hex, &try manifestDigest(a, v)));
}

test "cells need every repeat ok, and a kept failed warm-up makes a cell incomplete" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "[{\"fixture_id\": \"code\", \"temperature\": 0.0, \"repeat\": -1, \"status\": \"error\", \"delivery_tps\": null, \"ttft_seconds\": null}");
    for (0..5) |r| try text.print(a, ", {{\"fixture_id\": \"code\", \"temperature\": 0.0, \"repeat\": {d}, \"status\": \"ok\", \"delivery_tps\": {d}.0, \"ttft_seconds\": 0.5}}", .{ r, r + 1 });
    for (0..5) |r| try text.print(a, ", {{\"fixture_id\": \"chat\", \"temperature\": 0, \"repeat\": {d}, \"status\": \"ok\", \"delivery_tps\": {d}.0, \"ttft_seconds\": 0.25}}", .{ r, r + 1 });
    try text.append(a, ']');
    const samples = (try json.parse(a, text.items)).ok.array;
    const got = try cells(a, samples, 5);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expect(!got[0].complete and got[0].delivery == null);
    try std.testing.expect(got[1].protocol_complete);
    try std.testing.expectEqual(@as(f64, 3), got[1].delivery.?);
    try std.testing.expectEqual(@as(f64, 0.25), got[1].ttft.?);
    const custom = try cells(a, samples[6..7], 1);
    try std.testing.expect(custom[0].complete and !custom[0].protocol_complete and custom[0].delivery.? == 1);
}

test "temperatures print as Python's g format" {
    var buf: [32]u8 = undefined;
    for ([_]struct { f64, []const u8 }{ .{ 1.0, "1" }, .{ 0.0, "0" }, .{ 0.95, "0.95" }, .{ 1.2345678, "1.23457" }, .{ 0.00001, "1e-05" }, .{ 2, "2" } }) |c|
        try std.testing.expectEqualStrings(c[1], fmtG(&buf, c[0]));
}

/// A ranking partition is a recognized manifest, never a label supplied without its digest.
pub fn protocol(v: json.Value) ?[]const u8 {
    const sid = json.strOr(v.get("id"));
    const digest = json.strOr(v.get("sha256"));
    if (std.mem.eql(u8, sid, id) and std.mem.eql(u8, digest, sha256_hex)) return id;
    if (std.mem.eql(u8, sid, legacy_id) and std.mem.eql(u8, digest, legacy_sha256_hex)) return legacy_id;
    return null;
}

/// Cells retain their receipt's protocol so callers cannot present old results as new ones.
pub fn receiptCells(a: Allocator, v: json.Value) ![]Cell {
    const sid = protocol(v.get("suite") orelse return error.UnknownProtocol) orelse return error.UnknownProtocol;
    const stg = v.get("settings") orelse return error.UnknownProtocol;
    const out = try cells(a, v.get("samples").?.array, @intFromFloat(number(stg.get("repetitions")).?));
    for (out) |*c| c.protocol = sid;
    return out;
}
