//! The local draft counts beside a receipt (rounds, drafted, accepted per attempt); never uploaded to the board.
const std = @import("std");
const json = @import("json");
const suite = @import("benchmark_suite.zig");
const receipt = @import("benchmark_receipt.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const schema = "tensorfold-benchmark-drafts/1";

/// The attempts' counts summed over those whose server reported them; ``sums`` is null once a sum passes 64 bits.
pub const Totals = struct { attempts: usize = 0, reported: usize = 0, sums: ?suite.Drafts = .{ .rounds = 0, .drafted = 0, .accepted = 0 } };

pub fn totals(samples: []const suite.Sample) Totals {
    var t: Totals = .{ .attempts = samples.len };
    for (samples) |s| if (s.drafts) |d| {
        t.reported += 1;
        if (t.sums) |sum| t.sums = add(sum, d);
    };
    return t;
}

/// Each counter of ``x`` plus ``y``; null when one passes 64 bits, so two large counts give no totals, not a panic.
fn add(x: suite.Drafts, y: suite.Drafts) ?suite.Drafts {
    return .{
        .rounds = std.math.add(u64, x.rounds, y.rounds) catch return null,
        .drafted = std.math.add(u64, x.drafted, y.drafted) catch return null,
        .accepted = std.math.add(u64, x.accepted, y.accepted) catch return null,
    };
}

/// ``run.json`` -> ``run.drafts.json``; any other name gets ``.drafts.json`` appended.
pub fn path(a: Allocator, receipt_path: []const u8) ![]const u8 {
    const stem = if (std.mem.endsWith(u8, receipt_path, ".json")) receipt_path[0 .. receipt_path.len - ".json".len] else receipt_path;
    return std.mem.concat(a, u8, &.{ stem, ".drafts.json" });
}

/// What a run's draft counts show; ``passed``: a drafter went to the server (null when attached and unknown).
pub fn verdict(a: Allocator, t: Totals, passed: ?bool, file: []const u8) ![]const u8 {
    if (t.reported == 0) return std.fmt.allocPrint(a, "Draft counts: the server reported none; saved: {s}", .{file});
    const sum = t.sums orelse return std.fmt.allocPrint(a, "Draft counts: no totals, as a sum passes 64 bits ({d} of {d} attempts reported counts); saved: {s}", .{ t.reported, t.attempts, file });
    const line = try std.fmt.allocPrint(a, "Draft counts: {d} of {d} drafted tokens accepted over {d} rounds ({d} of {d} attempts reported them); saved: {s}", .{ sum.accepted, sum.drafted, sum.rounds, t.reported, t.attempts, file });
    if ((passed orelse false) and sum.drafted == 0) return std.mem.concat(a, u8, &.{ line, "\n", not_drafted });
    return line;
}

/// The note when a drafter was passed and the server proposed no tokens with it.
pub const not_drafted = "A drafter was passed, but the server drafted no tokens: this run does not show that the drafter ran.";

/// The file: the run it belongs to, the drafter, the totals and one entry per receipt sample, in receipt order.
pub fn value(a: Allocator, run_id: []const u8, receipt_name: []const u8, drafter_repo_id: ?[]const u8, passed: ?bool, samples: []const suite.Sample) !Value {
    const top = try json.newObject(a);
    try top.put(a, "schema", .{ .string = schema });
    try top.put(a, "run_id", .{ .string = run_id });
    try top.put(a, "receipt", .{ .string = receipt_name });
    try top.put(a, "drafter_repo_id", if (drafter_repo_id) |d| .{ .string = d } else .null);
    try top.put(a, "drafter_passed", if (passed) |p| .{ .bool = p } else .null);
    const t = totals(samples);
    const sum = try json.newObject(a);
    try sum.put(a, "attempts", try json.intValue(a, t.attempts));
    try sum.put(a, "reported", try json.intValue(a, t.reported));
    // a sum past 64 bits has no total: null, as an attempt without counts has
    inline for (.{ "rounds", "drafted", "accepted" }) |k| try sum.put(a, k, if (t.sums) |d| try json.intValue(a, @field(d, k)) else .null);
    try top.put(a, "totals", .{ .object = sum });
    const rows = try a.alloc(Value, samples.len);
    for (samples, rows) |s, *slot| {
        const o = try json.newObject(a);
        try o.put(a, "fixture_id", .{ .string = s.fixture_id });
        try o.put(a, "temperature", .{ .float = s.temperature });
        try o.put(a, "repeat", try json.intValue(a, s.repeat));
        try o.put(a, "status", .{ .string = @tagName(s.status) });
        try o.put(a, "completion_tokens", if (s.completion_tokens) |n| try json.intValue(a, n) else .null);
        inline for (.{ "rounds", "drafted", "accepted" }) |k| try o.put(a, k, if (s.drafts) |d| try json.intValue(a, @field(d, k)) else .null);
        slot.* = .{ .object = o };
    }
    try top.put(a, "samples", .{ .array = rows });
    return .{ .object = top };
}

/// Writes ``v`` as canonical JSON plus a newline, privately and atomically, as the receipt is written.
pub fn save(a: Allocator, io: std.Io, v: Value, file: []const u8, problem: *[]const u8) !bool {
    const bytes = try std.mem.concat(a, u8, &.{ try receipt.canonical(a, v), "\n" });
    const dir = std.fs.path.dirname(file) orelse ".";
    const cwd = std.Io.Dir.cwd();
    var name: [8]u8 = undefined;
    io.random(&name);
    const temp = try std.fmt.allocPrint(a, "{s}/.benchmark-{s}", .{ dir, std.fmt.bytesToHex(name, .lower) });
    problem.* = try std.fmt.allocPrint(a, "Could not save the draft counts to {s}", .{file});
    var f = cwd.createFile(io, temp, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch return false;
    const written = if (f.writeStreamingAll(io, bytes)) true else |_| false;
    f.close(io);
    if (!written or (if (cwd.rename(temp, cwd, file, io)) false else |_| true)) {
        cwd.deleteFile(io, temp) catch {};
        return false;
    }
    problem.* = "";
    return true;
}

test "the file's name sits beside the receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("/r/run.drafts.json", try path(a, "/r/run.json"));
    try std.testing.expectEqualStrings("/r/run.out.drafts.json", try path(a, "/r/run.out"));
}

test "totals sum only the attempts that reported counts, and the verdict says when a passed drafter drafted nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const drafted = [_]suite.Sample{
        .{ .fixture_id = "code", .temperature = 0, .repeat = 0, .seed = 1, .status = .ok, .drafts = .{ .rounds = 14, .drafted = 13, .accepted = 10 } },
        .{ .fixture_id = "chat", .temperature = 0, .repeat = 0, .seed = 1, .status = .ok, .drafts = .{ .rounds = 6, .drafted = 5, .accepted = 2 } },
        .{ .fixture_id = "chat", .temperature = 0, .repeat = 1, .seed = 1, .status = .@"error" },
    };
    const t = totals(&drafted);
    try std.testing.expect(t.attempts == 3 and t.reported == 2);
    try std.testing.expectEqual(suite.Drafts{ .rounds = 20, .drafted = 18, .accepted = 12 }, t.sums.?);
    const shown = try verdict(a, t, true, "f");
    try std.testing.expect(std.mem.startsWith(u8, shown, "Draft counts: 12 of 18 drafted tokens accepted over 20 rounds (2 of 3 attempts"));
    try std.testing.expect(std.mem.indexOf(u8, shown, not_drafted) == null);
    const plain = [_]suite.Sample{.{ .fixture_id = "code", .temperature = 0, .repeat = 0, .seed = 1, .status = .ok, .drafts = .{ .rounds = 32, .drafted = 0, .accepted = 0 } }};
    try std.testing.expect(std.mem.endsWith(u8, try verdict(a, totals(&plain), true, "f"), not_drafted));
    for ([_]?bool{ false, null }) |p| try std.testing.expect(std.mem.indexOf(u8, try verdict(a, totals(&plain), p, "f"), not_drafted) == null);
    const silent = [_]suite.Sample{.{ .fixture_id = "code", .temperature = 0, .repeat = 0, .seed = 1, .status = .ok }};
    try std.testing.expectEqualStrings("Draft counts: the server reported none; saved: f", try verdict(a, totals(&silent), true, "f"));
    const v = try value(a, "00000000-0000-4000-8000-000000000001", "run.json", "z-lab/Qwen3.8-27B-DFlash2", true, &drafted);
    const text = try json.stringify(a, v, .{ .compact = true });
    try std.testing.expect(std.mem.indexOf(u8, text, "\"totals\":{\"attempts\":3,\"reported\":2,\"rounds\":20,\"drafted\":18,\"accepted\":12}") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"repeat\":1,\"status\":\"error\",\"completion_tokens\":null,\"rounds\":null,\"drafted\":null,\"accepted\":null") != null);
}

test "sums that would pass 64 bits give no totals rather than a panic, and the attempts keep their own counts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const max = std.math.maxInt(u64);
    const at = struct {
        fn sample(repeat: i32, d: suite.Drafts) suite.Sample {
            return .{ .fixture_id = "code", .temperature = 0, .repeat = repeat, .seed = 1, .status = .ok, .drafts = d };
        }
    }.sample;
    // a sum of exactly 2^64 - 1 still fits
    const fits = [_]suite.Sample{ at(0, .{ .rounds = 1, .drafted = max - 1, .accepted = 1 }), at(1, .{ .rounds = 1, .drafted = 1, .accepted = 0 }) };
    try std.testing.expectEqual(suite.Drafts{ .rounds = 2, .drafted = max, .accepted = 1 }, totals(&fits).sums.?);
    // one past it gives no totals, whether every sum passes or only the rounds or only the drafted tokens
    const big: suite.Drafts = .{ .rounds = max - 1, .drafted = max, .accepted = max - 1 };
    for ([_]suite.Drafts{ big, .{ .rounds = 2, .drafted = 0, .accepted = 0 }, .{ .rounds = 0, .drafted = 1, .accepted = 0 } }) |second| {
        const over = [_]suite.Sample{ at(0, big), at(1, second), .{ .fixture_id = "chat", .temperature = 0, .repeat = 0, .seed = 1, .status = .@"error" } };
        const t = totals(&over);
        try std.testing.expect(t.attempts == 3 and t.reported == 2 and t.sums == null);
        const shown = try verdict(a, t, true, "f");
        try std.testing.expectEqualStrings("Draft counts: no totals, as a sum passes 64 bits (2 of 3 attempts reported counts); saved: f", shown);
        const text = try json.stringify(a, try value(a, "00000000-0000-4000-8000-000000000001", "run.json", null, true, &over), .{ .compact = true });
        try std.testing.expect(std.mem.indexOf(u8, text, "\"totals\":{\"attempts\":3,\"reported\":2,\"rounds\":null,\"drafted\":null,\"accepted\":null}") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "\"rounds\":18446744073709551614,\"drafted\":18446744073709551615,\"accepted\":18446744073709551614") != null);
    }
}
