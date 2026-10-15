//! Golden receipt bytes, round trips, and the board rules' negative and positive controls.
const std = @import("std");
const json = @import("json.zig");
const suite = @import("benchmark_suite.zig");
const receipt = @import("benchmark_receipt.zig");
const check = @import("benchmark_check.zig");
const fake = @import("benchmark_fake.zig");

const golden = @embedFile("fixtures/benchmark_golden.json");
const golden_preview = @embedFile("fixtures/benchmark_golden_preview.txt");
const now_ms: i64 = 1767225600000 + 60_000;

/// A valid attached cuda receipt: two ok samples, fictional labels.
pub fn valid(a: std.mem.Allocator) !json.Value {
    var samples: [2]suite.Sample = undefined;
    for (&samples, [_][]const u8{ "code", "chat" }) |*s, f| s.* = .{
        .fixture_id = f,
        .temperature = 0.0,
        .repeat = 0,
        .seed = 1234,
        .status = .ok,
        .prompt_tokens = 24,
        .completion_tokens = 4,
        .cached_tokens = 0,
        .delivery_seconds = 0.3,
        .delivery_tps = 10.0,
        .ttft_seconds = 0.05,
        .end_to_end_seconds = 0.4,
        .server_decode_tps = 12.0,
        .server_decode_seconds = 0.25,
        .prefill_seconds = 0.04,
        .token_sha = "0123456789ab",
    };
    return receipt.value(a, "00000000-0000-4000-8000-000000000001", "2026-01-01T00:00:00Z", .{ .repo_id = "example-org/test-model", .local = false }, .{
        .tensorfold_version = "0.0.0-test",
        .backend = "cuda",
        .python_version = "3.11.0",
        .platform = "linux",
    }, .{}, .{ .tokens = 4, .repetitions = 1, .temperatures = &.{0.0}, .serial = false, .context_tokens = null, .managed = false }, &samples);
}

/// Replaces the value at ``path`` (keys, or indices for lists) with parsed ``text``; ``null`` removes the key.
pub fn set(a: std.mem.Allocator, v: json.Value, path: []const []const u8, text: ?[]const u8) !void {
    var at = v;
    for (path[0 .. path.len - 1]) |k| at = if (at == .array) at.array[try std.fmt.parseInt(usize, k, 10)] else at.get(k).?;
    const last = path[path.len - 1];
    const new = if (text) |t| (try json.parse(a, t)).ok else null;
    if (at == .array) at.array[try std.fmt.parseInt(usize, last, 10)] = new.? else if (new) |n| try at.object.put(a, last, n) else _ = at.object.orderedRemove(last);
}

test "the receipt built from the golden facts equals the golden canonical bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.mem.concat(a, u8, &.{ try receipt.canonical(a, try valid(a)), "\n" });
    try std.testing.expectEqual(@as(usize, 1481), bytes.len);
    try std.testing.expectEqualStrings(golden, bytes);
}

test "the golden compare fails when 10.0 is written as an integer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try valid(a);
    try set(a, v, &.{ "samples", "0", "delivery_tps" }, "10");
    const bytes = try std.mem.concat(a, u8, &.{ try receipt.canonical(a, v), "\n" });
    try std.testing.expect(!std.mem.eql(u8, golden, bytes));
}

test "the parsed golden writes back byte for byte, and previews as json.dumps(indent=2)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = (try json.parse(a, golden[0 .. golden.len - 1])).ok;
    try std.testing.expectEqualStrings(golden[0 .. golden.len - 1], try receipt.canonical(a, v));
    var out: std.Io.Writer.Allocating = .init(a);
    try receipt.preview(&out.writer, v);
    try std.testing.expectEqual(@as(usize, 2008), out.written().len);
    try std.testing.expectEqualStrings(golden_preview, out.written());
}

test "a saved receipt is private, atomic and loads back equal" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try fake.tmpPath(a, &tmp);
    const path = try std.fs.path.join(a, &.{ root, "nested", "run.json" });
    var problem: []const u8 = "";
    try std.testing.expect(try receipt.save(a, io, try valid(a), path, now_ms, &problem));
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode() & 0o777)));
    var nested = try tmp.dir.openDir(io, "nested", .{ .iterate = true });
    defer nested.close(io);
    var it = nested.iterate();
    var names: usize = 0;
    while (try it.next(io)) |e| : (names += 1) try std.testing.expectEqualStrings("run.json", e.name);
    try std.testing.expectEqual(@as(usize, 1), names);
    const loaded = (try receipt.load(a, io, path, now_ms, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expect(json.equal(loaded, try valid(a)));
    try std.testing.expectEqualStrings(golden, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)));
}

test "oversized, duplicate-key, non-finite and deeply nested files are refused with fixed words" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(a, &.{ try fake.tmpPath(a, &tmp), "run.json" });
    const body = golden[0 .. golden.len - 2];
    const cases = [_]struct { []const u8, []const u8 }{
        .{ try std.mem.concat(a, u8, &.{ body, ",\"schema_version\":1}" }), "Duplicate JSON field" },
        .{ try std.mem.replaceOwned(u8, a, golden, "\"delivery_tps\":10.0", "\"delivery_tps\":NaN"), "Nonfinite JSON number" },
        .{ try std.mem.concat(a, u8, &.{ try repeat(a, '[', 1000), "0", try repeat(a, ']', 1000) }), "Invalid benchmark JSON" },
        .{ try std.mem.concat(a, u8, &.{ golden, try repeat(a, ' ', 1 << 20) }), "Benchmark receipt exceeds 1 MiB" },
    };
    for (cases) |c| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = c[0] });
        var problem: []const u8 = "";
        try std.testing.expect(try receipt.load(a, io, path, now_ms, &problem) == null);
        try std.testing.expectEqualStrings(c[1], problem);
    }
}

test "created_at has microseconds only when present, and run ids are lowercase v4" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("2026-01-01T00:00:00Z", receipt.createdAt(1767225600 * std.time.ns_per_s, &buf));
    try std.testing.expectEqualStrings("2026-01-01T00:00:00.000005Z", receipt.createdAt(1767225600 * std.time.ns_per_s + 5000, &buf));
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(std.testing.allocator);
    var ids: [1000][36]u8 = undefined;
    for (&ids) |*id| {
        id.* = receipt.runId(std.testing.io);
        var why: []const u8 = "";
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const v = try valid(arena.allocator());
        try v.object.put(arena.allocator(), "run_id", .{ .string = id });
        try std.testing.expect(check.receipt(v, .publish, now_ms, &why));
        try std.testing.expect(!(try seen.getOrPut(std.testing.allocator, id)).found_existing);
    }
}

test "a managed native runtime with python_version none passes publish; null and empty do not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try valid(a);
    try set(a, v, &.{ "runtime", "python_version" }, "\"none\"");
    var why: []const u8 = "";
    try std.testing.expect(check.receipt(v, .publish, now_ms, &why));
    for ([_][]const u8{ "null", "\"\"" }) |bad| {
        try set(a, v, &.{ "runtime", "python_version" }, bad);
        try std.testing.expect(!check.receipt(v, .publish, now_ms, &why));
        try std.testing.expectEqualStrings("runtime.python_version must be a version string", why);
    }
}

fn repeat(a: std.mem.Allocator, byte: u8, n: usize) ![]u8 {
    const out = try a.alloc(u8, n);
    @memset(out, byte);
    return out;
}

const Mutation = struct { path: []const []const u8, text: ?[]const u8, field: []const u8 };

test "every live-board mutation fails with a message naming its field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mutations = [_]Mutation{
        .{ .path = &.{ "suite", "sha256" }, .text = "\"9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5bf\"", .field = "suite" },
        .{ .path = &.{ "samples", "0", "token_sha" }, .text = null, .field = "samples entries" },
        .{ .path = &.{"client"}, .text = "\"x\"", .field = "receipt" },
        .{ .path = &.{ "model", "repo_id" }, .text = "\"https://example.test/private-model\"", .field = "model.repo_id" },
        .{ .path = &.{ "runtime", "backend" }, .text = "\"metal\"", .field = "runtime.backend" },
        .{ .path = &.{ "runtime", "backend" }, .text = "\"mlx\"", .field = "runtime.backend" },
        .{ .path = &.{ "runtime", "dependencies" }, .text = "{\"zig\": \"0.17.0\"}", .field = "runtime.dependencies" },
        .{ .path = &.{"created_at"}, .text = "\"2026-01-01T00:00:00+00:00\"", .field = "created_at" },
        .{ .path = &.{"created_at"}, .text = "\"2026-01-01T00:00:00.1234567Z\"", .field = "created_at" },
        .{ .path = &.{"created_at"}, .text = "\"2026-02-30T00:00:00Z\"", .field = "created_at" },
        .{ .path = &.{"created_at"}, .text = "\"2026-01-01T01:00:00Z\"", .field = "created_at" },
        .{ .path = &.{"run_id"}, .text = "\"00000000-0000-4000-8000-00000000000A\"", .field = "run_id" },
        .{ .path = &.{ "samples", "0", "token_sha" }, .text = "\"0123456789AB\"", .field = "samples token_sha" },
        .{ .path = &.{ "hardware", "gpu_count" }, .text = "1", .field = "hardware.gpu_count" },
        .{ .path = &.{ "samples", "0", "temperature" }, .text = "1.0", .field = "samples must belong" },
        .{ .path = &.{ "samples", "0", "completion_tokens" }, .text = "3", .field = "samples delivery_tps" },
        .{ .path = &.{"schema_version"}, .text = "2", .field = "schema_version" },
        .{ .path = &.{"samples"}, .text = "[]", .field = "samples must hold" },
        .{ .path = &.{ "samples", "0", "prompt_tokens" }, .text = "false", .field = "samples prompt_tokens" },
        .{ .path = &.{ "samples", "1", "fixture_id" }, .text = "\"code\"", .field = "samples must not repeat" },
        .{ .path = &.{ "samples", "0", "repeat" }, .text = "-1", .field = "only a failed warm-up" },
        .{ .path = &.{ "samples", "0", "cached_tokens" }, .text = "25", .field = "samples cached_tokens" },
        .{ .path = &.{ "samples", "0", "delivery_seconds" }, .text = "0.301", .field = "samples delivery_tps" },
        .{ .path = &.{ "settings", "tokens" }, .text = "true", .field = "settings.tokens" },
        .{ .path = &.{ "model", "family" }, .text = "\"<script>alert(1)</script>\"", .field = "model.family" },
    };
    for (mutations) |m| {
        const v = try valid(a);
        try set(a, v, m.path, m.text);
        var why: []const u8 = "";
        if (check.receipt(v, .publish, now_ms, &why)) {
            std.debug.print("mutation of {s} passed\n", .{m.path[m.path.len - 1]});
            return error.TestUnexpectedResult;
        }
        try std.testing.expect(std.mem.startsWith(u8, why, m.field));
    }
}

test "positive controls pass the publish rules" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const passes = [_]Mutation{
        .{ .path = &.{ "runtime", "python_version" }, .text = "\"unknown\"", .field = "" },
        .{ .path = &.{ "runtime", "dependencies" }, .text = "{\"driver\": \"580.65.06\"}", .field = "" },
        .{ .path = &.{"created_at"}, .text = "\"2026-01-01T00:00:00.123456Z\"", .field = "" },
        .{ .path = &.{ "settings", "tokens" }, .text = "4.0", .field = "" },
        .{ .path = &.{ "samples", "0", "token_sha" }, .text = "null", .field = "" },
    };
    for (passes) |m| {
        const v = try valid(a);
        try set(a, v, m.path, m.text);
        var why: []const u8 = "";
        if (!check.receipt(v, .publish, now_ms, &why)) {
            std.debug.print("positive control failed: {s}\n", .{why});
            return error.TestUnexpectedResult;
        }
    }
}

test "a local checkpoint and an over-budget sample save, and publishing refuses them before any network" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const local = try valid(a);
    try set(a, local, &.{ "model", "repo_id" }, "null");
    try set(a, local, &.{ "model", "local" }, "true");
    try std.testing.expect(check.receipt(local, .save, now_ms, &why));
    try std.testing.expect(!check.receipt(local, .publish, now_ms, &why));
    try std.testing.expectEqualStrings(check.local_label, why);
    const over = try valid(a);
    try set(a, over, &.{ "samples", "0", "status" }, "\"error\"");
    try set(a, over, &.{ "samples", "0", "error_code" }, "\"token_count_mismatch\"");
    try set(a, over, &.{ "samples", "0", "completion_tokens" }, "5");
    try set(a, over, &.{ "samples", "0", "delivery_tps" }, "null");
    try std.testing.expect(check.receipt(over, .save, now_ms, &why));
    try std.testing.expect(!check.receipt(over, .publish, now_ms, &why));
    try std.testing.expectEqualStrings(check.over_budget, why);
}

test "a rocm receipt on linux saves but never publishes; rocm on macos fails both modes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const v = try valid(a);
    try set(a, v, &.{ "runtime", "backend" }, "\"rocm\"");
    if (!check.receipt(v, .save, now_ms, &why)) {
        std.debug.print("rocm on linux failed save: {s}\n", .{why});
        return error.TestUnexpectedResult;
    }
    try std.testing.expect(!check.receipt(v, .publish, now_ms, &why));
    try std.testing.expectEqualStrings(check.rocm_local, why);
    try set(a, v, &.{ "runtime", "platform" }, "\"macos\"");
    try std.testing.expect(!check.receipt(v, .save, now_ms, &why));
    try std.testing.expectEqualStrings("runtime.backend and platform must be mlx on macos, or cuda or rocm on linux", why);
    try std.testing.expect(!check.receipt(v, .publish, now_ms, &why));
    try std.testing.expectEqualStrings("runtime.backend and platform must be mlx on macos or cuda on linux", why);
    const cuda_on_mac = try valid(a);
    try set(a, cuda_on_mac, &.{ "runtime", "platform" }, "\"macos\"");
    try std.testing.expect(!check.receipt(cuda_on_mac, .save, now_ms, &why));
}

test "a receipt path is checked before a run: a directory name is refused, its directory is made, a file fits, nothing is left" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try fake.tmpPath(a, &tmp);
    const cwd = std.Io.Dir.cwd();
    var problem: []const u8 = "";
    const sub = try std.fs.path.join(a, &.{ root, "new", "sub" });
    try std.testing.expect(try receipt.writable(a, io, try std.fs.path.join(a, &.{ sub, "run.json" }), &problem));
    var made = try cwd.openDir(io, sub, .{ .iterate = true });
    defer made.close(io);
    var it = made.iterate();
    try std.testing.expect(try it.next(io) == null);
    const blocker = try std.fs.path.join(a, &.{ root, "blocker" });
    try cwd.writeFile(io, .{ .sub_path = blocker, .data = "" });
    try std.testing.expect(!try receipt.writable(a, io, try std.fs.path.join(a, &.{ blocker, "run.json" }), &problem));
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Could not create the receipt directory {s}", .{blocker}), problem);
    try std.testing.expect(!try receipt.writable(a, io, sub, &problem));
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}", .{sub}), problem);
    const unmade = try std.fs.path.join(a, &.{ root, "unmade", "dir" });
    for ([_][]const u8{ "/", "/.", "/.." }) |suffix| {
        const named = try std.mem.concat(a, u8, &.{ unmade, suffix });
        try std.testing.expect(!try receipt.writable(a, io, named, &problem));
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}", .{named}), problem);
    }
    try std.testing.expectError(error.FileNotFound, cwd.statFile(io, try std.fs.path.join(a, &.{ root, "unmade" }), .{}));
    if (std.c.geteuid() == 0) return;
    const locked = try std.fs.path.join(a, &.{ root, "locked" });
    try cwd.createDirPath(io, locked);
    try cwd.setFilePermissions(io, locked, .fromMode(0o500), .{});
    defer cwd.setFilePermissions(io, locked, .fromMode(0o700), .{}) catch {};
    const inside = try std.fs.path.join(a, &.{ locked, "run.json" });
    try std.testing.expect(!try receipt.writable(a, io, inside, &problem));
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Could not save the benchmark receipt to {s}", .{inside}), problem);
}
