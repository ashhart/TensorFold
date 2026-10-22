//! cli.zig's public surface: what parse reads, what capabilities lists and what request asks an engine for.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const Usage = cli.Usage;
const parse = cli.parse;
const request = cli.request;
const capabilities = cli.capabilities;

test "vision image budget: explicit large budgets reach the engine, with bounded CLI inputs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    for ([_][]const u8{ "16", "16384", "65536", "262144" }) |value| {
        const n = try std.fmt.parseInt(u32, value, 10);
        for ([_][]const []const u8{
            &.{ "m", "--vision", "--vision-image-tokens", value },
            &.{ "m", "--vision", try std.fmt.allocPrint(a, "--vision-image-tokens={s}", .{value}) },
        }) |argv| {
            const args = try parse(a, argv, &u);
            const vision = (try request(a, "d", "t", args)).vision.?;
            try std.testing.expectEqual(n, vision.image_tokens);
            try std.testing.expect(vision.image_tokens_set);
        }
    }
    const normal = (try request(a, "d", "t", try parse(a, &.{ "m", "--vision" }, &u))).vision.?;
    try std.testing.expectEqual(@as(u32, 4096), normal.image_tokens);
    try std.testing.expect(!normal.image_tokens_set);
    for ([_][]const u8{ "15", "262145", "-1", "4294967296", "many" }) |bad|
        try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--vision", "--vision-image-tokens", bad }, &u));
}

test "parse and capabilities share the table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const args = try parse(a, &.{ "/models/x", "--port", "9000", "--api-key=k1", "--api-key", "k2", "--no-thinking", "--reasoning-effort", "low" }, &u);
    try std.testing.expectEqual(@as(u16, 9000), args.port);
    try std.testing.expectEqual(@as(usize, 2), args.api_key.len);
    try std.testing.expect(!args.thinking);
    try std.testing.expectEqualStrings("x", (try parse(a, &.{ "m", "--drafter", "x" }, &u)).drafter.?);
    try std.testing.expectEqual(@as(u8, 0), (try parse(a, &.{ "m", "--drafter-bits", "0" }, &u)).drafter_bits);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--drafter-bits", "8" }, &u));
    try std.testing.expectEqualStrings("/r", (try parse(a, &.{ "m", "--record", "/r" }, &u)).record.?);
    try std.testing.expectEqualStrings("fp8", (try parse(a, &.{ "m", "--kv-cache", "fp8" }, &u)).kv_cache);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--kv-cache", "int4" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--reasoning-effort", "max" }, &u));
    try std.testing.expectEqual(@as(?f64, 12.5), (try parse(a, &.{ "m", "--prompt-cache-gib", "12.5" }, &u)).prompt_cache_gib);
    try std.testing.expectEqual(@as(?f64, 0), (try parse(a, &.{ "m", "--prompt-cache-gib", "0" }, &u)).prompt_cache_gib);
    try std.testing.expect(!(try parse(a, &.{ "m", "--prompt-cache-gib", "16" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--prompt-cache-gib", "16", "--prompt-cache-over-cap" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--learn" }, &u)).learn);
    const dir = try parse(a, &.{ "m", "--learn-dir", "/x/y" }, &u);
    try std.testing.expect(dir.learn and std.mem.eql(u8, "/x/y", dir.learn_dir.?));
    const capped = try parse(a, &.{ "m", "--learn-gib", "8" }, &u);
    try std.testing.expect(capped.learn and capped.learn_gib == 8);
    const talks = try parse(a, &.{ "m", "--learn-conversations", "--learn-daily-gib", "2.5" }, &u);
    try std.testing.expect(talks.learn and talks.learn_conversations and talks.learn_daily_gib == 2.5);
    try std.testing.expect((try request(a, "d", "t", talks)).learn_conversations);
    try std.testing.expect(!(try request(a, "d", "t", capped)).learn_conversations);
    try std.testing.expect(!(try parse(a, &.{"m"}, &u)).recall and (try parse(a, &.{ "m", "--recall" }, &u)).recall);
    const named = try parse(a, &.{ "m", "--recall-dir", "/r", "--recall-gib", "0.5", "--recall-read-only" }, &u);
    try std.testing.expect(named.recall and named.recall_read_only and named.recall_gib == 0.5);
    try std.testing.expectEqualStrings("/r", named.recall_dir.?);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--recall-gib", "0.001" }, &u));
    for ([_][]const u8{ "1e300", "inf", "nan", "-1", "17179869184" }) |bad| try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--prompt-cache-gib", bad }, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0.6.5" });
    const doc = out.written();
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--no-thinking\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--recall-read-only\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--drafter\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--compact-at\": {}") != null);
    const on = try parse(a, &.{ "m", "--compact-at", "auto", "--compact-keep", "100", "--compact-memory", "notes" }, &u);
    try std.testing.expect(on.compact_auto);
    try std.testing.expectEqual(@as(?u32, 100), on.compact_keep);
    try std.testing.expectEqualStrings("notes", on.compact_memory.?);
    try std.testing.expectEqualStrings("graph.json", (try parse(a, &.{ "m", "--slide-graph", "graph.json" }, &u)).slide_graph.?);
    try std.testing.expect((try parse(a, &.{ "m", "--slide" }, &u)).slide);
    try std.testing.expect((try parse(a, &.{ "m", "--cortheon" }, &u)).cortheon and !(try parse(a, &.{"m"}, &u)).cortheon);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--cortheon\": {}") != null);
    try std.testing.expectEqual(@as(?f64, 0.5), (try parse(a, &.{ "m", "--compact-at", "0.5" }, &u)).compact_fraction);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "0" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "2" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-keep", "-1" }, &u));
    const urls = try parse(a, &.{ "m", "--vision", "--vision-urls" }, &u);
    try std.testing.expect(urls.vision and urls.vision_urls);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--vision-urls" }, &u));
    try std.testing.expectEqualStrings("--vision-urls needs --vision", u.message);
}

test "learned disk floor parses once with a finite byte-safe default and is a native capability" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var u: Usage = .{};
    const normal = try parse(arena.allocator(), &.{"m"}, &u);
    try std.testing.expectEqual(@as(f64, 4), normal.learn_min_free_gib);
    const explicit = try parse(arena.allocator(), &.{ "m", "--learn-min-free-gib", "1.5" }, &u);
    try std.testing.expectEqual(@as(f64, 1.5), explicit.learn_min_free_gib);
    for ([_][]const u8{ "nan", "inf", "-1", "17179869184" }) |bad| {
        try std.testing.expectError(error.Usage, parse(arena.allocator(), &.{ "m", "--learn-min-free-gib", bad }, &u));
    }
}

test "an engine is asked for what the flags say, and the defaults of the flags not given" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try request(a, "d", "t", .{});
    const want: api.Open = .{ .dir = "d", .model_type = "t", .learn = null };
    inline for (.{ "mtp_drafts", "mtp_confidence", "keep", "tp", "rank", "master_port", "lanes", "drafts", "device", "segments" }) |field| try std.testing.expectEqual(@field(want, field), @field(plain, field));
    try std.testing.expectEqualStrings("", plain.policy);
    try std.testing.expectEqualStrings("", plain.master);
    const gpu = try request(a, "d", "t", .{ .tp = 4, .rank = 1, .master = "node0", .master_port = 29600, .mtp_drafts = 2, .mtp_confidence = 0.5, .checkpoint_slots = 3, .policy = "matrix=off" });
    try std.testing.expectEqual(@as(u32, 4), gpu.tp);
    try std.testing.expectEqual(@as(u32, 1), gpu.rank);
    try std.testing.expectEqualStrings("node0", gpu.master);
    try std.testing.expectEqual(@as(u16, 29600), gpu.master_port);
    try std.testing.expectEqual(@as(?u32, 2), gpu.mtp_drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), gpu.mtp_confidence);
    try std.testing.expectEqual(@as(?i64, 3), gpu.keep);
    try std.testing.expectEqualStrings("matrix=off", gpu.policy);
}

test "runtime library: capabilities serializes found and missing dependencies with escaped descriptions" {
    const libs = @import("sys").optional_libraries;
    const specs = [_]libs.Spec{ libs.spec(.nccl), libs.spec(.cublaslt) };
    var rows: [2]libs.Status = undefined;
    for (specs, &rows, 0..) |s, *row, i| row.* = .{ .name = s.name, .feature = s.feature, .soname = s.names[0], .found = i == 0, .install = s.install };
    rows[1].install = "install the \"runtime\"";
    rows[1].usable = false;
    rows[1].problem = "wrong \"version\"";
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try capabilities(&out.writer, .{ .version = "test", .optional_libraries = &rows });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const values = parsed.value.object.get("optional_libraries").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), values.len);
    try std.testing.expect(values[0].object.get("found").?.bool);
    try std.testing.expect(!values[1].object.get("found").?.bool);
    try std.testing.expectEqualStrings("libnccl.so.2", values[0].object.get("soname").?.string);
    try std.testing.expectEqualStrings(rows[1].install, values[1].object.get("install").?.string);
    try std.testing.expect(values[0].object.get("usable").? == .null);
    try std.testing.expect(!values[1].object.get("usable").?.bool);
    try std.testing.expectEqualStrings(rows[1].problem.?, values[1].object.get("problem").?.string);
}
