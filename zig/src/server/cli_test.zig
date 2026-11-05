//! cli.zig's public surface: what parse reads, what capabilities lists and what request asks an engine for.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const Usage = cli.Usage;
const parse = cli.parse;
const request = cli.request;
const capabilities = cli.capabilities;
const spillWith = cli.Test.spillWith;

test "stable profile: default and explicit spelling preserve every argument and engine open option" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    for ([_][]const []const u8{
        &.{"m"},
        &.{ "m", "--learn-conversations", "--learn-dir=cache", "--prompt-cache-gib=2", "--parallel=4" },
        &.{ "m", "--no-thinking", "--temperature=0.7", "--no-drafts" },
    }) |argv| {
        const plain = try parse(a, argv, &u);
        var extended: std.ArrayList([]const u8) = .empty;
        try extended.appendSlice(a, argv);
        for ([_][]const []const u8{ &.{ "--profile", "stable" }, &.{"--profile=stable"} }) |suffix| {
            extended.shrinkRetainingCapacity(argv.len);
            try extended.appendSlice(a, suffix);
            const explicit = try parse(a, extended.items, &u);
            try std.testing.expectEqualDeep(plain, explicit);
            try std.testing.expectEqualDeep(try request(a, "d", "t", plain), try request(a, "d", "t", explicit));
        }
    }
    for ([_][]const []const u8{
        &.{ "m", "--profile" },                  &.{ "m", "--profile=" },
        &.{ "m", "--profile=fast-exact" },       &.{ "m", "--profile=fast" },
        &.{ "m", "--profile=Stable" },           &.{ "m", "--profile=unknown" },
        &.{ "m", "--profile", "--no-thinking" },
    }) |argv| {
        try std.testing.expectError(error.Usage, parse(a, argv, &u));
        try std.testing.expect(std.mem.indexOf(u8, u.message, "--profile") != null);
    }
    var output: std.Io.Writer.Allocating = .init(a);
    try capabilities(&output.writer, .{ .version = "test" });
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "\"--profile\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "fast-exact") == null);
    const decoded = try std.json.parseFromSlice(std.json.Value, a, output.written(), .{});
    const profiles = decoded.value.object.get("profiles").?.object;
    try std.testing.expectEqualStrings("stable", profiles.get("default").?.string);
    const available = profiles.get("available").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), available.len);
    try std.testing.expectEqualStrings("stable", available[0].string);
    const entries = profiles.get("registered_switches").?.array.items;
    try std.testing.expectEqual(api.profile.entries.len, entries.len);
    for (entries, api.profile.entries) |json_entry, row| {
        try std.testing.expectEqualStrings(row.name(), json_entry.object.get("id").?.string);
        try std.testing.expect(json_entry.object.get("stable").? == .null);
    }
}

test "stable profile: capabilities describe the shared paging variable by family without changing its name" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try capabilities(&output.writer, .{ .version = "test" });
    const decoded = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, output.written(), .{});
    defer decoded.deinit();
    const entries = decoded.value.object.get("profiles").?.object.get("registered_switches").?.array.items;
    var policies: usize = 0;
    for (entries) |row| {
        const fields = row.object;
        if (!std.mem.eql(u8, fields.get("id").?.string, "TF_KV_PAGES")) continue;
        policies += 1;
        try std.testing.expectEqualStrings("cuda", fields.get("backend").?.string);
        try std.testing.expect(fields.get("stable").? == .null);
        const families = fields.get("families").?.array.items;
        if (fields.get("report_values").? == .null) {
            try std.testing.expectEqualStrings("auto", fields.get("syntax").?.string);
            try std.testing.expectEqual(@as(usize, 1), families.len);
            try std.testing.expectEqualStrings("glm5_next", families[0].string);
        } else {
            const values = fields.get("report_values").?.array.items;
            try std.testing.expectEqual(@as(usize, 2), values.len);
            try std.testing.expectEqualStrings("0", values[0].string);
            try std.testing.expectEqualStrings("all", values[1].string);
            try std.testing.expectEqual(@as(usize, 2), families.len);
            try std.testing.expectEqualStrings("nemotron_h", families[0].string);
            try std.testing.expectEqualStrings("nemotron_h_omni", families[1].string);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), policies);
}

test "max frames CLI bounds and engine request preserve the default" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const defaults = try parse(a, &.{"m"}, &u);
    try std.testing.expectEqual(@as(?u32, null), defaults.max_frames);
    try std.testing.expectEqual(@as(?u32, null), (try request(a, "d", "fish_omni", defaults)).max_frames);
    for ([_][]const u8{ "1", "1024", "4096", "8190" }) |value| {
        const args = try parse(a, &.{ "m", "--max-frames", value }, &u);
        const expected = try std.fmt.parseInt(u32, value, 10);
        try std.testing.expectEqual(@as(?u32, expected), args.max_frames);
        try std.testing.expectEqual(args.max_frames, (try request(a, "d", "fish_omni", args)).max_frames);
    }
    for ([_][]const u8{ "0", "-1", "8191", "4294967296", "nan", "1.5" }) |value| {
        try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--max-frames", value }, &u));
    }
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--max-frames" }, &u));
}

test "voices: CLI defaults and explicit catalog" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const defaults = try parse(a, &.{"m"}, &u);
    try std.testing.expect(defaults.voices == null);
    try std.testing.expectEqualStrings("default", defaults.default_voice);
    const named = try parse(a, &.{ "m", "--voices", "catalog", "--default-voice=tone" }, &u);
    try std.testing.expectEqualStrings("catalog", named.voices.?);
    try std.testing.expectEqualStrings("tone", named.default_voice);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--default-voice=../tone" }, &u));
}

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

test "allowed controls CLI is opt-in and validates name syntax only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    try std.testing.expect((try parse(a, &.{"m"}, &u)).allowed_controls == null);
    const on = try parse(a, &.{ "m", "--allowed-controls", "sighs, laughs" }, &u);
    try std.testing.expectEqualDeep(try @import("drama_allow.zig").parse("sighs, laughs"), on.allowed_controls.?);
    const empty = try parse(a, &.{ "m", "--allowed-controls=" }, &u);
    try std.testing.expectEqualDeep(try @import("drama_allow.zig").parse(""), empty.allowed_controls.?);
    const open = try parse(a, &.{ "m", "--allowed-controls", "whisper,unknown,gentle robot" }, &u);
    try std.testing.expectEqualStrings("whisper,unknown,gentle robot", open.allowed_controls.?.list);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--allowed-controls", "[whisper]" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--allowed-controls" }, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "test" });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"--allowed-controls\": {}") != null);
}

test "silence redraw flags are opt-in and need a redraw count and the guard" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    try std.testing.expectEqualDeep(@import("speech_redraw.zig").Policy{}, (try parse(a, &.{"m"}, &u)).redraw);
    const guard = [_][]const u8{ "m", "--silence-floor-dbfs=-60", "--silence-seconds=1" };
    const on = try parse(a, &(guard ++ .{ "--silence-redraws", "2", "--silence-redraw-budget-s=30", "--silence-redraw-controls=", "--short-reply-ratio=0.25", "--short-reply-chars-per-s", "12" }), &u);
    try std.testing.expectEqual(@as(?u8, 2), on.redraw.max);
    try std.testing.expectEqual(@as(?f64, 30), on.redraw.budget_s);
    try std.testing.expectEqualStrings("", on.redraw.controls.?.list);
    try std.testing.expectEqual(@as(?f64, 0.25), on.redraw.short_ratio);
    try std.testing.expectEqual(@as(?f64, 12), on.redraw.short_chars_per_s);
    try std.testing.expect(on.redraw.refusal("nemotron_h") != null);
    const zero = try parse(a, &.{ "m", "--silence-redraws=0" }, &u);
    try std.testing.expect(zero.redraw.refusal("nemotron_h") != null);
    try std.testing.expect(zero.redraw.refusal("fish_omni") == null);
    for ([_][]const []const u8{
        &.{ "m", "--silence-redraws", "1" },
        &(guard ++ .{ "--silence-redraw-budget-s", "30" }),
        &(guard ++ .{ "--silence-redraw-controls", "sighs" }),
        &(guard ++ .{ "--silence-redraws", "256" }),
        &(guard ++ .{ "--silence-redraws", "1", "--silence-redraw-budget-s", "0" }),
        &(guard ++ .{ "--silence-redraws", "1", "--silence-redraw-controls", "[sighs]" }),
        &(guard ++ .{ "--short-reply-ratio", "0.25", "--short-reply-chars-per-s", "12" }),
        &(guard ++ .{ "--silence-redraws", "1", "--short-reply-ratio", "0.25" }),
        &(guard ++ .{ "--silence-redraws", "1", "--short-reply-chars-per-s", "12" }),
        &(guard ++ .{ "--silence-redraws", "1", "--short-reply-ratio", "1", "--short-reply-chars-per-s", "12" }),
        &(guard ++ .{ "--silence-redraws", "1", "--short-reply-ratio", "0.25", "--short-reply-chars-per-s", "0" }),
    }) |bad| try std.testing.expectError(error.Usage, parse(a, bad, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "test" });
    for ([_][]const u8{ "--silence-redraws", "--silence-redraw-budget-s", "--silence-redraw-controls", "--short-reply-ratio", "--short-reply-chars-per-s" }) |flag|
        try std.testing.expect(std.mem.indexOf(u8, out.written(), try std.fmt.allocPrint(a, "\"{s}\": {{}}", .{flag})) != null);
}

test "register flags are opt-in, all or none, and may set a redraw budget without silence redraws" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    try std.testing.expect(!(try parse(a, &.{"m"}, &u)).redraw.register.on());
    const rule = [_][]const u8{ "m", "--register-redraw-st", "6", "--register-redraw-high=50", "--register-high-hz", "300" };
    const on = try parse(a, &(rule ++ .{ "--silence-redraw-budget-s", "30" }), &u);
    try std.testing.expectEqualDeep(@import("speech_register.zig").Rule{ .st = 6, .high = 50, .high_hz = 300 }, on.redraw.register);
    try std.testing.expectEqual(@as(?f64, 30), on.redraw.budget_s);
    for ([_][]const []const u8{
        &.{ "m", "--register-redraw-st", "6" },
        &.{ "m", "--register-redraw-high", "50", "--register-high-hz", "300" },
        rule[0..4],
        &(rule ++ .{ "--register-redraw-st", "0" }),
        &(rule ++ .{ "--register-high-hz", "700" }),
        &(rule ++ .{ "--silence-redraw-controls", "sighs" }),
    }) |bad| try std.testing.expectError(error.Usage, parse(a, bad, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "test" });
    for ([_][]const u8{ "--register-redraw-st", "--register-redraw-high", "--register-high-hz" }) |flag|
        try std.testing.expect(std.mem.indexOf(u8, out.written(), try std.fmt.allocPrint(a, "\"{s}\": {{}}", .{flag})) != null);
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
    try std.testing.expectEqual(@as(?u8, 0), (try parse(a, &.{ "m", "--drafter-bits", "0" }, &u)).drafter_bits);
    try std.testing.expectEqual(@as(?u8, 8), (try parse(a, &.{ "m", "--drafter-bits", "8" }, &u)).drafter_bits);
    try std.testing.expectEqual(@as(?u8, null), (try parse(a, &.{ "m", "--drafter", "x" }, &u)).drafter_bits);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--drafter-bits", "3" }, &u));
    try std.testing.expectEqualStrings("/r", (try parse(a, &.{ "m", "--record", "/r" }, &u)).record.?);
    try std.testing.expectEqualStrings("fp8", (try parse(a, &.{ "m", "--kv-cache", "fp8" }, &u)).kv_cache);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--kv-cache", "int4" }, &u));
    try std.testing.expectEqualStrings("int8", (try parse(a, &.{ "m", "--prompt-precision", "int8" }, &u)).prompt_precision);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--prompt-precision", "fp4" }, &u));
    try std.testing.expectEqualStrings("max", (try parse(a, &.{ "m", "--reasoning-effort", "max" }, &u)).reasoning_effort.?);
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
    const pdfs = try parse(a, &.{ "m", "--vision", "--vision-pdf" }, &u);
    try std.testing.expect(pdfs.vision and pdfs.vision_pdf);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--vision-pdf" }, &u));
    try std.testing.expectEqualStrings("--vision-pdf needs --vision", u.message);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--vision-pdf\": {}") != null);
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
    try std.testing.expectEqual(@as(i64, 3), gpu.keep);
    try std.testing.expectEqualStrings("matrix=off", gpu.policy);
}

test "--spill-gib asks the engine for the conversation store, spilling only what RAM evicts, and is off without it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const plain = try request(a, "d", "t", .{});
    try std.testing.expect(plain.spill == .off and plain.learn == null and !plain.learn_conversations and plain.learn_gib == 32);
    const zero = try request(a, "d", "t", .{ .spill_gib = 0 });
    try std.testing.expect(zero.spill == .off and zero.learn == null and !zero.learn_conversations and zero.learn_gib == 32);
    const alone = try request(a, "d", "t", .{ .spill_gib = 2, .checkpoint_slots = 3 });
    try std.testing.expect(alone.spill == .evicted and alone.learn_conversations and alone.learn_gib == 2);
    try std.testing.expectStringEndsWith(alone.learn.?, "/.cache/tensorfold/spilled");
    try std.testing.expectEqual(@as(?i64, 3), alone.keep);
    const both = try request(a, "d", "t", .{ .spill_gib = 8, .learn = true, .learn_conversations = true, .learn_dir = "/x" });
    try std.testing.expect(both.spill == .finish and both.learn_conversations and both.learn_gib == 8);
    try std.testing.expectEqualStrings("/x", both.learn.?);
    const capped = try request(a, "d", "t", .{ .spill_gib = 64, .learn = true, .learn_conversations = true, .learn_gib = 16 });
    try std.testing.expect(capped.spill == .finish and capped.learn_gib == 16); // the smaller cap
    try spillWith(&u, a, .{ .spill_gib = 2 });
    try spillWith(&u, a, .{ .spill_gib = 2, .learn = true, .learn_conversations = true });
    try spillWith(&u, a, .{ .learn = true });
    try std.testing.expectError(error.Usage, spillWith(&u, a, .{ .spill_gib = 2, .learn = true }));
    try std.testing.expectStringStartsWith(u.message, "--spill-gib writes evicted states to the conversation store");
}

test "silence guard flags are opt-in and reach native open options" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: Usage = .{};
    try std.testing.expect(!(try parse(a, &.{"m"}, &usage)).silence.enabled());
    try std.testing.expect((try parse(a, &.{ "m", "--silence-seconds=0" }, &usage)).silence.refusal("nemotron_h") != null);
    const args = try parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "0.5" }, &usage);
    const opened = try request(a, "m", "drama3", args);
    try std.testing.expectEqual(args.silence, opened.silence);
    try std.testing.expect(opened.silence.enabled());
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-seconds", "1" }, &usage));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-seconds", "nan" }, &usage));
    const onset = try parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "2", "--silence-onset-s", "1", "--silence-onset-voiced-s", "0.5" }, &usage);
    try std.testing.expectEqual(api.speech_guard.Config{ .floor_dbfs = -80, .seconds = 2, .onset_seconds = 1, .onset_voiced_seconds = 0.5 }, (try request(a, "m", "drama3", onset)).silence);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "2", "--silence-onset-s", "1" }, &usage));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-onset-s=0" }, &usage));
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
    try std.testing.expectEqualStrings(libs.spec(.nccl).names[0], values[0].object.get("soname").?.string);
    try std.testing.expectEqualStrings(rows[1].install, values[1].object.get("install").?.string);
    try std.testing.expect(values[0].object.get("usable").? == .null);
    try std.testing.expect(!values[1].object.get("usable").?.bool);
    try std.testing.expectEqualStrings(rows[1].problem.?, values[1].object.get("problem").?.string);
}

test "capabilities: a p2p summary appears only when the engine gives one, as its own object" {
    var none: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer none.deinit();
    try capabilities(&none.writer, .{ .version = "test" });
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "p2p") == null);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try capabilities(&out.writer, .{ .version = "test", .p2p = "{\"driver\":\"580.65.06\",\"gpus\":[],\"pairs\":[]}" });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("580.65.06", parsed.value.object.get("p2p").?.object.get("driver").?.string);
}

test "--silence-mid-s is opt-in, needs the tail guard and reaches native open options" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    try std.testing.expectEqual(@as(?f64, null), (try parse(a, &.{"m"}, &u)).silence.mid_seconds);
    const args = try parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "10", "--silence-mid-s", "3", "--silence-mid-min-voiced-s", "1" }, &u);
    try std.testing.expectEqual(@as(?f64, 3), (try request(a, "m", "drama3", args)).silence.mid_seconds);
    try std.testing.expectEqual(@as(?f64, 1), (try request(a, "m", "drama3", args)).silence.mid_min_voiced_s);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "10", "--silence-mid-s", "3" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-mid-s", "3" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--silence-floor-dbfs=-80", "--silence-seconds", "3", "--silence-mid-s", "3" }, &u));
}

test "prompt precision: parsing and request forwarding cover defaults, explicit modes and refusals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: Usage = .{};
    for ([_][]const []const u8{ &.{"m"}, &.{ "m", "--prompt-precision", "bf16" }, &.{ "m", "--prompt-precision", "int8" } }, 0..) |argv, i| {
        const args = try parse(a, argv, &usage);
        try std.testing.expectEqualStrings(if (i == 2) "int8" else "bf16", args.prompt_precision);
        try std.testing.expectEqual(i == 2, (try request(a, "d", "t", args)).prompt_int8);
    }
    for ([_][]const []const u8{ &.{ "m", "--prompt-precision" }, &.{ "m", "--prompt-precision", "unknown" }, &.{ "m", "--prompt-precision", "fp4" } }) |argv|
        try std.testing.expectError(error.Usage, parse(a, argv, &usage));
}

test "prompt cache disk: new spellings ask the engine for exactly the legacy conversation store" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const legacy = try parse(a, &.{ "m", "--learn-conversations", "--learn-dir", "cache with spaces" }, &u);
    for ([_][]const []const u8{
        &.{ "m", "--prompt-cache-disk", "cache with spaces" },
        &.{ "m", "--prompt-cache-disk=cache with spaces" },
        &.{ "m", "--learn", "--prompt-cache-disk=cache with spaces" },
        &.{ "m", "--prompt-cache-disk=cache with spaces", "--learn" },
    }) |argv| {
        const named = try parse(a, argv, &u);
        try std.testing.expectEqualDeep(legacy, named);
        try std.testing.expectEqualDeep(try request(a, "d", "t", legacy), try request(a, "d", "t", named));
    }
    const defaults = try request(a, "d", "t", try parse(a, &.{"m"}, &u));
    try std.testing.expect(defaults.learn == null and !defaults.learn_conversations and defaults.spill == .off);
}

test "prompt cache disk: mixed spellings refuse both flag orders and name the conflict" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const named = [_][]const []const u8{ &.{ "--prompt-cache-disk", "cache" }, &.{"--prompt-cache-disk=cache"} };
    const legacy = [_][]const []const u8{
        &.{ "--learn-dir", "cache" }, &.{"--learn-dir=cache"},
        &.{ "--learn-dir", "other" }, &.{"--learn-dir=other"},
        &.{"--learn-conversations"},
    };
    for (named) |new| for (legacy) |old| {
        for ([_]bool{ false, true }) |reverse| {
            var argv: std.ArrayList([]const u8) = .empty;
            try argv.append(a, "m");
            try argv.appendSlice(a, if (reverse) old else new);
            try argv.appendSlice(a, if (reverse) new else old);
            try std.testing.expectError(error.Usage, parse(a, argv.items, &u));
            try std.testing.expect(std.mem.indexOf(u8, u.message, "--prompt-cache-disk") != null);
            const old_name = old[0][0 .. std.mem.indexOfScalar(u8, old[0], '=') orelse old[0].len];
            try std.testing.expect(std.mem.indexOf(u8, u.message, old_name) != null);
        }
    };
    for ([_][]const []const u8{
        &.{ "m", "--learn-conversations", "--learn-dir=cache", "--prompt-cache-disk=cache" },
        &.{ "m", "--prompt-cache-disk=cache", "--learn-conversations", "--learn-dir=cache" },
    }) |argv| try std.testing.expectError(error.Usage, parse(a, argv, &u));
}

test "prompt cache disk: repeated same spelling keeps last-value precedence and existing disk limits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    for ([_][]const []const u8{
        &.{ "m", "--prompt-cache-disk=old", "--prompt-cache-disk=new" },
        &.{ "m", "--learn-dir=old", "--learn-dir=new", "--learn-conversations" },
        &.{ "m", "--learn-conversations", "--learn-dir=old", "--learn-dir=new" },
    }) |argv| {
        const opened = try request(a, "d", "t", try parse(a, argv, &u));
        try std.testing.expectEqualStrings("new", opened.learn.?);
        try std.testing.expect(opened.learn_conversations);
    }
    var named = try parse(a, &.{ "m", "--prompt-cache-disk=cache", "--prompt-cache-gib=2", "--learn-gib=16", "--learn-daily-gib=3", "--learn-min-free-gib=1" }, &u);
    named.spill_gib = 8;
    try spillWith(&u, a, named);
    const opened = try request(a, "d", "t", named);
    try std.testing.expectEqualStrings("cache", opened.learn.?);
    try std.testing.expect(opened.learn_conversations and opened.spill == .finish and opened.learn_gib == 8);
    try std.testing.expect(opened.prompt_cache_gib == 2 and opened.learn_daily_gib == 3 and opened.learn_min_free_gib == 1);
}

test "prompt cache disk: missing and empty directories are usage errors and the flag is advertised" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    for ([_][]const []const u8{
        &.{ "m", "--prompt-cache-disk" },
        &.{ "m", "--prompt-cache-disk", "--no-thinking" },
        &.{ "m", "--prompt-cache-disk=" },
        &.{ "m", "--prompt-cache-disk", "" },
    }) |argv| {
        try std.testing.expectError(error.Usage, parse(a, argv, &u));
        try std.testing.expect(std.mem.indexOf(u8, u.message, "--prompt-cache-disk") != null);
    }
    const dash = try parse(a, &.{ "m", "--prompt-cache-disk=--cache" }, &u);
    try std.testing.expectEqualStrings("--cache", dash.learn_dir.?);
    var output: std.Io.Writer.Allocating = .init(a);
    try capabilities(&output.writer, .{ .version = "test" });
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "\"--prompt-cache-disk\": {}") != null);
}
