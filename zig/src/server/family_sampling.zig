//! Served sampling: family defaults, generation_config.json, the flags; and --instruct-sampling's thinking-off set.
const std = @import("std");
const json = @import("json");

/// The serve flags' sampling values; each set one replaces what the layers below gave.
pub const Flags = struct {
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
};

/// The request fields --instruct-sampling may name.
pub const instruct_fields = [_][]const u8{ "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty" };

/// DeepSeek-V4.x: temperature 1.0, top_p 1.0, no top_k (the V4 and V4.1 model cards' recommended sampling).
pub fn put(a: std.mem.Allocator, out: *json.Object, model_type: []const u8) !void {
    if (!std.mem.eql(u8, model_type, "deepseek_v41") and !std.mem.eql(u8, model_type, "deepseek_v4")) return;
    try out.put(a, "temperature", .{ .float = 1.0 });
    try out.put(a, "top_p", .{ .float = 1.0 });
}

/// Qwen 3.5, 3.6, Qwen3.8-27B and Flash-Next cards: a thinking set (generation_config.json) and an instruct set.
pub fn hasInstruct(model_type: []const u8) bool {
    for ([_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text" }) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// The Qwen cards' instruct set: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 where penalties apply.
pub fn putInstruct(a: std.mem.Allocator, out: *json.Object, model_type: []const u8, penalties: bool) !bool {
    if (!hasInstruct(model_type)) return false;
    try out.put(a, "temperature", .{ .float = 0.7 });
    try out.put(a, "top_p", .{ .float = 0.8 });
    try out.put(a, "top_k", try json.intValue(a, 20));
    if (penalties) try out.put(a, "presence_penalty", .{ .float = 1.5 });
    return true;
}

/// The three layers over each other (``do_sample`` false is greedy); null when none sets a value.
pub fn resolve(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags) !?json.Value {
    const out = try json.newObject(a);
    try base(a, out, model_type, generation_config);
    try applyFlags(a, out, flags);
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

/// True when generation_config.json says ``do_sample: false``: the checkpoint decodes greedily.
pub fn greedyConfig(a: std.mem.Allocator, generation_config: ?[]const u8) !bool {
    const bytes = generation_config orelse return false;
    const parsed = try json.parse(a, bytes);
    if (parsed != .ok) return false;
    const d = parsed.ok.get("do_sample") orelse return false;
    return d == .bool and !d.bool;
}

/// --instruct-sampling's set: the layers, the card set (``card``, or under pairs), then the flags; ``none``: null.
pub fn resolveInstruct(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags, penalties: bool, text: []const u8) !?json.Value {
    const word = std.mem.trim(u8, text, " ");
    if (std.mem.eql(u8, word, "none")) return null;
    const card = std.mem.eql(u8, word, "card");
    if (card and !hasInstruct(model_type)) return error.NoInstructSet;
    const out = try json.newObject(a);
    try base(a, out, model_type, generation_config);
    _ = try putInstruct(a, out, model_type, penalties);
    if (!card) if (try readPairs(a, out, word)) |_| return error.InvalidInstructSampling;
    try applyFlags(a, out, flags);
    return json.Value{ .object = out };
}

/// What the server does with ``--instruct-sampling`` at startup (`startup`).
pub const Startup = struct {
    /// The set thinking-off sampled chats start from; null: none is served.
    set: ?json.Value = null,
    /// A startup line about what is not served, else null.
    note: ?[]const u8 = null,
    /// Why the server refuses to start, else null.
    problem: ?[]const u8 = null,
};

/// ``--instruct-sampling text`` at startup: the set and a note, or the problem that refuses the start.
pub fn startup(a: std.mem.Allocator, model_type: []const u8, generation_config: ?[]const u8, flags: Flags, penalties: bool, text: []const u8) !Startup {
    if (try greedyConfig(a, generation_config))
        return .{ .note = "--instruct-sampling is not served: generation_config.json has do_sample false, so replies decode greedily" };
    const set = resolveInstruct(a, model_type, generation_config, flags, penalties, text) catch |e| switch (e) {
        error.NoInstructSet => return .{ .problem = try std.fmt.allocPrint(a, "--instruct-sampling card: model type {s} has no instruct set in its family defaults; give name=value pairs", .{model_type}) },
        error.InvalidInstructSampling => return .{ .problem = try std.fmt.allocPrint(a, "argument --instruct-sampling: {s}: '{s}'", .{ (try pairsProblem(a, text)).?, text }) },
        else => |x| return x,
    } orelse return .{};
    if (penalties) return .{ .set = set };
    for ([_][]const u8{ "presence_penalty", "frequency_penalty" }) |k| if (set.get(k)) |v| if ((v.float64() orelse 0) != 0)
        return .{ .problem = try std.fmt.allocPrint(a, "--instruct-sampling names {s} {d}, but this engine applies no penalties: name 0 or leave it out", .{ k, v.float64().? }) };
    const note: ?[]const u8 = if (hasInstruct(model_type) and std.mem.indexOf(u8, text, "presence_penalty") == null)
        "presence_penalty 1.5 from the model card's instruct set is not served: this engine applies no penalties"
    else
        null;
    return .{ .set = set, .note = note };
}

/// ``--instruct-sampling``'s text checked: null when it reads, else what is wrong with it, for the argument's refusal.
pub fn pairsProblem(a: std.mem.Allocator, text: []const u8) error{OutOfMemory}!?[]const u8 {
    const word = std.mem.trim(u8, text, " ");
    if (std.mem.eql(u8, word, "none") or std.mem.eql(u8, word, "card")) return null;
    const out = try json.newObject(a);
    return readPairs(a, out, word);
}

const pairs_form = "expected none, card, or comma-separated name=value pairs naming temperature, top_p, top_k, min_p, presence_penalty or frequency_penalty";

/// Comma-separated ``name=value`` instruct pairs into ``out``, each in a request's range: null, or the problem.
pub fn readPairs(a: std.mem.Allocator, out: *json.Object, text: []const u8) error{OutOfMemory}!?[]const u8 {
    var any = false;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |raw| {
        const pair = std.mem.trim(u8, raw, " ");
        if (pair.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return pairs_form;
        const name = std.mem.trim(u8, pair[0..eq], " ");
        const value = std.mem.trim(u8, pair[eq + 1 ..], " ");
        const known = for (instruct_fields) |f| {
            if (std.mem.eql(u8, f, name)) break true;
        } else false;
        if (!known) return pairs_form;
        if (std.mem.eql(u8, name, "top_k")) {
            const k = std.fmt.parseInt(i64, value, 10) catch return "top_k must be an integer";
            if (k < 0) return "top_k must be 0 (off) or more";
            try out.put(a, name, try json.intValue(a, k));
        } else {
            const x = std.fmt.parseFloat(f64, value) catch return try std.fmt.allocPrint(a, "{s} must be a finite number", .{name});
            if (!std.math.isFinite(x)) return try std.fmt.allocPrint(a, "{s} must be a finite number", .{name});
            if (std.mem.eql(u8, name, "temperature") and x < 0) return "temperature must be 0 (greedy) or more";
            if (std.mem.eql(u8, name, "top_p") and !(x > 0 and x <= 1)) return "top_p must be above 0 and at most 1";
            if (std.mem.eql(u8, name, "min_p") and !(x >= 0 and x <= 1)) return "min_p must be between 0 and 1";
            if (std.mem.endsWith(u8, name, "_penalty") and !(x >= -2 and x <= 2)) return try std.fmt.allocPrint(a, "{s} must be between -2 and 2", .{name});
            try out.put(a, name, .{ .float = x });
        }
        any = true;
    }
    return if (any) null else pairs_form;
}

/// Family defaults, then generation_config.json's sampling (``do_sample`` false: temperature 0).
fn base(a: std.mem.Allocator, out: *json.Object, model_type: []const u8, generation_config: ?[]const u8) !void {
    try put(a, out, model_type);
    if (generation_config) |bytes| if ((try json.parse(a, bytes)) == .ok) {
        const cfg = (try json.parse(a, bytes)).ok;
        for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
        if (cfg.get("do_sample")) |d| if (d == .bool) {
            if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
        };
    };
}

/// The serve flags over everything below them.
fn applyFlags(a: std.mem.Allocator, out: *json.Object, flags: Flags) !void {
    if (flags.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (flags.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (flags.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (flags.min_p) |t| try out.put(a, "min_p", .{ .float = t });
}

test "DeepSeek-V4.x serves its cards' sampling defaults; other families keep none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "deepseek_v41", "deepseek_v4" }) |kind| {
        const out = (try resolve(a, kind, null, .{})).?;
        try std.testing.expectEqual(@as(f64, 1.0), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(f64, 1.0), out.get("top_p").?.float);
        try std.testing.expect(out.get("top_k") == null);
    }
    try std.testing.expect((try resolve(a, "glm5_next", null, .{})) == null);
}

test "family sampling defaults sit under generation_config.json and the serve flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6, \"top_k\": 40}", .{});
    try std.testing.expectEqual(@as(f64, 0.6), cfg.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), cfg.?.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 1.0), cfg.?.get("top_p").?.float);
    const flag = try resolve(a, "deepseek_v41", "{\"temperature\": 0.6}", .{ .temperature = 0.3, .top_p = 0.9 });
    try std.testing.expectEqual(@as(f64, 0.3), flag.?.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0.9), flag.?.get("top_p").?.float);
    const greedy = try resolve(a, "deepseek_v41", "{\"do_sample\": false}", .{});
    try std.testing.expectEqual(@as(f64, 0), greedy.?.get("temperature").?.float);
}

test "--instruct-sampling card serves the Qwen cards' instruct set under the flags; pairs edit it; none serves none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"do_sample\": true, \"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95}";
    for ([_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text" }) |kind| {
        const thinking = (try resolve(a, kind, cfg, .{})).?;
        try std.testing.expectEqual(@as(f64, 1.0), thinking.get("temperature").?.float);
        try std.testing.expect(thinking.get("presence_penalty") == null);
        const out = (try resolveInstruct(a, kind, cfg, .{}, true, "card")).?;
        try std.testing.expectEqual(@as(f64, 0.7), out.get("temperature").?.float);
        try std.testing.expectEqual(@as(f64, 0.8), out.get("top_p").?.float);
        try std.testing.expectEqual(@as(?i64, 20), out.get("top_k").?.int64());
        try std.testing.expectEqual(@as(f64, 1.5), out.get("presence_penalty").?.float);
        try std.testing.expect(out.get("frequency_penalty") == null);
    }
    // an engine without penalties gets the card set without presence_penalty
    const plain = (try resolveInstruct(a, "qwen3_5", cfg, .{}, false, " card ")).?;
    try std.testing.expectEqual(@as(f64, 0.7), plain.get("temperature").?.float);
    try std.testing.expect(plain.get("presence_penalty") == null);
    // the serve flags apply over the card set
    const flagged = (try resolveInstruct(a, "qwen3_5", cfg, .{ .temperature = 0, .top_k = 40 }, true, "card")).?;
    try std.testing.expectEqual(@as(f64, 0), flagged.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), flagged.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 1.5), flagged.get("presence_penalty").?.float);
    // pairs edit the card set and sit below the flags, so a server started greedy stays greedy
    const pairs = (try resolveInstruct(a, "qwen3_5", cfg, .{}, true, "temperature=0.6, presence_penalty=0,top_k=10")).?;
    try std.testing.expectEqual(@as(f64, 0.6), pairs.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0), pairs.get("presence_penalty").?.float);
    try std.testing.expectEqual(@as(?i64, 10), pairs.get("top_k").?.int64());
    try std.testing.expectEqual(@as(f64, 0.8), pairs.get("top_p").?.float);
    const under = (try resolveInstruct(a, "qwen3_5", cfg, .{ .temperature = 0, .top_k = 40 }, true, "temperature=0.6,top_k=10")).?;
    try std.testing.expectEqual(@as(f64, 0), under.get("temperature").?.float);
    try std.testing.expectEqual(@as(?i64, 40), under.get("top_k").?.int64());
    // none serves no set; a family without a card set refuses card but takes pairs over its defaults
    try std.testing.expect((try resolveInstruct(a, "qwen3_5", cfg, .{}, true, "none")) == null);
    try std.testing.expectError(error.NoInstructSet, resolveInstruct(a, "deepseek_v41", null, .{}, true, "card"));
    try std.testing.expectError(error.NoInstructSet, resolveInstruct(a, "glm5_next", cfg, .{}, true, "card"));
    const given = (try resolveInstruct(a, "glm5_next", cfg, .{}, true, "presence_penalty=1")).?;
    try std.testing.expectEqual(@as(f64, 1.0), given.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 1.0), given.get("presence_penalty").?.float);
    // a checkpoint that decodes greedily
    try std.testing.expect(try greedyConfig(a, "{\"do_sample\": false}"));
    try std.testing.expect(!try greedyConfig(a, cfg));
    try std.testing.expect(!try greedyConfig(a, null));
    // malformed text and out-of-range values are refused by name
    for ([_][]const u8{ "", " ", "temperature", "temperature=x", "foo=1", "top_k=1.5", "temperature=inf", "=0.5", "cards" }) |bad| {
        try std.testing.expectError(error.InvalidInstructSampling, resolveInstruct(a, "qwen3_5", cfg, .{}, true, bad));
        try std.testing.expect((try pairsProblem(a, bad)) != null);
    }
    const ranges = [_][2][]const u8{
        .{ "temperature=-1", "temperature must be 0 (greedy) or more" },
        .{ "top_p=0", "top_p must be above 0 and at most 1" },
        .{ "top_p=1.5", "top_p must be above 0 and at most 1" },
        .{ "top_k=-1", "top_k must be 0 (off) or more" },
        .{ "min_p=2", "min_p must be between 0 and 1" },
        .{ "presence_penalty=3", "presence_penalty must be between -2 and 2" },
        .{ "frequency_penalty=-2.5", "frequency_penalty must be between -2 and 2" },
        .{ "temperature=0.7,top_p=nan", "top_p must be a finite number" },
        .{ "foo=1", pairs_form },
    };
    for (ranges) |r| {
        try std.testing.expectEqualStrings(r[1], (try pairsProblem(a, r[0])).?);
        try std.testing.expectError(error.InvalidInstructSampling, resolveInstruct(a, "qwen3_5", cfg, .{}, true, r[0]));
    }
    for ([_][]const u8{ "none", "card", "top_k=0", "temperature=0", "top_p=1", "temperature=0.7,top_p=0.8,top_k=20,min_p=0,presence_penalty=1.5,frequency_penalty=0", "presence_penalty=-2,frequency_penalty=2" }) |good| {
        try std.testing.expect((try pairsProblem(a, good)) == null);
    }
}

test "--instruct-sampling at startup: the set and its notes, or the problem that refuses the start" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"do_sample\": true, \"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95}";
    // card on an engine with penalties: the whole set, nothing to note
    const served = try startup(a, "qwen3_5", cfg, .{}, true, "card");
    try std.testing.expectEqual(@as(f64, 1.5), served.set.?.get("presence_penalty").?.float);
    try std.testing.expect(served.note == null and served.problem == null);
    // a checkpoint whose generation_config.json has do_sample false leaves the flag unserved, with a line
    const greedy = try startup(a, "qwen3_5", "{\"do_sample\": false}", .{}, true, "card");
    try std.testing.expect(greedy.set == null and greedy.problem == null);
    try std.testing.expect(std.mem.indexOf(u8, greedy.note.?, "do_sample false") != null);
    // card on an engine without penalties: the set without presence_penalty, with a line
    const plain = try startup(a, "qwen3_5", cfg, .{}, false, "card");
    try std.testing.expectEqual(@as(f64, 0.7), plain.set.?.get("temperature").?.float);
    try std.testing.expect(plain.set.?.get("presence_penalty") == null);
    try std.testing.expect(std.mem.indexOf(u8, plain.note.?, "presence_penalty 1.5") != null);
    try std.testing.expect(plain.problem == null);
    // pairs that name the penalty 0 on that engine: no line
    const zeroed = try startup(a, "qwen3_5", cfg, .{}, false, "presence_penalty=0");
    try std.testing.expect(zeroed.set != null and zeroed.note == null and zeroed.problem == null);
    // pairs naming a non-zero penalty on that engine refuse the start, naming the pair
    const refused = try startup(a, "qwen3_5", cfg, .{}, false, "temperature=0.6,frequency_penalty=0.5");
    try std.testing.expect(refused.set == null);
    try std.testing.expectEqualStrings("--instruct-sampling names frequency_penalty 0.5, but this engine applies no penalties: name 0 or leave it out", refused.problem.?);
    // card on a model type without a card set
    const none = try startup(a, "glm5_next", cfg, .{}, true, "card");
    try std.testing.expect(std.mem.indexOf(u8, none.problem.?, "glm5_next has no instruct set") != null);
    // none serves nothing, quietly
    const off = try startup(a, "qwen3_5", cfg, .{}, true, "none");
    try std.testing.expect(off.set == null and off.note == null and off.problem == null);
    // an out-of-range pair that reached startup is refused as the argument is
    const range = try startup(a, "qwen3_5", cfg, .{}, true, "min_p=2");
    try std.testing.expectEqualStrings("argument --instruct-sampling: min_p must be between 0 and 1: 'min_p=2'", range.problem.?);
}
