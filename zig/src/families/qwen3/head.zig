//! What a Qwen3 checkpoint folder serves, read from its sentence-transformers files, and the window a budget holds.
const std = @import("std");
const api = @import("engine_api");

const Allocator = std.mem.Allocator;

pub const Refused = error{Refused};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The head a checkpoint folder declares.
pub const Head = union(enum) {
    /// sentence-transformers last-token pooling, unit-normed when modules.json lists Normalize.
    embedding: struct { normalize: bool },
    /// A reranker: `1_LogitScore` names the yes and no tokens its last row's logits score.
    rerank: api.Rerank,
};

const pooling_modes = [_][]const u8{
    "pooling_mode_cls_token",            "pooling_mode_mean_tokens",         "pooling_mode_max_tokens",
    "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken",
};

/// The head from `1_Pooling/config.json` and `modules.json`, or a refusal naming what the folder lacks.
pub fn headOf(a: Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) (Refused || Allocator.Error)!Head {
    if (try readJson(a, io, dir, "1_LogitScore/config.json", problem)) |v| return .{ .rerank = try labelsOf(a, v, problem) };
    const pooling = try readJson(a, io, dir, "1_Pooling/config.json", problem) orelse return refuse(
        a,
        problem,
        "qwen3 serves sentence-transformers embedding checkpoints: {s} has no 1_Pooling/config.json",
        .{dir},
    );
    if (pooling != .object) return refuse(a, problem, "1_Pooling/config.json is not a JSON object", .{});
    for (pooling_modes) |mode| {
        const on = if (pooling.object.get(mode)) |v| v == .bool and v.bool else false;
        const want = std.mem.eql(u8, mode, "pooling_mode_lasttoken");
        if (on != want) return refuse(a, problem, "1_Pooling/config.json: {s} is {s}; qwen3 pools the last token only", .{ mode, if (on) "on" else "off" });
    }
    // no modules.json: the pooled rows as they are; one that is there must be a list of modules with string types
    var normalize = false;
    if (try readJson(a, io, dir, "modules.json", problem)) |mods| {
        if (mods != .array) return refuse(a, problem, "modules.json is not a JSON list of modules", .{});
        for (mods.array.items) |m| {
            const kind = if (m == .object) m.object.get("type") else null;
            if (kind == null or kind.? != .string) return refuse(a, problem, "modules.json: a module without a string \"type\"", .{});
            if (std.mem.eql(u8, kind.?.string, "sentence_transformers.models.Normalize")) normalize = true;
        }
    }
    return .{ .embedding = .{ .normalize = normalize } };
}

/// The yes and no tokens `1_LogitScore/config.json` names (the vocabulary bound is checked once the config is read).
fn labelsOf(a: Allocator, v: std.json.Value, problem: *[]const u8) (Refused || Allocator.Error)!api.Rerank {
    const name = "1_LogitScore/config.json";
    if (v != .object) return refuse(a, problem, "{s} is not a JSON object", .{name});
    var ids: [2]u32 = undefined;
    for ([_][]const u8{ "true_token_id", "false_token_id" }, &ids) |key, *id| {
        const x = v.object.get(key) orelse return refuse(a, problem, "{s} has no {s}", .{ name, key });
        if (x != .integer or x.integer < 0 or x.integer > std.math.maxInt(u32))
            return refuse(a, problem, "{s}: {s} is not a token id", .{ name, key });
        id.* = @intCast(x.integer);
    }
    if (ids[0] == ids[1]) return refuse(a, problem, "{s} names token {d} for both answers", .{ name, ids[0] });
    return .{ .yes = ids[0], .no = ids[1] };
}

/// The folder's JSON file `name`, null when it is absent; unreadable or malformed JSON is refused.
fn readJson(a: Allocator, io: std.Io, dir: []const u8, name: []const u8, problem: *[]const u8) (Refused || Allocator.Error)!?std.json.Value {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch |err| switch (err) {
        error.FileNotFound => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(a, problem, "{s}: cannot read it ({s})", .{ name, @errorName(err) }),
    };
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => refuse(a, problem, "{s} is not valid JSON ({s})", .{ name, @errorName(err) }),
    };
}

/// The window the budget holds: `asked`, halved while its scratch passes `room`, down to `least`.
pub fn fitWindow(asked: usize, room: usize, least: usize, ctx: anytype, bytesFor: fn (@TypeOf(ctx), usize) anyerror!usize) !?usize {
    var window = asked;
    while (true) {
        if (try bytesFor(ctx, window) <= room) return window;
        if (window <= least) return null;
        window = @max(window / 2, least);
    }
}

test "the window halves to fit the room, down to the least window" {
    const Count = struct {
        per: usize,
        fn bytes(c: @This(), window: usize) anyerror!usize {
            return window * c.per;
        }
    };
    const c: Count = .{ .per = 10 };
    try std.testing.expectEqual(@as(?usize, 32768), try fitWindow(32768, 327680, 2048, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 8192), try fitWindow(32768, 100000, 2048, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 2048), try fitWindow(32768, 20480, 2048, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, null), try fitWindow(32768, 20479, 2048, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 1000), try fitWindow(1000, 10000, 2048, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, null), try fitWindow(1000, 9999, 2048, c, Count.bytes));
}

test "a pooling folder declares the embedding head: last token only, normalized when modules.json says so" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "1_Pooling") != null);
    try tmp.dir.createDirPath(io, "1_Pooling");
    const last =
        \\{"word_embedding_dimension": 4096, "pooling_mode_cls_token": false, "pooling_mode_mean_tokens": false,
        \\ "pooling_mode_max_tokens": false, "pooling_mode_mean_sqrt_len_tokens": false,
        \\ "pooling_mode_weightedmean_tokens": false, "pooling_mode_lasttoken": true, "include_prompt": true}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "1_Pooling/config.json", .data = last });
    try std.testing.expect(!(try headOf(a, io, dir, &problem)).embedding.normalize);
    try tmp.dir.writeFile(io, .{ .sub_path = "modules.json", .data = "[{\"type\": \"sentence_transformers.models.Pooling\"}, {\"type\": \"sentence_transformers.models.Normalize\"}]" });
    try std.testing.expect((try headOf(a, io, dir, &problem)).embedding.normalize);
    try tmp.dir.writeFile(io, .{ .sub_path = "modules.json", .data = "[{\"type\": \"sentence_transformers.models.Normalize\"" });
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "not valid JSON") != null);
    try tmp.dir.writeFile(io, .{ .sub_path = "modules.json", .data = "{\"type\": \"sentence_transformers.models.Normalize\"}" });
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "modules.json", .data = "[{\"path\": \"1_Pooling\"}]" });
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "modules.json", .data = "[]" });
    try tmp.dir.writeFile(io, .{ .sub_path = "1_Pooling/config.json", .data = "{\"pooling_mode_lasttoken\": tru" });
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "1_Pooling/config.json", .data = "{\"pooling_mode_mean_tokens\": true, \"pooling_mode_lasttoken\": true}" });
    try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "pooling_mode_mean_tokens") != null);
}

test "a 1_LogitScore folder declares the reranker head, before any pooling, and a malformed one is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    try tmp.dir.createDirPath(io, "1_LogitScore");
    try tmp.dir.writeFile(io, .{ .sub_path = "1_LogitScore/config.json", .data = "{\"true_token_id\": 9693, \"false_token_id\": 2152}" });
    const labels = (try headOf(a, io, dir, &problem)).rerank;
    try std.testing.expectEqual(api.Rerank{ .yes = 9693, .no = 2152 }, labels);
    // the reranker's folder also carries no pooling; one that did would still be a reranker by its scores
    try tmp.dir.createDirPath(io, "1_Pooling");
    try tmp.dir.writeFile(io, .{ .sub_path = "1_Pooling/config.json", .data = "{\"pooling_mode_lasttoken\": true}" });
    try std.testing.expect(try headOf(a, io, dir, &problem) == .rerank);
    const bad = [_][]const u8{
        "{not json",
        "[9693, 2152]",
        "{\"false_token_id\": 2152}",
        "{\"true_token_id\": 9693}",
        "{\"true_token_id\": \"9693\", \"false_token_id\": 2152}",
        "{\"true_token_id\": 9693.5, \"false_token_id\": 2152}",
        "{\"true_token_id\": -1, \"false_token_id\": 2152}",
        "{\"true_token_id\": 4294967296, \"false_token_id\": 2152}",
        "{\"true_token_id\": 2152, \"false_token_id\": 2152}",
    };
    for (bad) |text| {
        problem = "";
        try tmp.dir.writeFile(io, .{ .sub_path = "1_LogitScore/config.json", .data = text });
        try std.testing.expectError(error.Refused, headOf(a, io, dir, &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem, "1_LogitScore/config.json") != null);
    }
}
