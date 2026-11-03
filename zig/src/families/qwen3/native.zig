//! Qwen3 dense checkpoints for the native HIP server: the head the folder declares, sized to the memory budget.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const api = @import("engine_api");
const engine = @import("engine.zig");
const prompt = @import("prompt.zig");
const score = @import("score.zig");

const Engine = engine.Engine;
const Allocator = std.mem.Allocator;

/// The registry entry: `native/hip.zig` opens `qwen3` checkpoints through it.
pub const model_type = "qwen3";
pub const formats: []const []const u8 = &.{"bf16"};
/// The window when neither --context nor the checkpoint's config names one (the Qwen3 8B serving default).
pub const default_context: i64 = 32768;
/// The engine takes a stream's whole prompt in one step.
pub const prefill_step: u32 = 0;
pub const slide_refusal = "--slide: Qwen3 dense checkpoints do not learn on HIP";
pub const confidence_refusal = "--mtp-confidence: Qwen3 dense checkpoints have no draft head";
/// An embedding engine writes no reply text, so no reasoning block (Info.reasoning).
pub const reasoning = false;
/// Every product is a BF16 dense one, which gfx12's WMMA tile runs: matrix=on is admitted on gfx12.
pub const matrix_wmma12 = true;
/// Checked by native/hip.zig before any rank joins a group.
pub const tp_refusal = "--tp: Qwen3 dense checkpoints run on one HIP GPU; serve with --tp 1";
pub const open = openEngine;
pub const follow = followRank;

/// What the server drives, as the other HIP families' `Loaded`, plus what this engine serves.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
    /// No chat or completions: the server refuses the generation routes.
    text_generation: bool = false,
    embeddings: ?api.EmbeddingInfo = null,
    /// The window the engine was sized to (the asked one, or less where the budget shrank it).
    window: ?usize = null,
    /// A reranker: the engine scores labels (the score hook) and serves no embeddings.
    decides: bool = false,
    /// The reranker's label tokens, from `1_LogitScore/config.json`.
    rerank: ?api.Rerank = null,
};

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
        "qwen3 on HIP serves sentence-transformers embedding checkpoints: {s} has no 1_Pooling/config.json",
        .{dir},
    );
    if (pooling != .object) return refuse(a, problem, "1_Pooling/config.json is not a JSON object", .{});
    for (pooling_modes) |mode| {
        const on = if (pooling.object.get(mode)) |v| v == .bool and v.bool else false;
        const want = std.mem.eql(u8, mode, "pooling_mode_lasttoken");
        if (on != want) return refuse(a, problem, "1_Pooling/config.json: {s} is {s}; qwen3 on HIP pools the last token only", .{ mode, if (on) "on" else "off" });
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

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The window the budget holds: `asked`, halved while its scratch passes `room`, down to `engine.least_window`.
pub fn fitWindow(asked: usize, room: usize, ctx: anytype, bytesFor: fn (@TypeOf(ctx), usize) anyerror!usize) !?usize {
    var window = asked;
    while (true) {
        if (try bytesFor(ctx, window) <= room) return window;
        if (window <= engine.least_window) return null;
        window = @max(window / 2, engine.least_window);
    }
}

fn scratchOf(e: *Engine, window: usize) anyerror!usize {
    return e.scratchBytes(window);
}

const Owned = struct {
    gpa: Allocator,
    e: *Engine,
    p: prompt.Prompt,
    startup: []const u8,
    /// The checkpoint folder, kept for the reranker's head rows.
    dir: []const u8,
};

fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.p.deinit();
    own.e.deinit();
    own.gpa.free(own.startup);
    own.gpa.free(own.dir);
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`, its head from the folder and its window from the memory budget.
fn openEngine(a: Allocator, gpa: Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, problem: *[]const u8) anyerror!Loaded {
    if (o.world > 1) return refuse(a, problem, "--tp {d}: Qwen3 dense checkpoints run on one HIP GPU", .{o.world});
    // the BF16 weights run on the BF16 dot2 GEMM tile: gfx11 and gfx12, not an FP16 build (RDNA2)
    if (dev.caps.act != .bf16) return refuse(a, problem, "Qwen3 dense checkpoints on HIP need a GPU with BF16 dot2 (gfx11 or gfx12); {s} runs FP16 activations", .{hip.kernels.arch});
    const head = try headOf(a, io, dir, problem);
    const e = Engine.load(gpa, io, dir, dev.index, dev.policy.choices()) catch |err|
        return refuse(a, problem, "the native HIP engine cannot load {s} ({s})", .{ dir, @errorName(err) });
    errdefer e.deinit();
    const pool = hip.admission.budget(gpa, io, e.ctx) catch |err|
        return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    const room = pool.room(e.weights);
    const window = (fitWindow(o.window, room, e, scratchOf) catch |err|
        return refuse(a, problem, "the native HIP engine cannot count its scratch ({s})", .{@errorName(err)})) orelse {
        const least = e.scratchBytes(@min(o.window, engine.least_window)) catch 0;
        return refuse(a, problem, "the HIP memory budget fits no {d}-token pass: it takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; free device memory or lower --context", .{ @min(o.window, engine.least_window), gibs(least), gibs(room), gibs(e.weights), gibs(pool.reserve) });
    };
    e.size(window) catch |err|
        return refuse(a, problem, "the native HIP engine cannot allocate its scratch ({s})", .{@errorName(err)});
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const normalize = switch (head) {
        .embedding => |h| h.normalize,
        .rerank => false,
    };
    const labels: ?api.Rerank = switch (head) {
        .embedding => null,
        .rerank => |l| l,
    };
    if (labels) |l| if (l.yes >= e.m.spec.vocab or l.no >= e.m.spec.vocab)
        return refuse(a, problem, "1_LogitScore/config.json: tokens {d} and {d} are not both in this {d}-token vocabulary", .{ l.yes, l.no, e.m.spec.vocab });
    const kept = try gpa.dupe(u8, dir);
    errdefer gpa.free(kept);
    own.* = .{ .gpa = gpa, .e = e, .p = prompt.Prompt.init(gpa, e, normalize), .startup = undefined, .dir = kept };
    errdefer own.p.deinit();
    if (labels) |l| {
        own.p.heads = score.Heads.of(&e.m, io, own.dir);
        own.p.table.score = prompt.Prompt.scoreLabels;
        // both label rows read at load: a bad head fails startup, and the first request reads nothing
        for ([_]u32{ l.yes, l.no }) |id| _ = own.p.heads.?.row(gpa, id) catch |err|
            return refuse(a, problem, "the native HIP engine cannot read lm_head row {d} ({s})", .{ id, @errorName(err) });
    }
    const shrunk: []const u8 = if (window < o.window) " (shrunk from the asked window to fit the budget)" else "";
    own.startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}: qwen3 {s}, model {d:.2} GiB; {d}-token passes{s}, {d:.2} GiB scratch, of {d:.1} GiB left after a {d:.1} GiB reserve; {s}", .{
        hip.kernels.arch, dev.index, if (head == .rerank) "reranker" else "embedding", gibs(e.weights), window, shrunk, gibs(e.scratchBytes(window) catch 0), gibs(room), gibs(pool.reserve),
        switch (head) {
            .embedding => |h| if (h.normalize) "vectors unit-normed" else "vectors as pooled",
            .rerank => "label scores from lm_head rows read at load, on the host",
        },
    });
    const spec = e.m.spec;
    return .{
        .backend = own.p.backend(),
        .facts = .{},
        .rows = 1,
        .ctx = own,
        .deinit = release,
        .streams = 1,
        .startup = own.startup,
        .text_generation = false,
        .embeddings = if (head == .rerank) null else .{ .dimensions = @intCast(spec.hidden), .vocabulary = @intCast(spec.vocab), .max_batch_tokens = @intCast(window), .max_inputs = @intCast(engine.max_inputs), .mrl_min = @min(32, @as(u32, @intCast(spec.hidden))) },
        .window = window,
        .decides = head == .rerank,
        .rerank = labels,
    };
}

/// Ranks above 0 have nothing to hold: the engine runs on one GPU.
fn followRank(a: Allocator, _: Allocator, _: std.Io, _: hip.Device, _: []const u8, o: anytype, problem: *[]const u8) anyerror!void {
    return refuse(a, problem, "--tp {d}: Qwen3 dense checkpoints run on one HIP GPU", .{o.world});
}

test "the window halves to fit the room, down to the least window" {
    const Count = struct {
        per: usize,
        fn bytes(c: @This(), window: usize) anyerror!usize {
            return window * c.per;
        }
    };
    const c: Count = .{ .per = 10 };
    try std.testing.expectEqual(@as(?usize, 32768), try fitWindow(32768, 327680, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 8192), try fitWindow(32768, 100000, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 2048), try fitWindow(32768, 20480, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, null), try fitWindow(32768, 20479, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, 1000), try fitWindow(1000, 10000, c, Count.bytes));
    try std.testing.expectEqual(@as(?usize, null), try fitWindow(1000, 9999, c, Count.bytes));
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
