//! Which engine a qwen3_5 checkpoint opens on Macs: the 27B's, or the pinned 2B's.
const std = @import("std");
const tf = @import("tensorfold");
const Allocator = std.mem.Allocator;

/// The 27B engine, unless the checkpoint has the pinned 2B's hidden size or a tied head (the 2B's engine then).
pub fn owns(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return true;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return true;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return true;
    if (doc != .object) return true;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    for ([_]std.json.Value{ doc, text }) |o| {
        if (o.object.get("tie_word_embeddings")) |v| if (v == .bool and v.bool) return false;
        if (o.object.get("hidden_size")) |v| if (v == .integer and v.integer == tf.qwen35.config.hidden) return false;
    }
    return true;
}

/// True for looped SenseNova checkpoints (`sensenova_loop`).
pub fn isSenseNova(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "sensenova_loop");
}

/// Whether `dir` opens on the Qwen3.8-27B-class engine: the 27B-class qwen3_5 checkpoints, and Qwen3.5-MoE.
pub fn ownsQwen27(a: Allocator, io: std.Io, t: []const u8, dir: []const u8) bool {
    return (std.mem.eql(u8, t, "qwen3_5") and owns(a, io, dir)) or std.mem.eql(u8, t, "qwen3_5_moe");
}

/// The context a Qwen3.8-27B-class start serves (--context, else 32,768); null with `problem` set past 1 to 262,144.
pub fn qwen27Context(a: Allocator, context: ?i64, problem: *[]const u8) !?u32 {
    const n = context orelse 32768;
    if (n <= 0 or n > 262144) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this Qwen checkpoint's 1 to 262,144 tokens", .{n});
        return null;
    }
    return @intCast(n);
}

/// Validates and resolves `--context` for Qwen3.8-27B-class (`1..262144`) or `sensenova_loop` (config/YaRN ceiling).
pub fn laneContext(a: Allocator, io: std.Io, sn: bool, dir: []const u8, rope_yarn: ?f64, context: ?i64, problem: *[]const u8) !?u32 {
    if (!sn) return qwen27Context(a, context, problem);
    const n = context orelse 32768;
    if (n <= 0) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this SenseNova checkpoint's context window", .{n});
        return null;
    }
    if (!try senseNovaContextOk(a, io, dir, rope_yarn, n, problem)) return null;
    return @intCast(n);
}

/// Validates a `sensenova_loop` context request against the checkpoint's resolved base or YaRN ceiling.
pub fn senseNovaContextOk(a: Allocator, io: std.Io, dir: []const u8, rope_yarn: ?f64, context: i64, problem: *[]const u8) !bool {
    const max_ctx: i64 = @intCast(tf.qwen27.config.senseNovaMaxContext(a, io, dir, rope_yarn) catch |err| {
        problem.* = switch (err) {
            error.UnsupportedRopeScaling, error.RopeYarnFactor, error.RopeYarnConflict => "--rope-yarn / rope_parameters: static YaRN (factor > 1) is supported on looped SenseNova checkpoints without conflicting factors",
            else => try std.fmt.allocPrint(a, "the native SenseNova engine cannot load {s} ({s})", .{ dir, @errorName(err) }),
        };
        return false;
    });
    if (context <= max_ctx) return true;
    problem.* = try std.fmt.allocPrint(a, "--context {d} is outside this SenseNova checkpoint's 1 to {d} tokens", .{ context, max_ctx });
    return false;
}

test "a qwen3_5 checkpoint opens the 27B engine unless it has the 2B's geometry or a tied head" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(owns(a, io, dir)); // no config.json: the 27B's loader explains the refusal
    const cases = [_]struct { config: []const u8, big: bool }{
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{\"tie_word_embeddings\":false}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":4096}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":2048}}", .big = false },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        try std.testing.expectEqual(c.big, owns(a, io, dir));
    }
}
