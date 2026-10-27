//! The control vector's serve checks: the file, the reader, the family and the line that reports what is in force.

const std = @import("std");
const builtin = @import("builtin");
const glp = @import("engine_api").glp;
const serve = @import("glp_startup.zig");
const log = @import("log.zig");

const io = std.testing.io;

/// The width Qwen3.8-27B has; a vector that steers it is this wide.
const width = 5120;

/// Whether this platform's engine steers: only the Metal engine does, and only on a Mac.
const steers = builtin.os.tag == .macos;

/// The lines the server printed, for the report the operator reads.
const Sink = struct {
    list: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,
};

fn capture(ctx: *anyopaque, text: []const u8) void {
    const sink: *Sink = @ptrCast(@alignCast(ctx));
    sink.list.appendSlice(sink.a, text) catch {};
}

/// A checkpoint folder with a config.json, a vector file beside it and the vector read.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    /// The model's folder, as the serve path names it.
    model: []const u8,
    /// The vector the file holds, read once.
    g: ?serve.Vec,

    fn make(a: std.mem.Allocator, w: usize, hidden: usize, ids: []const u32, spec: glp.WriteSpec, alpha: ?f64) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const model = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
        const json = try std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5\",\"hidden_size\":{d},\"text_config\":{{\"hidden_size\":{d}}}}}", .{ hidden, hidden });
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        const bytes = try vectorBytes(a, w, ids, spec);
        try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = bytes });
        const path = try std.fs.path.join(a, &.{ model, "vector.gguf" });
        var g: ?serve.Vec = serve.withPath(null, path);
        if (alpha) |x| g = serve.withAlpha(g, x);
        if (try serve.read(a, &g)) |why| {
            std.debug.print("refused: {s}\n", .{why});
            return error.UnexpectedRefusal;
        }
        return .{ .tmp = tmp, .model = model, .g = g };
    }
};

/// The reason the file was refused, as the operator reads it.
fn refused(a: std.mem.Allocator, g: *?serve.Vec) ![]const u8 {
    return (try serve.read(a, g)) orelse return error.UnexpectedlyAccepted;
}

/// A vector file with one direction per layer, each a unit vector the reader accepts.
fn vectorBytes(a: std.mem.Allocator, w: usize, ids: []const u32, spec: glp.WriteSpec) ![]u8 {
    const layers = try a.alloc(glp.Layer, ids.len);
    for (ids, 0..) |id, i| {
        const dir = try a.alloc(f32, w);
        dir[0] = 1.0;
        layers[i] = .{ .id = id, .dir = dir };
    }
    var s = spec;
    s.layers = layers;
    return glp.write(a, s);
}

test "a missing file is refused with the path and the reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g: ?serve.Vec = serve.withPath(null, ".zig-cache/tmp/no-such-vector.gguf");
    const why = try refused(a, &g);
    try std.testing.expect(std.mem.indexOf(u8, why, "argument --glp:") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "cannot read") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "no such file") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "no-such-vector.gguf") != null);
}

test "a file the reader refuses is refused with the reader's reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = "not a control vector" });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "vector.gguf" });
    var g: ?serve.Vec = serve.withPath(null, path);
    const why = try refused(a, &g);
    try std.testing.expect(std.mem.indexOf(u8, why, "argument --glp:") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "not a GGUF container") != null);
}

test "--glp-alpha without --glp is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g: ?serve.Vec = serve.withAlpha(null, 0.5);
    const why = try refused(a, &g);
    try std.testing.expect(std.mem.indexOf(u8, why, "needs --glp") != null);
}

test "the strength must be finite and above 0" {
    try std.testing.expect(serve.strength("0.5") != null);
    try std.testing.expect(serve.strength("0") == null);
    try std.testing.expect(serve.strength("-1") == null);
    try std.testing.expect(serve.strength("inf") == null);
    try std.testing.expect(serve.strength("nan") == null);
    try std.testing.expect(serve.strength("abc") == null);
}

test "the line says what is in force" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .alpha = 0.5, .base_model = "TensorFold/Qwen3.8-27B-MLX-4bit", .base_revision = "8f2c1d4e", .layers = &.{} }, null);
    defer f.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, f.g.?);
    const line = sink.list.items;
    try std.testing.expect(std.mem.indexOf(u8, line, "glp on:") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "vector.gguf") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "mode: project") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "alpha: 0.50") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "layers: 2 (1-2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "TensorFold/Qwen3.8-27B-MLX-4bit @ 8f2c1d4e") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "verified") != null);
}

test "a good vector is accepted where the engine steers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .alpha = 0.5, .base_model = "TensorFold/Qwen3.8-27B-MLX-4bit", .base_revision = "8f2c1d4e", .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = try serve.refusal(a, f.model, "qwen3_5", "auto", f.g.?);
    if (steers) {
        try std.testing.expect(why == null);
        return;
    }
    // Only the Metal engine steers, so on this platform the same file is refused by name.
    try std.testing.expect(why != null);
    try std.testing.expect(std.mem.indexOf(u8, why.?, "Metal") != null);
}

test "the operator's strength is the one in force" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .alpha = 0.5, .layers = &.{} }, 0.25);
    defer f.tmp.cleanup();
    if (!steers) {
        try std.testing.expectError(error.GlpRefused, serve.engine(a, f.model, "qwen3_5", "auto", f.g));
        return;
    }
    const g = (try serve.engine(a, f.model, "qwen3_5", "auto", f.g)).?;
    try std.testing.expectEqual(@as(f32, 0.25), g.alpha);
    try std.testing.expectEqual(@as(usize, 2), g.vector.layers.len);
}

test "a family that is not steered is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = (try serve.refusal(a, f.model, "glm5_next", "auto", f.g.?)).?;
    try std.testing.expect(std.mem.indexOf(u8, why, "glm5_next") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "Qwen3.8-27B") != null);
}

test "a Gemma family is refused outright" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = (try serve.refusal(a, f.model, "gemma4", "auto", f.g.?)).?;
    try std.testing.expect(std.mem.indexOf(u8, why, "Gemma") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "terms of use") != null);
}

test "a vector the model's width does not match is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, 4096, width, &.{ 1, 2 }, .{ .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = (try serve.refusal(a, f.model, "qwen3_5", "auto", f.g.?)).?;
    try std.testing.expect(std.mem.indexOf(u8, why, "4096") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "5120") != null);
}

test "the CUDA engine does not steer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = (try serve.refusal(a, f.model, "qwen3_5", "cuda", f.g.?)).?;
    try std.testing.expect(std.mem.indexOf(u8, why, "Metal") != null);
}

test "a vector derived at another site is reported" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .derived_at = "ffn_out_pre_residual", .layers = &.{} }, null);
    defer f.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, f.g.?);
    try std.testing.expect(std.mem.indexOf(u8, sink.list.items, "warning:") != null);
    try std.testing.expect(std.mem.indexOf(u8, sink.list.items, "ffn_out_pre_residual") != null);
}

test "a vector derived at another site is not refused where the engine steers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, width, width, &.{ 1, 2 }, .{ .derived_at = "ffn_out_pre_residual", .layers = &.{} }, null);
    defer f.tmp.cleanup();
    const why = try serve.refusal(a, f.model, "qwen3_5", "auto", f.g.?);
    if (steers) {
        try std.testing.expect(why == null);
        return;
    }
    // The site's warning is not the refusal here; the engine is.
    try std.testing.expect(why != null);
    try std.testing.expect(std.mem.indexOf(u8, why.?, "Metal") != null);
}
