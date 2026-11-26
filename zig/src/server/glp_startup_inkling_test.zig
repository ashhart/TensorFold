//! Inkling 2 Mini's control vector checks before load: Metal only, residual vectors, ids that are decoder visits 1-47.

const std = @import("std");
const builtin = @import("builtin");
const glp = @import("engine_api").glp;
const serve = @import("glp_startup.zig");
const log = @import("log.zig");

const io = std.testing.io;
const hidden = 2304;
const steers = builtin.os.tag == .macos;
const rev = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
const snapshot = "models--thinkingmachines--Inkling-2-Mini/snapshots/" ++ rev;

/// The checkpoint's loop: layers 6-17 run `passes` times; 24 layers and 3 passes make 48 visits.
const Loop = struct { passes: usize = 3, varloop: bool = false };

const Opts = struct {
    width: usize = hidden,
    ids: []const u32 = &.{ 1, 18, 30, 47 },
    version: u32 = 1,
    alpha: f32 = 0.5,
    peak: f32 = 1.0,
    base_model: []const u8 = "",
    base_revision: []const u8 = "",
    loop: ?Loop = .{},
    folder: []const u8 = "",
    alpha_arg: ?f32 = null,
};

const Fixture = struct {
    tmp: std.testing.TmpDir,
    model: []const u8,
    g: ?serve.Vec,

    fn make(a: std.mem.Allocator, o: Opts) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const base = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
        const model = if (o.folder.len > 0) try std.fs.path.join(a, &.{ base, o.folder }) else base;
        if (o.folder.len > 0) try tmp.dir.createDirPath(io, o.folder);
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ o.folder, "config.json" }), .data = try configJson(a, o.loop) });
        try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = try vectorBytes(a, o) });
        var g: ?serve.Vec = serve.withPath(null, try std.fs.path.join(a, &.{ base, "vector.gguf" }));
        if (o.alpha_arg) |x| g = serve.withAlpha(g, x);
        if (try serve.read(a, &g)) |why| {
            std.debug.print("refused: {s}\n", .{why});
            return error.UnexpectedRefusal;
        }
        return .{ .tmp = tmp, .model = model, .g = g };
    }

    fn refusal(f: Fixture, a: std.mem.Allocator, backend: []const u8) !?[]const u8 {
        return serve.refusal(a, f.model, serve.inkling_type, backend, f.g.?);
    }
};

fn configJson(a: std.mem.Allocator, loop: ?Loop) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("{{\"model_type\":\"inkling2_loop\",\"text_config\":{{\"hidden_size\":{d},\"num_hidden_layers\":24", .{hidden});
    if (loop) |l| try w.writer.print(",\"loop_config\":{{\"core_start\":6,\"core_end\":18,\"num_passes\":{d},\"gate_rank\":64,\"is_varloop\":{}}}", .{ l.passes, l.varloop });
    try w.writer.writeAll("}}");
    return w.toOwnedSlice();
}

fn vectorBytes(a: std.mem.Allocator, o: Opts) ![]u8 {
    const layers = try a.alloc(glp.Layer, o.ids.len);
    for (o.ids, layers) |id, *l| {
        const dir = try a.alloc(f32, o.width);
        @memset(dir, 0);
        dir[0] = o.peak;
        l.* = .{ .id = id, .dir = dir };
    }
    return glp.write(a, .{
        .spec_version = o.version,
        .hook_point = if (o.version == 3) glp.paired_site else glp.residual_site,
        .alpha = o.alpha,
        .base_model = o.base_model,
        .base_revision = o.base_revision,
        .layers = layers,
    });
}

fn has(text: ?[]const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text orelse return false, needle) != null;
}

test "an Inkling 2 Mini vector over visits 1-47 is accepted on Metal and refused by name elsewhere" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .ids = &.{ 1, 6, 18, 30, 47 } });
    defer f.tmp.cleanup();
    const why = try f.refusal(a, "auto");
    if (steers) try std.testing.expect(why == null) else try std.testing.expect(has(why, "only the Metal engine"));
    try std.testing.expect(has(try f.refusal(a, "cuda"), "the CUDA engine does not steer Inkling 2 Mini"));
    try std.testing.expect(has(try f.refusal(a, "rocm"), "the ROCm engine does not steer Inkling 2 Mini"));
}

test "a vector that names visit 48 or later is refused before the model loads" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .ids = &.{ 2, 48 } });
    defer f.tmp.cleanup();
    try std.testing.expect(has(try f.refusal(a, "auto"), "decoder visit 48; a token makes 48 visits, so its deepest is 47"));
}

test "a loop other than three fixed passes, or a config without one, is refused" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var two = try Fixture.make(a, .{ .loop = .{ .passes = 2 } });
    defer two.tmp.cleanup();
    try std.testing.expect(has(try two.refusal(a, "auto"), "makes 36 decoder visits"));
    var varies = try Fixture.make(a, .{ .loop = .{ .varloop = true } });
    defer varies.tmp.cleanup();
    try std.testing.expect(has(try varies.refusal(a, "auto"), "is_varloop"));
    var none = try Fixture.make(a, .{ .loop = null });
    defer none.tmp.cleanup();
    try std.testing.expect(has(try none.refusal(a, "auto"), "no text hidden size and loop_config"));
}

test "a width other than 2304 and the DeepSeek paired file are refused" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var wide = try Fixture.make(a, .{ .width = 5120 });
    defer wide.tmp.cleanup();
    try std.testing.expect(has(try wide.refusal(a, "auto"), "5120 wide; the model is 2304"));
    var paired = try Fixture.make(a, .{ .version = 3 });
    defer paired.tmp.cleanup();
    try std.testing.expect(has(try paired.refusal(a, "auto"), "residual vectors (version 1) only"));
}

test "the vector's base model and revision must be the served snapshot's" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var same = try Fixture.make(a, .{ .folder = snapshot, .base_model = "thinkingmachines/Inkling-2-Mini", .base_revision = rev[0..10] });
    defer same.tmp.cleanup();
    try std.testing.expect(try same.refusal(a, "auto") == null);
    var moved = try Fixture.make(a, .{ .folder = snapshot, .base_model = "thinkingmachines/Inkling-2-Mini", .base_revision = "c9d02830aa" });
    defer moved.tmp.cleanup();
    try std.testing.expect(has(try moved.refusal(a, "auto"), "the vector is for revision"));
    var other = try Fixture.make(a, .{ .folder = snapshot, .base_model = "Qwen/Qwen3.8-27B" });
    defer other.tmp.cleanup();
    try std.testing.expect(has(try other.refusal(a, "auto"), "the vector is for Qwen/Qwen3.8-27B"));
}

test "strength 0 is accepted and the line says replies are unsteered" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .alpha_arg = 0 });
    defer f.tmp.cleanup();
    const Sink = struct {
        list: std.ArrayList(u8) = .empty,
        a: std.mem.Allocator,
        fn capture(ctx: *anyopaque, text: []const u8) void {
            const sink: *@This() = @ptrCast(@alignCast(ctx));
            sink.list.appendSlice(sink.a, text) catch {};
        }
    };
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = Sink.capture };
    defer log.Test.sink = null;
    const g = (try serve.engine(a, f.model, serve.inkling_type, "auto", f.g)).?;
    try std.testing.expect(g.off());
    try std.testing.expect(has(sink.list.items, "glp loaded but inactive: alpha 0 bypasses every edit and replies are unsteered"));
}
