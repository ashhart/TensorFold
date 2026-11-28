//! DeepSeek-V4.1's control vector at startup: the paired file, the CUDA engine alone, and Qwen's behaviour kept.

const std = @import("std");
const builtin = @import("builtin");
const glp = @import("engine_api").glp;
const serve = @import("glp_startup.zig");
const log = @import("log.zig");

const io = std.testing.io;

/// The text model's hidden size and backbone depth DeepSeek-V4.1 has; its last block is layer 39.
const hidden = 5120;
const depth = 40;

/// The base repo and commit a synthetic vector names, and the hub cache folders that hold such a checkpoint.
const repo = "example-org/Synthetic-HC-Model";
const commit = "0123456789abcdef0123456789abcdef01234567";
const snapshot = "models--example-org--Synthetic-HC-Model/snapshots/" ++ commit;

/// The lines the server printed, for the report the operator reads.
const Sink = struct {
    list: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,

    fn has(self: Sink, needle: []const u8) bool {
        return std.mem.indexOf(u8, self.list.items, needle) != null;
    }

    fn lines(self: Sink) usize {
        return std.mem.count(u8, self.list.items, "\n");
    }
};

fn capture(ctx: *anyopaque, text: []const u8) void {
    const sink: *Sink = @ptrCast(@alignCast(ctx));
    sink.list.appendSlice(sink.a, text) catch {};
}

/// What the checkpoint's config.json holds, so a test can bend one number at a time.
const Config = struct {
    model_type: []const u8 = "deepseek_v41",
    hidden: ?usize = hidden,
    layers: ?usize = depth,
    hc: ?usize = 4,
    /// A vision tower's numbers beside the text model's, which a reader must never take for the model's.
    decoy: bool = true,
    /// Whether the folder holds a config.json at all.
    present: bool = true,
};

/// What a vector file and its checkpoint hold.
const Opts = struct {
    width: usize = hidden,
    ids: []const u32 = &.{ 1, 2 },
    /// The version the file is written as: 3 the paired DeepSeek vector, 1 the residual one.
    version: u32 = 3,
    /// The strength the file carries as its own.
    alpha: f32 = 0.5,
    /// The first component of every direction.
    peak: f32 = 1.0,
    base_model: []const u8 = "",
    base_revision: []const u8 = "",
    derived_at: ?[]const u8 = null,
    config: Config = .{},
    /// The checkpoint folder under the temporary one, as a hub cache names it; empty: the temporary folder itself.
    folder: []const u8 = "",
    /// The strength the operator gave.
    alpha_arg: ?f32 = null,
};

/// A checkpoint folder with a config.json, a vector file beside the temporary folder's root, and the vector read.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    /// The model's folder, as the serve path names it.
    model: []const u8,
    /// The vector file's path.
    path: []const u8,
    g: ?serve.Vec,

    fn make(a: std.mem.Allocator, o: Opts) !Fixture {
        var f = try place(a, o, try vectorBytes(a, o));
        if (try serve.read(a, &f.g)) |why| {
            std.debug.print("refused: {s}\n", .{why});
            return error.UnexpectedRefusal;
        }
        return f;
    }

    /// The folder and the file, the vector not yet read.
    fn place(a: std.mem.Allocator, o: Opts, bytes: []const u8) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const base = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
        const model = if (o.folder.len > 0) try std.fs.path.join(a, &.{ base, o.folder }) else base;
        if (o.folder.len > 0) try tmp.dir.createDirPath(io, o.folder);
        if (o.config.present) try tmp.dir.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ o.folder, "config.json" }), .data = try configJson(a, o.config) });
        try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = bytes });
        const path = try std.fs.path.join(a, &.{ base, "vector.gguf" });
        var g: ?serve.Vec = serve.withPath(null, path);
        if (o.alpha_arg) |x| g = serve.withAlpha(g, x);
        return .{ .tmp = tmp, .model = model, .path = path, .g = g };
    }

    fn refusal(self: Fixture, a: std.mem.Allocator, model_type: []const u8, backend: []const u8) !?[]const u8 {
        return serve.refusal(a, self.model, model_type, backend, self.g.?);
    }
};

/// The config shape of a text model with a vision tower beside it: the text model's numbers live in text_config.
fn configJson(a: std.mem.Allocator, c: Config) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("{{\"model_type\":\"{s}\"", .{c.model_type});
    if (c.decoy) try w.writer.writeAll(",\"vision_config\":{\"hidden_size\":1024,\"num_hidden_layers\":32}");
    try w.writer.writeAll(",\"text_config\":{\"model_type\":\"text\"");
    if (c.hidden) |n| try w.writer.print(",\"hidden_size\":{d}", .{n});
    if (c.layers) |n| try w.writer.print(",\"num_hidden_layers\":{d}", .{n});
    if (c.hc) |n| try w.writer.print(",\"hc_mult\":{d}", .{n});
    try w.writer.writeAll("}}");
    return w.toOwnedSlice();
}

/// A vector file with one direction per layer, each along the first axis.
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
        .derived_at = o.derived_at,
        .layers = layers,
    });
}

fn contains(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

/// Why `f`'s vector is refused for this checkpoint type and backend, or a failure when it is taken.
fn refusedFor(a: std.mem.Allocator, f: Fixture, model_type: []const u8, backend: []const u8) ![]const u8 {
    return (try f.refusal(a, model_type, backend)) orelse return error.UnexpectedlyTaken;
}

test "a paired vector is taken by the CUDA engine, which is handed it whole and reports it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .alpha = 0.5 });
    defer f.tmp.cleanup();
    try std.testing.expect(try f.refusal(a, "deepseek_v41", "cuda") == null);
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    const g = (try serve.engine(a, f.model, "deepseek_v41", "cuda", f.g)).?;
    try std.testing.expect(g.vector.isPaired());
    try std.testing.expectEqual(@as(f32, 0.5), g.alpha);
    try std.testing.expect(!g.off());
    try std.testing.expectEqual(@as(usize, 2), g.vector.layers.len);
    try std.testing.expectEqual(@as(f32, 1), g.vector.layer(2).?[0]);
    try std.testing.expectEqualSlices(u8, &f.g.?.vector.?.file_sha256, &g.vector.file_sha256);
    try std.testing.expect(sink.has("glp on: replies are steered (add vector in"));
    try std.testing.expect(sink.has("hook: hc_fold_pre_engram; spec v3;"));
    try std.testing.expect(sink.has("alpha: 0.5;"));
    try std.testing.expect(sink.has("layers: 2 (1-2)"));
    try std.testing.expect(!sink.has("inactive"));
    try std.testing.expectEqual(@as(usize, 1), sink.lines());
    // the operator's strength, given with the file, replaces the file's own
    var given = try Fixture.make(a, .{ .alpha = 0.5, .alpha_arg = 0.25 });
    defer given.tmp.cleanup();
    try std.testing.expectEqual(@as(f32, 0.25), (try serve.engine(a, given.model, "deepseek_v41", "cuda", given.g)).?.alpha);
}

test "no vector named leaves DeepSeek's startup as it was" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(try serve.engine(a, "no-such-folder", "deepseek_v41", "cuda", null) == null);
    try std.testing.expect(try serve.engine(a, "no-such-folder", "deepseek_v41", "mlx", .{}) == null);
    try std.testing.expect(try serve.refusal(a, "no-such-folder", "deepseek_v41", "mlx", .{}) == null);
}

test "auto is the CUDA engine on Linux, and a Mac's auto is Metal, which never steers DeepSeek" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    if (builtin.os.tag == .linux) {
        try std.testing.expect(try f.refusal(a, "deepseek_v41", "auto") == null);
    } else {
        try std.testing.expect(try f.refusal(a, "deepseek_v41", "auto") != null);
    }
    if (builtin.os.tag == .macos) {
        const why = try refusedFor(a, f, "deepseek_v41", "auto");
        try std.testing.expect(contains(why, "the Metal engine does not steer DeepSeek-V4.1: only the CUDA engine does"));
    }
}

test "Metal and ROCm never steer DeepSeek-V4.1, and the engine refuses to be handed the vector" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    const metal = try refusedFor(a, f, "deepseek_v41", "mlx");
    try std.testing.expect(contains(metal, "--glp: the Metal engine does not steer DeepSeek-V4.1: only the CUDA engine does"));
    const rocm = try refusedFor(a, f, "deepseek_v41", "rocm");
    try std.testing.expect(contains(rocm, "the ROCm engine does not steer DeepSeek-V4.1"));
    try std.testing.expectError(error.GlpRefused, serve.engine(a, f.model, "deepseek_v41", "mlx", f.g));
    try std.testing.expectError(error.GlpRefused, serve.engine(a, f.model, "deepseek_v41", "rocm", f.g));
}

test "no other family is steered, each refused by name, and Gemma outright" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    for ([_][]const u8{ "glm5_next", "deepseek_v4", "qwen3_5_moe" }) |model_type| {
        const why = try refusedFor(a, f, model_type, "cuda");
        try std.testing.expect(contains(why, model_type));
        try std.testing.expect(contains(why, "only Qwen3.8-27B, Inkling 2 Mini and DeepSeek-V4.1 take a control vector"));
    }
    try std.testing.expect(contains(try refusedFor(a, f, "gemma4", "cuda"), "Gemma"));
}

test "a residual vector is not a DeepSeek vector, and a paired vector is not a Qwen one, on any engine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var residual = try Fixture.make(a, .{ .version = 1 });
    defer residual.tmp.cleanup();
    const not_paired = try refusedFor(a, residual, "deepseek_v41", "cuda");
    try std.testing.expect(contains(not_paired, "takes version 3 at hc_fold_pre_engram"));
    const qwen: Config = .{ .model_type = "qwen3_5", .layers = 64, .hc = null, .decoy = false };
    var paired = try Fixture.make(a, .{ .config = qwen });
    defer paired.tmp.cleanup();
    for ([_][]const u8{ "auto", "mlx", "cuda", "rocm" }) |backend| {
        const why = try refusedFor(a, paired, "qwen3_5", backend);
        try std.testing.expect(contains(why, "residual vectors (version 1) only"));
    }
}

test "Qwen keeps its Metal permission, and its other backends their refusals" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qwen: Config = .{ .model_type = "qwen3_5", .layers = 64, .hc = null, .decoy = false };
    var f = try Fixture.make(a, .{ .version = 1, .config = qwen, .alpha = 0.5 });
    defer f.tmp.cleanup();
    if (builtin.os.tag == .macos) {
        try std.testing.expect(try f.refusal(a, "qwen3_5", "mlx") == null);
        try std.testing.expect(try f.refusal(a, "qwen3_5", "auto") == null);
        const g = (try serve.engine(a, f.model, "qwen3_5", "mlx", f.g)).?;
        try std.testing.expect(!g.vector.isPaired());
        try std.testing.expectEqual(@as(u32, 1), g.vector.spec_version);
    } else {
        try std.testing.expect(contains(try refusedFor(a, f, "qwen3_5", "auto"), "only the Metal engine steers"));
    }
    for ([_][]const u8{ "cuda", "rocm" }) |backend| {
        try std.testing.expect(contains(try refusedFor(a, f, "qwen3_5", backend), "only the Metal engine steers"));
    }
}

test "the checkpoint's own numbers decide: the width, the depth, the copies and what the config leaves out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var narrow = try Fixture.make(a, .{ .width = 4096 });
    defer narrow.tmp.cleanup();
    const wide = try refusedFor(a, narrow, "deepseek_v41", "cuda");
    try std.testing.expect(contains(wide, "4096 wide; the model is 5120"));

    var past = try Fixture.make(a, .{ .ids = &.{ 1, depth } });
    defer past.tmp.cleanup();
    const deep = try refusedFor(a, past, "deepseek_v41", "cuda");
    try std.testing.expect(contains(deep, "layer 40 of a model with 40 backbone layers: its deepest steerable layer is 39"));

    var last = try Fixture.make(a, .{ .ids = &.{ 1, depth - 1 } });
    defer last.tmp.cleanup();
    try std.testing.expect(try last.refusal(a, "deepseek_v41", "cuda") == null);

    var copies = try Fixture.make(a, .{ .config = .{ .hc = 2 } });
    defer copies.tmp.cleanup();
    const two = try refusedFor(a, copies, "deepseek_v41", "cuda");
    try std.testing.expect(contains(two, "applied to 4 stream copies; this checkpoint carries 2"));

    var no_copies = try Fixture.make(a, .{ .config = .{ .hc = null } });
    defer no_copies.tmp.cleanup();
    try std.testing.expect(try no_copies.refusal(a, "deepseek_v41", "cuda") == null);

    for ([_]Config{ .{ .hidden = null }, .{ .layers = null }, .{ .present = false } }) |config| {
        var f = try Fixture.make(a, .{ .config = config });
        defer f.tmp.cleanup();
        const why = try refusedFor(a, f, "deepseek_v41", "cuda");
        try std.testing.expect(contains(why, "gives no text hidden size and depth"));
    }
}

test "a hub snapshot's repo is the vector's very repo, and its commit the vector's pin" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const named: Opts = .{ .base_model = repo, .base_revision = commit, .folder = snapshot };
    var same = try Fixture.make(a, named);
    defer same.tmp.cleanup();
    try std.testing.expect(try same.refusal(a, "deepseek_v41", "cuda") == null);

    var short = try Fixture.make(a, .{ .base_model = repo, .base_revision = commit[0..7], .folder = snapshot });
    defer short.tmp.cleanup();
    try std.testing.expect(try short.refusal(a, "deepseek_v41", "cuda") == null);

    var unpinned = try Fixture.make(a, .{ .base_model = repo, .folder = snapshot });
    defer unpinned.tmp.cleanup();
    try std.testing.expect(try unpinned.refusal(a, "deepseek_v41", "cuda") == null);

    // a repo that merely resembles it, or a fork of it, is another checkpoint
    for ([_][]const u8{ "models--example-org--Synthetic-HC-Model-EXL3", "models--other-org--Synthetic-HC-Model" }) |cache| {
        const folder = try std.fmt.allocPrint(a, "{s}/snapshots/{s}", .{ cache, commit });
        var other = try Fixture.make(a, .{ .base_model = repo, .base_revision = commit, .folder = folder });
        defer other.tmp.cleanup();
        const why = try refusedFor(a, other, "deepseek_v41", "cuda");
        try std.testing.expect(contains(why, "the vector is for example-org/Synthetic-HC-Model; the model is "));
    }

    // the same repo at another commit
    const moved = "models--example-org--Synthetic-HC-Model/snapshots/fedcba9876543210fedcba9876543210fedcba98";
    var elsewhere = try Fixture.make(a, .{ .base_model = repo, .base_revision = commit, .folder = moved });
    defer elsewhere.tmp.cleanup();
    const why = try refusedFor(a, elsewhere, "deepseek_v41", "cuda");
    try std.testing.expect(contains(why, "the vector is for revision"));

    // a local conversion names no repo: nothing to compare, so only the numbers were checked
    var local = try Fixture.make(a, .{ .base_model = repo, .base_revision = commit });
    defer local.tmp.cleanup();
    try std.testing.expect(try local.refusal(a, "deepseek_v41", "cuda") == null);
}

test "a zero strength loads, is inactive, and is a policy of its own, never the absence of one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .alpha = 0 });
    defer f.tmp.cleanup();
    try std.testing.expect(try f.refusal(a, "deepseek_v41", "cuda") == null);
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    const g = (try serve.engine(a, f.model, "deepseek_v41", "cuda", f.g)).?;
    try std.testing.expect(g.off());
    try std.testing.expectEqual(@as(f32, 0), g.alpha);
    try std.testing.expect(!std.mem.allEqual(u8, &g.vector.file_sha256, 0));
    try std.testing.expect(sink.has("glp loaded but inactive: alpha 0 bypasses every edit and replies are unsteered"));
    try std.testing.expect(sink.has("alpha: 0;"));
    try std.testing.expect(!sink.has("glp on"));
    // the operator's zero over a live file is the same, and a negative zero is that zero
    var live = try Fixture.make(a, .{ .alpha = 0.5, .alpha_arg = -0.0 });
    defer live.tmp.cleanup();
    const off = (try serve.engine(a, live.model, "deepseek_v41", "cuda", live.g)).?;
    try std.testing.expect(off.off() and !std.math.signbit(off.alpha));
    // and the operator's strength over a zero file is a live one
    var woken = try Fixture.make(a, .{ .alpha = 0, .alpha_arg = 0.25 });
    defer woken.tmp.cleanup();
    try std.testing.expect(!(try serve.engine(a, woken.model, "deepseek_v41", "cuda", woken.g)).?.off());
}

test "Qwen takes no zero strength: it was refused before and still is" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qwen: Config = .{ .model_type = "qwen3_5", .layers = 64, .hc = null, .decoy = false };
    for ([_]Opts{ .{ .version = 1, .config = qwen, .alpha = 0 }, .{ .version = 1, .config = qwen, .alpha = 0.5, .alpha_arg = 0 } }) |o| {
        var f = try Fixture.make(a, o);
        defer f.tmp.cleanup();
        const why = try refusedFor(a, f, "qwen3_5", "mlx");
        if (builtin.os.tag == .macos) {
            try std.testing.expect(contains(why, "Qwen3.8-27B takes a strength above 0"));
        } else {
            try std.testing.expect(contains(why, "only the Metal engine steers"));
        }
    }
}

test "an edit past what BF16 holds is refused when the file is read, by the strength that causes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fits = try Fixture.make(a, .{ .peak = 1e38, .alpha = 0.5, .alpha_arg = 3 });
    defer fits.tmp.cleanup();
    var f = try Fixture.place(a, .{ .peak = 1e38, .alpha_arg = 10 }, try vectorBytes(a, .{ .peak = 1e38 }));
    defer f.tmp.cleanup();
    const why = (try serve.read(a, &f.g)).?;
    try std.testing.expect(contains(why, "argument --glp: layer 1's direction times the strength is past what BF16 holds"));
}

/// Overwrite the bytes a key's value starts with: `skip` bytes past the key's own, which holds the value's type.
fn patch(bytes: []u8, key: []const u8, skip: usize, with: []const u8) void {
    const at = std.mem.indexOf(u8, bytes, key).?;
    @memcpy(bytes[at + key.len + skip ..][0..with.len], with);
}

/// The reason startup refuses a vector file that has been bent, as the operator reads it.
fn malformed(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var f = try Fixture.place(a, .{}, bytes);
    defer f.tmp.cleanup();
    return (try serve.read(a, &f.g)) orelse return error.UnexpectedlyAccepted;
}

test "a bent paired file is refused at startup with the reader's reason, whatever the engine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good = try vectorBytes(a, .{});
    {
        const bytes = try a.dupe(u8, good);
        patch(bytes, "glp.mode", 12, "mul");
        try std.testing.expect(contains(try malformed(a, bytes), "argument --glp: the file's mode is not one this reader applies; it applies add vectors only"));
    }
    {
        const bytes = try a.dupe(u8, good);
        patch(bytes, "glp.hook_point", 12, "hc_fold_pre_engrax");
        try std.testing.expect(contains(try malformed(a, bytes), "is not applied; version 3 applies at hc_fold_pre_engram only"));
    }
    {
        const bytes = try a.dupe(u8, good);
        patch(bytes, "glp.spec_version", 4, &std.mem.toBytes(@as(u32, 2)));
        try std.testing.expect(contains(try malformed(a, bytes), "glp.spec_version 2"));
    }
    {
        const bytes = try a.dupe(u8, good);
        patch(bytes, "glp.spec_version", 4, &std.mem.toBytes(@as(u32, 1)));
        try std.testing.expect(contains(try malformed(a, bytes), "needs glp.spec_version 3"));
    }
    {
        const bytes = try a.dupe(u8, good);
        bytes[bytes.len - 1] ^= 0x01;
        try std.testing.expect(contains(try malformed(a, bytes), "content digest does not match"));
    }
    {
        const bytes = try a.dupe(u8, good);
        @memcpy(bytes[bytes.len - 4 ..], &std.mem.toBytes(@as(u32, 0x7fc00000)));
        try std.testing.expect(contains(try malformed(a, bytes), "not finite"));
    }
    try std.testing.expect(contains(try malformed(a, good[0 .. good.len - 100]), "points past the file"));
    try std.testing.expect(contains(try malformed(a, "GGUF"), "too short"));
}

test "the line shows a paired file's name escaped, so a name cannot add a line of its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .derived_at = "ffn_out_pre_residual" });
    defer f.tmp.cleanup();
    var named = f.g.?;
    named.path = "dir/a\nglp off: \"x\".gguf";
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, named);
    try std.testing.expect(sink.has("dir/a\\nglp off: \\\"x\\\".gguf"));
    // one line for the vector and one for the warning that it was derived elsewhere
    try std.testing.expectEqual(@as(usize, 2), sink.lines());
    try std.testing.expect(sink.has("derived at ffn_out_pre_residual, not at hc_fold_pre_engram: it is applied at hc_fold_pre_engram"));
}
