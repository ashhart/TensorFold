//! Conversion boundary regressions over actual CLI, encoder and loader validators, without a network or GPU.
const std = @import("std");
const tfq3 = @import("tfq3.zig");
const encoder = @import("tfq3_encode.zig");
const validate = @import("tfq3_validate.zig");
const check = @import("flashnext_memory").tfq3_files.check;
const fixture = @import("tfq3_test.zig");

fn run(a: std.mem.Allocator, root: []const u8, cap: *fixture.Capture, args: []const []const u8) !u8 {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = fixture.makeCtx(a, &env, root, cap, &out.writer, &err.writer);
    return tfq3.runWith(&ctx, args);
}

fn rootPath(a: std.mem.Allocator, tmp: anytype) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

fn absent(io: std.Io, path: []const u8) bool {
    _ = std.Io.Dir.cwd().statFile(io, path, .{}) catch return true;
    return false;
}

test "tfq3 boundary: native inspection accepts only the Flash Next mtp_00 shard identity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    var h = std.mem.zeroes(check.LayerHeader);
    h.magic = check.magic_layer;
    h.version = check.format_version;
    h.layer_idx = 42;
    h.num_experts = check.num_experts;
    h.num_tensors = 3 * check.num_experts;
    h.hidden_size = check.hidden_size;
    h.moe_inter = check.moe_inter;
    h.stream_words = try check.matWords(check.hidden_size, check.moe_inter, 4);
    h.scale_halfs = check.matScales(check.hidden_size, check.moe_inter);
    h.off_meta = 16384;
    h.off_scales = std.mem.alignForward(u64, h.off_meta + check.num_experts * @sizeOf(check.ExpertMeta), 16384);
    h.off_weights = std.mem.alignForward(u64, h.off_scales + h.scale_halfs * 2, 16384);
    const size = std.mem.alignForward(u64, h.off_weights + h.stream_words * 4, 16384);
    const path = try std.fs.path.join(a, &.{ root, "mtp_00.tffpw" });
    {
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.setLength(io, size);
        try f.writePositionalAll(io, std.mem.asBytes(&h), 0);
        var meta: [check.num_experts]check.ExpertMeta = @splat(std.mem.zeroes(check.ExpertMeta));
        for (&meta) |*m| m.b_bits = 4;
        try f.writePositionalAll(io, std.mem.sliceAsBytes(&meta), h.off_meta);
    }
    const got = try validate.shard(a, io, path, size);
    try std.testing.expectEqual(check.num_experts, got.experts);
    try std.testing.expectEqual(@as(u32, 42), got.layer);
    const ordinary = try std.fs.path.join(a, &.{ root, "layer_42.tffpw" });
    try std.Io.Dir.cwd().rename(path, std.Io.Dir.cwd(), ordinary, io);
    _ = try validate.shard(a, io, ordinary, size);
    for ([_][]const u8{ "mtp_01.tffpw", "mtp_0.tffpw", "layer_41.tffpw", "layer_.tffpw" }) |name| {
        const wrong = try std.fs.path.join(a, &.{ root, name });
        try std.Io.Dir.cwd().rename(ordinary, std.Io.Dir.cwd(), wrong, io);
        try std.testing.expectError(error.Tfq3Layer, validate.shard(a, io, wrong, size));
        try std.Io.Dir.cwd().rename(wrong, std.Io.Dir.cwd(), ordinary, io);
    }
    try std.Io.Dir.cwd().rename(ordinary, std.Io.Dir.cwd(), path, io);
    h.hidden_size = 4096;
    h.moe_inter = 2048;
    {
        const f = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
        defer f.close(io);
        try f.writePositionalAll(io, std.mem.asBytes(&h), 0);
    }
    try std.testing.expectError(error.Tfq3Layer, validate.shard(a, io, path, size));
}

test "tfq3 boundary: generic residual choices reach the external converter" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const src = try std.fs.path.join(a, &.{ root, "model" });
    try fixture.writeSyntheticBf16Ckpt(a, std.testing.io, src, 128);
    for ([_][]const u8{ "q8", "bf16", "auto" }) |residual| {
        var cap: fixture.Capture = .{};
        try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ src, "--residual", residual }));
        var got: ?[]const u8 = null;
        for (cap.argv[0 .. cap.argv.len - 1], 0..) |arg, index| {
            if (std.mem.eql(u8, arg, "--residual")) got = cap.argv[index + 1];
        }
        try std.testing.expectEqualStrings(if (std.mem.eql(u8, residual, "auto")) "bf16" else residual, got orelse "missing");
    }
}

test "tfq3 boundary: dot components derive the same sibling default output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const src = try std.fs.path.join(a, &.{ root, "model" });
    try fixture.writeSyntheticBf16Ckpt(a, std.testing.io, src, 128);
    const src_real = try std.Io.Dir.cwd().realPathFileAlloc(std.testing.io, src, a);
    const want = try std.fmt.allocPrint(a, "{s}-TFQ3", .{src_real});
    for ([_][]const u8{ try std.fmt.allocPrint(a, "./{s}/.", .{src}), src, try std.fmt.allocPrint(a, "{s}/../model/.", .{src}) }) |path| {
        var cap: fixture.Capture = .{};
        try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{path}));
        try std.testing.expectEqualStrings(want, cap.argv[4]);
    }
}

test "tfq3 boundary: omitted revision uses the newest valid cached snapshot while explicit main stays strict" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const sha = "abcdef0123456789abcdef0123456789abcdef01";
    const snap = try std.fs.path.join(a, &.{ root, "models--Org--Model", "snapshots", sha });
    try std.Io.Dir.cwd().createDirPath(io, snap);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snap, "config.json" }), .data = "{\"model_type\":\"generic_model\"}" });
    var cap: fixture.Capture = .{};
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ "Org/Model", "--no-pull" }));
    try std.testing.expectEqualStrings(snap, cap.argv[2]);
    try std.testing.expectEqual(@as(usize, 0), cap.pulls);
    cap = .{};
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ "Org/Model@main", "--no-pull" }));
    try std.testing.expectEqual(@as(usize, 0), cap.spawns);
}

fn convert(a: std.mem.Allocator, src: []const u8, out_path: []const u8, head_bits: u32, err: *std.Io.Writer) !u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    return encoder.convertDir(a, std.testing.io, &out.writer, err, .{ .src_dir = src, .out_dir = out_path, .head_bits = head_bits });
}

test "tfq3 boundary: direct encoding refuses an empty metadata stem before publishing output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const header = "{\".weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,32768]}}";
    const bytes = try a.alloc(u8, 8 + header.len + 32768);
    @memset(bytes, 0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    const src = try std.fs.path.join(a, &.{ root, "model" });
    try std.Io.Dir.cwd().createDirPath(io, src);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ src, "model.safetensors" }), .data = bytes });
    const out = try std.fs.path.join(a, &.{ root, "out" });
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try convert(a, src, out, 4, &err.writer));
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "EmptyTensorName") != null);
    try std.testing.expect(absent(io, out));
}

test "tfq3 boundary: direct head bits accept only zero four or five" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const src = try std.fs.path.join(a, &.{ root, "model" });
    try fixture.writeSyntheticBf16Ckpt(a, std.testing.io, src, 128);
    for ([_]u32{ 1, 2, 3, 6, std.math.maxInt(u32) }) |bits| {
        const out = try std.fmt.allocPrint(a, "{s}/bad-{d}", .{ root, bits });
        var err: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try convert(a, src, out, bits, &err.writer));
        try std.testing.expect(std.mem.indexOf(u8, err.written(), "InvalidBits") != null);
        try std.testing.expect(absent(std.testing.io, out));
    }
    for ([_]u32{ 0, 4, 5 }) |bits| {
        const out = try std.fmt.allocPrint(a, "{s}/good-{d}", .{ root, bits });
        var err: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 0), try convert(a, src, out, bits, &err.writer));
    }
}

test "tfq3 boundary: config stat errors propagate while genuine absence remains optional" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try rootPath(a, tmp);
    const src = try std.fs.path.join(a, &.{ root, "model" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 128);
    const loop = try std.fs.path.join(a, &.{ src, "generation_config.json" });
    try std.Io.Dir.cwd().symLink(io, "generation_config.json", loop, .{});
    var err: std.Io.Writer.Allocating = .init(a);
    const out = try std.fs.path.join(a, &.{ root, "out" });
    try std.testing.expectEqual(@as(u8, 1), try convert(a, src, out, 4, &err.writer));
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "SymLinkLoop") != null);
    try std.testing.expect(absent(io, out));
    try std.Io.Dir.cwd().deleteFile(io, loop);
    try std.testing.expectEqual(@as(u8, 0), try convert(a, src, out, 4, &err.writer));
}

test "tfq3 boundary: packed-word allocation failure releases the preceding scales" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var failing = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = 2 });
    const table = encoder.generateTables();
    const weights: [128 * 128]f32 = @splat(1.0);
    try std.testing.expectError(error.OutOfMemory, encoder.encodeMatrix(failing.allocator(), &table, &weights, 128, 128, 4));
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}
