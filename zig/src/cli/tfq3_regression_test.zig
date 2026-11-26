//! KR135 regression tests: revision routing, safe native encoding, complete inspection and exact serialization.
const std = @import("std");
const tfq3 = @import("tfq3.zig");
const encode = @import("tfq3_encode.zig");
const fixture = @import("tfq3_test.zig");

fn run(a: std.mem.Allocator, root: []const u8, cap: *fixture.Capture, argv: []const []const u8, out: *std.Io.Writer, err: *std.Io.Writer) !u8 {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    const ctx = fixture.makeCtx(a, &env, root, cap, out, err);
    return tfq3.runWith(&ctx, argv);
}

fn absent(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return true;
    dir.close(io);
    return false;
}

fn writeRaw(a: std.mem.Allocator, io: std.Io, dir: []const u8, header: []const u8, data_len: usize) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = "{\"model_type\":\"llama\"}" });
    const data = try a.alloc(u8, 8 + header.len + data_len);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8 .. 8 + header.len], header);
    const body = data[8 + header.len ..];
    @memset(body, 0);
    if (body.len >= 2) for (0..body.len / 2) |i| std.mem.writeInt(u16, body[i * 2 ..][0..2], 0x3f80, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors" }), .data = data });
}

fn expectNativeRefusal(a: std.mem.Allocator, io: std.Io, root: []const u8, src: []const u8, name: []const u8, extra: []const []const u8) !void {
    const out_path = try std.fmt.allocPrint(a, "{s}/refused-{d}", .{ root, std.Io.Clock.awake.now(io).toNanoseconds() });
    const argv = try std.mem.concat(a, []const u8, &.{ &.{ src, "--out", out_path, "--native" }, extra });
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, argv, &out.writer, &err.writer));
    try std.testing.expect(std.mem.indexOf(u8, err.written(), name) != null);
    try std.testing.expect(absent(io, out_path));
}

test "KR135.2: owner/name@rev selects that cached revision before and after pull" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const main_rev = "0123456789abcdef0123456789abcdef01234567";
    const asked_rev = "abcdef0123456789abcdef0123456789abcdef01";
    const pulled_rev = "fedcba9876543210fedcba9876543210fedcba98";
    const repo = try std.fs.path.join(a, &.{ root, "models--Org--Name" });
    for ([_][]const u8{ main_rev, asked_rev }) |rev| {
        const snap = try std.fs.path.join(a, &.{ repo, "snapshots", rev });
        try std.Io.Dir.cwd().createDirPath(io, snap);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snap, "config.json" }), .data = "{\"model_type\":\"llama\"}" });
    }
    try std.Io.Dir.cwd().createDirPath(io, try std.fs.path.join(a, &.{ repo, "refs" }));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ repo, "refs", "main" }), .data = main_rev });
    var cap: fixture.Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const spec = try std.fmt.allocPrint(a, "Org/Name@{s}", .{asked_rev});
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ spec, "--no-pull" }, &out.writer, &err.writer));
    try std.testing.expect(cap.pulled == null);
    try std.testing.expect(std.mem.endsWith(u8, cap.argv[2], asked_rev));

    var pulled: fixture.Capture = .{ .pull_creates_snapshot = true, .cache_root = root };
    const pull_spec = try std.fmt.allocPrint(a, "Org/Name@{s}", .{pulled_rev});
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &pulled, &.{pull_spec}, &out.writer, &err.writer));
    try std.testing.expectEqualStrings(pull_spec, pulled.pulled.?);
    try std.testing.expect(std.mem.endsWith(u8, pulled.argv[2], pulled_rev));
}

test "KR135.3: native output cannot alias source, reuse stale output, or survive a failed write" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "src" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 128);
    try expectNativeRefusal(a, io, root, src, "source", &.{ "--out", src });

    const occupied = try std.fs.path.join(a, &.{ root, "occupied" });
    try std.Io.Dir.cwd().createDirPath(io, occupied);
    const stale = try std.fs.path.join(a, &.{ occupied, "stale.tffpw" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = stale, .data = "do not mix" });
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ src, "--out", occupied, "--native" }, &out.writer, &err.writer));
    try std.testing.expectEqualStrings("do not mix", std.Io.Dir.cwd().readFileAlloc(io, stale, a, .limited(32)) catch "");

    const malformed = try std.fs.path.join(a, &.{ root, "malformed" });
    const header = "{\"model.norm.weight\":{\"dtype\":3,\"shape\":[128],\"data_offsets\":[0,256]}}";
    try writeRaw(a, io, malformed, header, 256);
    const failed_out = try std.fs.path.join(a, &.{ root, "failed-output" });
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ malformed, "--out", failed_out, "--native" }, &out.writer, &err.writer));
    try std.testing.expect(absent(io, failed_out));
}

test "KR135.4: malformed Safetensors shapes and offsets are rejected before output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const cases = [_][]const u8{
        "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,2]}}",
        "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[256,0]}}",
        "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[-1,256]}}",
        "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128]}}",
    };
    for (cases, 0..) |header, i| {
        const src = try std.fmt.allocPrint(a, "{s}/bad-{d}", .{ root, i });
        const out_path = try std.fmt.allocPrint(a, "{s}/out-{d}", .{ root, i });
        try writeRaw(a, io, src, header, 256);
        var cap: fixture.Capture = .{ .converter = null };
        var out: std.Io.Writer.Allocating = .init(a);
        var err: std.Io.Writer.Allocating = .init(a);
        try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ src, "--out", out_path, "--native" }, &out.writer, &err.writer));
        try std.testing.expect(absent(io, out_path));
    }
}

test "KR135.5: native encoding streams compressed tensors within a bounded live allocation" {
    var fixture_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixture_arena.deinit();
    const fa = fixture_arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(fa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(fa, &.{ root, "many" });
    const out_path = try std.fs.path.join(fa, &.{ root, "out" });
    var header: std.Io.Writer.Allocating = .init(fa);
    try header.writer.writeByte('{');
    const tensor_bytes = 128 * 8192 * 2;
    for (0..8) |i| {
        if (i != 0) try header.writer.writeByte(',');
        try header.writer.print("\"model.layers.{d}.weight\":{{\"dtype\":\"BF16\",\"shape\":[128,8192],\"data_offsets\":[{d},{d}]}}", .{ i, i * tensor_bytes, (i + 1) * tensor_bytes });
    }
    try header.writer.writeByte('}');
    try writeRaw(fa, io, src, header.written(), 8 * tensor_bytes);

    var limited: std.heap.DebugAllocator(.{ .enable_memory_limit = true, .thread_safe = false }) = .init;
    limited.requested_memory_limit = 12 << 20;
    var out: std.Io.Writer.Allocating = .init(limited.allocator());
    var err: std.Io.Writer.Allocating = .init(limited.allocator());
    try std.testing.expectEqual(@as(u8, 0), try encode.convertDir(limited.allocator(), io, &out.writer, &err.writer, .{ .src_dir = src, .out_dir = out_path, .bits = 4 }));
    const shard = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(fa, &.{ out_path, "tfq3-00001-of-00001.tffpw" }), fa, .limited(32 << 20));
    try std.testing.expectEqual(@as(u32, 8), std.mem.readInt(u32, shard[8..12], .little));
}

test "KR135.6: unsupported native options fail by name before writing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "src" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 128);
    try expectNativeRefusal(a, io, root, src, "--profile", &.{ "--profile", "3.5b" });
    try expectNativeRefusal(a, io, root, src, "--residual", &.{ "--residual", "q8" });
    try expectNativeRefusal(a, io, root, src, "--ple-bits", &.{ "--ple-bits", "3" });
    try expectNativeRefusal(a, io, root, src, "--trace", &.{ "--trace", "trace.txt" });
    try expectNativeRefusal(a, io, root, src, "--nonmoe-model", &.{ "--nonmoe-model", "other" });
    try expectNativeRefusal(a, io, root, src, "--skip-existing", &.{"--skip-existing"});
    try expectNativeRefusal(a, io, root, src, "--family", &.{ "--family", "glm" });

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ src, "config.json" }), .data = "{\"model_type\":\"qwen3_5\"}" });
    try expectNativeRefusal(a, io, root, src, "qwen27", &.{});
}

test "KR135.7: inspect rejects corrupt tensor ranges and table values behind valid headers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/inspect", .{tmp.sub_path});
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const table = try a.alloc(u8, 16384);
    @memset(table, 0);
    std.mem.writeInt(u32, table[0..4], 0x33565750, .little);
    std.mem.writeInt(u32, table[4..8], 3, .little);
    const table_path = try std.fs.path.join(a, &.{ dir, "pw_v3_tables.bin" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = table_path, .data = table });
    try fixture.writeValidDenseShard(a, io, dir);
    const dense_path = try std.fs.path.join(a, &.{ dir, "nonmoe.tffpw" });
    const bad = try std.Io.Dir.cwd().readFileAlloc(io, dense_path, a, .limited(64 << 10));
    std.mem.writeInt(u64, bad[128 + 96 ..][0..8], 2048, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dense_path, .data = bad });
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, dir, &cap, &.{ "inspect", dir }, &out.writer, &err.writer));

    try fixture.writeValidDenseShard(a, io, dir);
    std.mem.writeInt(u32, table[8..12], 0x7fc00000, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = table_path, .data = table });
    try std.testing.expectEqual(@as(u8, 1), try run(a, dir, &cap, &.{ "inspect", dir }, &out.writer, &err.writer));
}

test "KR135.8: distinct tensor names sharing their last 79 bytes cannot collide" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "collision" });
    const out_path = try std.fs.path.join(a, &.{ root, "out" });
    const suffix = try a.alloc(u8, 80);
    @memset(suffix, 'z');
    @memcpy(suffix[73..], ".weight");
    const first = try std.mem.concat(a, u8, &.{ "model.layers.a.", suffix });
    const second = try std.mem.concat(a, u8, &.{ "model.layers.b.", suffix });
    const header = try std.fmt.allocPrint(a, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,32768]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[32768,65536]}}}}", .{ first, second });
    try writeRaw(a, io, src, header, 65536);
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ src, "--out", out_path, "--native" }, &out.writer, &err.writer));
    try std.testing.expect(absent(io, out_path));
}

test "KR135.9: residual tensor names and dtypes are valid JSON strings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "json" });
    const out_path = try std.fs.path.join(a, &.{ root, "out" });
    const header = "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,32768]},\"model.norm.\\\"quoted\\\\.weight\":{\"dtype\":\"BF16\",\"shape\":[128],\"data_offsets\":[32768,33024]}}";
    try writeRaw(a, io, src, header, 33024);
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ src, "--out", out_path, "--native", "--bits", "4" }, &out.writer, &err.writer));
    const residual = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ out_path, "residual.safetensors" }), a, .limited(64 << 10));
    const header_len = std.mem.readInt(u64, residual[0..8], .little);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, residual[8 .. 8 + @as(usize, @intCast(header_len))], .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("model.norm.\"quoted\\.weight") != null);
}

test "KR135.10: unaligned native matrices refuse; aligned output passes inspection" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "padded" });
    const out_path = try std.fs.path.join(a, &.{ root, "out" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 192);
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, root, &cap, &.{ src, "--out", out_path, "--native", "--bits", "4" }, &out.writer, &err.writer));
    try std.testing.expect(absent(io, out_path));

    const aligned = try std.fs.path.join(a, &.{ root, "aligned" });
    try fixture.writeSyntheticBf16Ckpt(a, io, aligned, 128);
    const aligned_out = try std.fs.path.join(a, &.{ root, "aligned-out" });
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ aligned, "--out", aligned_out, "--native", "--bits", "4" }, &out.writer, &err.writer));
}

test "KR135.1 and .11: impossible and huge budgets refuse; an ordinary exact budget succeeds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "budget" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 128);
    try expectNativeRefusal(a, io, root, src, "target-gib", &.{ "--target-gib", "0.00000001" });
    const out_path = try std.fs.path.join(a, &.{ root, "fit" });
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ src, "--out", out_path, "--native", "--bits", "5", "--target-gib", "0.001" }, &out.writer, &err.writer));
    const shard = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ out_path, "tfq3-00001-of-00001.tffpw" }), a, .limited(4 << 20));
    try std.testing.expectEqual(@as(u32, 5), std.mem.bytesToValue(encode.TfpnTensorMeta, shard[64..192]).b_bits);
}

test "KR135.6: drafter quantization follows tensor role, not a global metadata flag" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "drafter" });
    const header = "{\"model.layers.0.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[0,32768]},\"mtp.drafter.weight\":{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[32768,65536]},\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[128],\"data_offsets\":[65536,65792]}}";
    try writeRaw(a, io, src, header, 65792);
    var cap: fixture.Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const plain = try std.fs.path.join(a, &.{ root, "plain" });
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ src, "--out", plain, "--native", "--bits", "4" }, &out.writer, &err.writer));
    const plain_shard = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ plain, "tfq3-00001-of-00001.tffpw" }), a, .limited(4 << 20));
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, plain_shard[8..12], .little));
    const residual = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ plain, "residual.safetensors" }), a, .limited(128 << 10));
    try std.testing.expect(std.mem.indexOf(u8, residual, "mtp.drafter.weight") != null);

    const enabled = try std.fs.path.join(a, &.{ root, "enabled" });
    try std.testing.expectEqual(@as(u8, 0), try run(a, root, &cap, &.{ src, "--out", enabled, "--native", "--bits", "4", "--quantize-drafter" }, &out.writer, &err.writer));
    const shard = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ enabled, "tfq3-00001-of-00001.tffpw" }), a, .limited(4 << 20));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, shard[8..12], .little));
    const ordinary = std.mem.bytesToValue(encode.TfpnTensorMeta, shard[64..192]);
    const drafter = std.mem.bytesToValue(encode.TfpnTensorMeta, shard[192..320]);
    try std.testing.expect((ordinary.pad[2] & encode.flag_drafter) == 0);
    try std.testing.expect((drafter.pad[2] & encode.flag_drafter) != 0);
}

test "KR135.12: an explicit bare converter uses PATH and never a CWD decoy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src = try std.fs.path.join(a, &.{ root, "src" });
    try fixture.writeSyntheticBf16Ckpt(a, io, src, 128);
    const converter_name = "tfq3-kr135-cwd-must-not-win";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = converter_name, .data = "cwd decoy" });
    defer std.Io.Dir.cwd().deleteFile(io, converter_name) catch {};
    const bin = try std.fs.path.join(a, &.{ root, "bin" });
    try std.Io.Dir.cwd().createDirPath(io, bin);
    const selected = try std.fs.path.join(a, &.{ bin, converter_name });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = selected, .data = "PATH selection" });

    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    try env.put("PATH", bin);
    var cap: fixture.Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx: tfq3.Context = .{
        .a = a,
        .io = io,
        .out = &out.writer,
        .err = &err.writer,
        .env = &env,
        .cache_override = root,
        .test_ctx = &cap,
        .hooks = .{ .pull_fn = fixture.Capture.pullHook, .spawn_fn = fixture.Capture.spawnHook },
    };
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx, &.{ src, "--converter", converter_name, "--out", try std.fs.path.join(a, &.{ root, "out" }) }));
    try std.testing.expectEqual(@as(usize, 1), cap.spawns);
    try std.testing.expectEqualStrings(selected, cap.argv[0]);
}
