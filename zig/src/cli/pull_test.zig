//! `pull` against a fake hub on 127.0.0.1: downloads, retries, resume, shared pieces and the whole-file fallback.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");
const ranged = @import("pull_parts.zig");
const single = @import("pull_file.zig");
const exlr8_pull = @import("pull_exlr8.zig");
const TestClient = @import("pull_test_client.zig").Client;
pub const FakeHub = @import("pull_fake_hub.zig").FakeHub;

fn testEnv(a: Allocator, endpoint_url: []const u8) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", endpoint_url);
    return env;
}

test "pull downloads, verifies, retries a cut body, resumes and refuses a family Zig cannot serve" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    const snapshot = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1" });
    const config_link = try std.fs.path.join(a, &.{ snapshot, "config.json" });
    const weights_link = try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" });
    const config_read = try std.Io.Dir.cwd().readFileAlloc(io, config_link, a, .limited(1 << 20));
    try std.testing.expectEqualStrings(FakeHub.config_body, config_read);
    const weights_read = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, weights_read);
    // A shard in a folder inside the model folder lands as a real subfolder in the snapshot, still a blob link.
    const nested_link = try std.fs.path.join(a, &.{ snapshot, "tensors/model-00001-of-00001.safetensors" });
    const nested_read = try std.Io.Dir.cwd().readFileAlloc(io, nested_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.nested_body, nested_read);
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.nested_sha_hex }), .{});
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), a, .limited(64));
    try std.testing.expectEqualStrings("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1", std.mem.trim(u8, ref, " \r\n"));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors") != null);
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    // The blob is named by the hub's bare-hex oid, and the body cut short once came down on the retry.
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex }), .{});
    try std.testing.expectEqual(@as(u32, 2), fake_hub.flaky_requests.load(.monotonic));
    // Small files are blobs named by their git blob sha1, as huggingface_hub names them.
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", &fake_hub.config_git_hex }), .{});
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", &fake_hub.flaky_git_hex }), .{});
    try std.testing.expectEqualStrings(FakeHub.flaky_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "flaky.json" }), a, .limited(1 << 20)));

    // A second pull skips the verified blob and touches the weights endpoint no more.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "weights.safetensors: cached") != null);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "tensors/model-00001-of-00001.safetensors: cached") != null);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "flaky.json: cached") != null);
    try std.testing.expectEqual(@as(u32, 2), fake_hub.flaky_requests.load(.monotonic));

    // A partial blob resumes from its own length.
    const blob_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex });
    std.Io.Dir.cwd().deleteFile(io, blob_path) catch {};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{blob_path}), .data = FakeHub.weights_body[0..2000] });
    var out3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out3.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u64, 2000), fake_hub.range_starts.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), fake_hub.weight_requests.load(.monotonic));
    const resumed = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, resumed);

    // A family Zig cannot serve is refused with the 0.6 line, after the config check only.
    var refused_out: std.Io.Writer.Allocating = .init(a);
    var refused_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &refused_out.writer, &refused_err.writer, &env, root, "Org/Draft"));
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "model_type gemma3n") != null);
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "tensorfold@0.6") != null);

    // A DFlash2 drafter repo comes down for serve --drafter although no family serves its model_type.
    var drafter_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &drafter_out.writer, &err_out.writer, &env, root, "Org/DFlash"));
    try std.testing.expect(std.mem.indexOf(u8, drafter_out.written(), "DFlash2 drafter (qwen3)") != null);
    try std.testing.expectEqualStrings(FakeHub.drafter_config, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--DFlash/snapshots/ddddddddddddddddddddddddddddddddddddddd4/config.json" }), a, .limited(1 << 20)));

    // CLI classification accepts a Transcribe-shaped DFlash2 config; it is not a runtime-support receipt.
    var transcribe_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &transcribe_out.writer, &err_out.writer, &env, root, "Org/TranscribeDraft"));
    try std.testing.expect(std.mem.indexOf(u8, transcribe_out.written(), "DFlash2 drafter (nemotron_parse)") != null);
    try std.testing.expectEqualStrings(FakeHub.transcribe_drafter_config, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--TranscribeDraft/snapshots", FakeHub.transcribe_drafter_sha, "config.json" }), a, .limited(1 << 20)));

    const before_lilicorr = fake_hub.weight_requests.load(.monotonic);
    var lilicorr_out: std.Io.Writer.Allocating = .init(a);
    var lilicorr_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &lilicorr_out.writer, &lilicorr_err.writer, &env, root, "Org/LiLiCorr"));
    try std.testing.expect(std.mem.indexOf(u8, lilicorr_err.written(), "no LiLiCorr drafter backend") != null);
    try std.testing.expect(std.mem.indexOf(u8, lilicorr_err.written(), "refusing before download") != null);
    try std.testing.expect(std.mem.indexOf(u8, lilicorr_out.written(), "DFlash2 drafter") == null);
    try std.testing.expectEqual(before_lilicorr, fake_hub.weight_requests.load(.monotonic));
    const lilicorr_config_path = try std.fs.path.join(a, &.{ root, "models--Org--LiLiCorr/snapshots", FakeHub.lilicorr_sha, "config.json" });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().readFileAlloc(io, lilicorr_config_path, a, .limited(1 << 20)));

    // A bad repo id is usage.
    var bad_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 2), try pull.run(a, io, &out.writer, &bad_err.writer, &env, root, "no-slash"));
}

test "large files come down as shared pieces, and whole files when the hub ignores ranges" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-pieces-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));
    const small: pull.Sizes = .{ .threshold = 4096, .piece = 1250 };

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash", small));
    try std.testing.expectEqual(@as(u32, 4), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors: 0.00 MiB in 4 pieces") != null);
    const flash = try std.fs.path.join(a, &.{ root, "models--Org--Flash" });
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ flash, "snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/weights.safetensors" }), a, .limited(1 << 20)));
    const blob = try std.fs.path.join(a, &.{ flash, "blobs", fake_hub.weights_sha_hex });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.ranges.part0", .{blob}), .{}));

    // A repo whose weights ignore ranges: the pieces give way to one stream, and no piece file is left behind.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Whole", small));
    const whole = try std.fs.path.join(a, &.{ root, "models--Org--Whole" });
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ whole, "snapshots/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2/weights.safetensors" }), a, .limited(1 << 20)));
    const whole_blob = try std.fs.path.join(a, &.{ whole, "blobs", fake_hub.weights_sha_hex });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.ranges", .{whole_blob}), .{}));
}

test "shared pieces fill every byte across files, restart only unmarked pieces and refuse a server without ranges" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-ranges-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    try std.Io.Dir.cwd().createDirPath(io, root);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake_hub.endpoint(a), FakeHub.weights_path });
    const size = FakeHub.weights_body.len;
    const files = [_]ranged.File{
        .{ .url = url, .path = try std.fs.path.join(a, &.{ root, "first.ranges" }), .size = size },
        .{ .url = url, .path = try std.fs.path.join(a, &.{ root, "second.ranges" }), .size = size },
    };

    try ranged.fetch(a, io, &files, 1250, .{}, null);
    try std.testing.expectEqual(@as(u32, 8), fake_hub.weight_requests.load(.monotonic));
    for (files) |f| try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, f.path, a, .limited(1 << 20)));

    // Pieces 0 and 2 already marked done: only 1 and 3 are fetched again.
    const first = files[0];
    try std.Io.Dir.cwd().deleteFile(io, first.path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = first.path, .data = FakeHub.weights_body });
    for ([_]u32{ 0, 2 }) |i| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.part{d}", .{ first.path, i }), .data = "" });
    try ranged.fetch(a, io, &.{first}, 1250, .{}, null);
    try std.testing.expectEqual(@as(u32, 10), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expectEqualSlices(u8, &(try ranged.digest(a, io, first.path, size)), &(try hexBytes(fake_hub.weights_sha_hex)));

    // A server that answers a range with the whole file (the config endpoint here) is refused for pieces.
    const config_url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake_hub.endpoint(a), FakeHub.config_path });
    try std.testing.expectError(error.RangeUnsupported, ranged.fetch(a, io, &.{.{ .url = config_url, .path = try std.fs.path.join(a, &.{ root, "config.ranges" }), .size = FakeHub.config_body.len }}, 50, .{}, null));
}

fn hexBytes(text: []const u8) ![32]u8 {
    var out: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, text);
    return out;
}

test "an EXLR8 node's whole files land in a relative directory, small files and LFS pieces alike" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const dir = try std.fmt.allocPrint(a, ".tf-pull-exlr8-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    var sha256: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&sha256, fake_hub.weights_sha_hex);
    var git_sha1: [20]u8 = undefined;
    _ = try std.fmt.hexToBytes(&git_sha1, &fake_hub.config_git_hex);
    const whole = [_]single.Entry{
        .{ .path = "config.json", .size = FakeHub.config_body.len, .git_sha1 = git_sha1 },
        .{ .path = "weights.safetensors", .size = FakeHub.weights_body.len, .sha256 = sha256 },
    };
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u32, 2), try exlr8_pull.fetchWhole(a, io, &client, &out.writer, &err_out.writer, .{ .endpoint = try fake_hub.endpoint(a) }, "Org/Flash", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1", dir, &whole));
    const cwd = std.Io.Dir.cwd();
    try std.testing.expectEqualStrings(FakeHub.config_body, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20)));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.safetensors" }), a, .limited(1 << 20)));
}

test "security: a hub response cannot write beyond the manifest file size" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client_io.io();
    var fake = try FakeHub.open();
    fake.start();
    defer fake.stopServer();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake.endpoint(a), FakeHub.weights_path });
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectError(error.SizeMismatch, single.downloadRetrying(a, io, &client, .{}, &out.writer, url, dir, .{
        .path = "overflow.bin",
        .size = 8,
    }));
    const partial = try std.fs.path.join(a, &.{ dir, "overflow.bin.incomplete" });
    const written = single.fileStat(io, partial) catch |err| switch (err) {
        error.FileNotFound => 0,
        else => return err,
    };
    try std.testing.expect(written <= 8);
}

test "security: a shard in a folder inside the model folder lands inside the snapshot" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-nested-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    const snapshot = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1" });
    // NVIDIA's own export puts every shard in `tensors/`; the snapshot keeps that folder and still links the blob.
    const link = try std.fs.path.join(a, &.{ snapshot, "tensors/model-00001-of-00001.safetensors" });
    const body = try std.Io.Dir.cwd().readFileAlloc(io, link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.nested_body, body);
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.nested_sha_hex }), .{});
}

test "security: a hub tree path TensorFold reserves is refused, naming the path and why" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena_state = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = client_io.io();
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-reserved-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));
    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);

    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Reserved"));
    const text = err_out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, ".tensorfold-save-pending") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "refusing") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "reserves") != null);
    // the planted name is not linked into the snapshot
    const planted = try std.fs.path.join(a, &.{ root, "models--Org--Reserved/snapshots/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee5/.tensorfold-save-pending" });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, planted, .{}));
}

test "pull refuses cache mutation while rm holds the cache directory" {
    var client_io: TestClient = undefined;
    client_io.init();
    defer client_io.deinit();
    var arena = std.heap.ArenaAllocator.init(client_io.allocator());
    defer arena.deinit();
    const a = arena.allocator();
    const io = client_io.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    const env = try testEnv(a, "http://127.0.0.1:1");
    const lease = try @import("sys").folder_lock.hold(a, root, true);
    defer lease.release();
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &out.writer, &err.writer, &env, root, "Org/Flash"));
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "cache") != null);
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "Busy") != null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash" }), .{}));
}
