//! Host tests for `tensorfold tfq3`: argument parsing, pull flow, family detection, and inspect.
const std = @import("std");
const tfq3 = @import("tfq3.zig");
const Environ = std.process.Environ.Map;

const Capture = struct {
    pulled: ?[]const u8 = null,
    pull_code: u8 = 0,
    pull_creates_snapshot: bool = false,
    cache_root: []const u8 = "",
    converter: ?[]const u8 = "/fake/bin/tfq3",
    argv: []const []const u8 = &.{},
    spawns: usize = 0,

    fn pullHook(ctx: *const tfq3.Context, spec: []const u8) anyerror!u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        c.pulled = try ctx.a.dupe(u8, spec);
        if (c.pull_creates_snapshot and c.pull_code == 0) {
            const at = std.mem.indexOfScalar(u8, spec, '@');
            const repo = if (at) |idx| spec[0..idx] else spec;
            const slash = std.mem.indexOfScalar(u8, repo, '/').?;
            const sha = "0123456789abcdef0123456789abcdef01234567";
            const dir_name = try std.fmt.allocPrint(ctx.a, "models--{s}--{s}", .{ repo[0..slash], repo[slash + 1 ..] });
            const snap = try std.fs.path.join(ctx.a, &.{ c.cache_root, dir_name, "snapshots", sha });
            const refs = try std.fs.path.join(ctx.a, &.{ c.cache_root, dir_name, "refs" });
            try std.Io.Dir.cwd().createDirPath(ctx.io, snap);
            try std.Io.Dir.cwd().createDirPath(ctx.io, refs);
            try std.Io.Dir.cwd().writeFile(ctx.io, .{ .sub_path = try std.fs.path.join(ctx.a, &.{ refs, "main" }), .data = sha ++ "\n" });
            try std.Io.Dir.cwd().writeFile(ctx.io, .{ .sub_path = try std.fs.path.join(ctx.a, &.{ snap, "config.json" }), .data = "{\"model_type\":\"qwen3_5\"}" });
        }
        return c.pull_code;
    }

    fn findHook(ctx: *const tfq3.Context, explicit: ?[]const u8) anyerror!?[]const u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        if (explicit) |p| return p;
        return c.converter;
    }

    fn spawnHook(ctx: *const tfq3.Context, argv: []const []const u8) anyerror!u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        c.argv = try ctx.a.dupe([]const u8, argv);
        c.spawns += 1;
        return 0;
    }
};

fn makeCtx(a: std.mem.Allocator, env: *const Environ, cache_root: []const u8, cap: *Capture, out: *std.Io.Writer, err: *std.Io.Writer) tfq3.Context {
    return .{
        .a = a,
        .io = std.testing.io,
        .out = out,
        .err = err,
        .env = env,
        .cache_override = cache_root,
        .test_ctx = cap,
        .hooks = .{ .pull_fn = Capture.pullHook, .find_converter = Capture.findHook, .spawn_fn = Capture.spawnHook },
    };
}

test "parseArgs accepts shorthand, convert, inspect, doctor, and rejects invalid flags" {
    const a = tfq3.parseArgs(&.{"Qwen/Qwen3.8-27B"}).?;
    try std.testing.expectEqual(tfq3.Mode.convert, a.mode);
    try std.testing.expectEqualStrings("Qwen/Qwen3.8-27B", a.model);
    try std.testing.expectEqualStrings("4b", a.profile);

    const b = tfq3.parseArgs(&.{ "convert", "/m/q27", "--out", "/o/q27", "--profile", "3.5b", "--residual", "q8", "--head-bits", "5", "--skip-existing" }).?;
    try std.testing.expectEqualStrings("/m/q27", b.model);
    try std.testing.expectEqualStrings("/o/q27", b.out_dir.?);
    try std.testing.expectEqualStrings("3.5b", b.profile);
    try std.testing.expectEqual(tfq3.Residual.q8, b.residual);
    try std.testing.expectEqualStrings("5", b.head_bits);
    try std.testing.expect(b.skip_existing);

    const ins = tfq3.parseArgs(&.{ "inspect", "-v", "/o/q27" }).?;
    try std.testing.expectEqual(tfq3.Mode.inspect, ins.mode);
    try std.testing.expect(ins.verbose);
    try std.testing.expectEqualStrings("/o/q27", ins.model);

    try std.testing.expectEqual(tfq3.Mode.doctor, tfq3.parseArgs(&.{"doctor"}).?.mode);
    try std.testing.expectEqual(tfq3.Mode.help, tfq3.parseArgs(&.{"--help"}).?.mode);
    try std.testing.expect(tfq3.parseArgs(&.{}) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--profile", "9b" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--bogus" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{"inspect"}) == null);
}

test "tfq3 pulls an uncached repo id, detects qwen27, and spawns converter with exact args" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cache_root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/hub", .{tmp.sub_path});
    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: Capture = .{ .pull_creates_snapshot = true, .cache_root = cache_root };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = makeCtx(a, &env, cache_root, &cap, &out.writer, &err.writer);

    const code = try tfq3.runWith(&ctx, &.{"Qwen/Qwen3.8-27B"});
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expectEqualStrings("Qwen/Qwen3.8-27B", cap.pulled.?);
    try std.testing.expectEqual(@as(usize, 1), cap.spawns);
    try std.testing.expectEqualStrings("/fake/bin/tfq3", cap.argv[0]);
    try std.testing.expectEqualStrings("convert", cap.argv[1]);
    try std.testing.expect(std.mem.endsWith(u8, cap.argv[2], "0123456789abcdef0123456789abcdef01234567"));
    try std.testing.expectEqualStrings("--out", cap.argv[3]);
    try std.testing.expectEqualStrings("./Qwen3.8-27B-TFQ3", cap.argv[4]);
    try std.testing.expectEqualStrings("--family", cap.argv[6]);
    try std.testing.expectEqualStrings("qwen27", cap.argv[7]);
    try std.testing.expectEqualStrings("--residual", cap.argv[10]);
    try std.testing.expectEqualStrings("bf16", cap.argv[11]);
}

test "tfq3 refuses unsupported model_type and reports missing converter with install command" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const bad_dir = try std.fs.path.join(a, &.{ root, "unsupported" });
    const q27_dir = try std.fs.path.join(a, &.{ root, "qwen27-q8" });
    try std.Io.Dir.cwd().createDirPath(io, bad_dir);
    try std.Io.Dir.cwd().createDirPath(io, q27_dir);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ bad_dir, "config.json" }), .data = "{\"model_type\":\"llama\"}" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ q27_dir, "config.json" }), .data = "{\"model_type\":\"qwen3_5\",\"quantization\":{\"bits\":8,\"group_size\":64}}" });

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();

    var cap1: Capture = .{};
    var out1: std.Io.Writer.Allocating = .init(a);
    var err1: std.Io.Writer.Allocating = .init(a);
    const ctx1 = makeCtx(a, &env, root, &cap1, &out1.writer, &err1.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx1, &.{bad_dir}));
    try std.testing.expectEqual(@as(usize, 0), cap1.spawns);
    try std.testing.expect(std.mem.indexOf(u8, err1.written(), "unsupported model_type 'llama'") != null);

    var cap2: Capture = .{ .converter = null };
    var out2: std.Io.Writer.Allocating = .init(a);
    var err2: std.Io.Writer.Allocating = .init(a);
    const ctx2 = makeCtx(a, &env, root, &cap2, &out2.writer, &err2.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx2, &.{q27_dir}));
    try std.testing.expect(std.mem.indexOf(u8, err2.written(), tfq3.install_cmd) != null);

    var out3: std.Io.Writer.Allocating = .init(a);
    var err3: std.Io.Writer.Allocating = .init(a);
    const ctx3 = makeCtx(a, &env, root, &cap2, &out3.writer, &err3.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx3, &.{ q27_dir, "--dry-run" }));
    try std.testing.expect(std.mem.indexOf(u8, out3.written(), "--residual q8") != null);
}

test "tfq3 native inspect validates PWV3 + TFPN/TFPW shards and rejects corrupted alignment" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/ckpt", .{tmp.sub_path});
    try std.Io.Dir.cwd().createDirPath(io, dir);

    var tbl = try a.alloc(u8, 16384);
    @memset(tbl, 0);
    std.mem.writeInt(u32, tbl[0..4], 0x33565750, .little);
    std.mem.writeInt(u32, tbl[4..8], 3, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "pw_v3_tables.bin" }), .data = tbl });

    var shard = try a.alloc(u8, 16384);
    @memset(shard, 0);
    std.mem.writeInt(u32, shard[0..4], 0x4E504654, .little);
    std.mem.writeInt(u32, shard[4..8], 3, .little);
    std.mem.writeInt(u32, shard[8..12], 12, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "nonmoe.tffpw" }), .data = shard });

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = makeCtx(a, &env, dir, &cap, &out.writer, &err.writer);

    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx, &.{ "inspect", dir }));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "TFPN v3 | 12 tensors") != null);

    var moe_shard = try a.alloc(u8, 16384);
    @memset(moe_shard, 0);
    std.mem.writeInt(u32, moe_shard[0..4], 0x57504654, .little);
    std.mem.writeInt(u32, moe_shard[4..8], 3, .little);
    std.mem.writeInt(u32, moe_shard[8..12], 3, .little);
    std.mem.writeInt(u32, moe_shard[12..16], 288, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "layer_03.tffpw" }), .data = moe_shard });
    var out_moe: std.Io.Writer.Allocating = .init(a);
    var err_moe: std.Io.Writer.Allocating = .init(a);
    const ctx_moe = makeCtx(a, &env, dir, &cap, &out_moe.writer, &err_moe.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx_moe, &.{ "inspect", dir }));
    try std.testing.expect(std.mem.indexOf(u8, out_moe.written(), "TFPW v3 | L03 | 288 experts") != null);

    // Negative control: unaligned shard must fail with nonzero exit code.
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "bad.tffpw" }), .data = shard[0..100] });
    var out_bad: std.Io.Writer.Allocating = .init(a);
    var err_bad: std.Io.Writer.Allocating = .init(a);
    const ctx_bad = makeCtx(a, &env, dir, &cap, &out_bad.writer, &err_bad.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx_bad, &.{ "inspect", dir }));
    try std.testing.expect(std.mem.indexOf(u8, err_bad.written(), "not 16 KiB-aligned") != null);
}

test "tfq3 routes flashnext and glm families and enforces --no-pull" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const fn_dir = try std.fs.path.join(a, &.{ root, "flashnext" });
    const glm_dir = try std.fs.path.join(a, &.{ root, "glm" });
    try std.Io.Dir.cwd().createDirPath(io, fn_dir);
    try std.Io.Dir.cwd().createDirPath(io, glm_dir);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ fn_dir, "config.json" }), .data = "{\"model_type\":\"qwen4_exp\"}" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ glm_dir, "config.json" }), .data = "{\"model_type\":\"glm5_next\"}" });

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();

    var cap_fn: Capture = .{};
    var out_fn: std.Io.Writer.Allocating = .init(a);
    var err_fn: std.Io.Writer.Allocating = .init(a);
    const ctx_fn = makeCtx(a, &env, root, &cap_fn, &out_fn.writer, &err_fn.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx_fn, &.{ fn_dir, "--ple-bits", "4", "--skip-existing" }));
    try std.testing.expectEqualStrings("flashnext", cap_fn.argv[7]);
    try std.testing.expectEqualStrings("--ple-bits", cap_fn.argv[8]);
    try std.testing.expectEqualStrings("4", cap_fn.argv[9]);
    try std.testing.expectEqualStrings("--skip-existing", cap_fn.argv[10]);

    var cap_glm: Capture = .{};
    var out_glm: std.Io.Writer.Allocating = .init(a);
    var err_glm: std.Io.Writer.Allocating = .init(a);
    const ctx_glm = makeCtx(a, &env, root, &cap_glm, &out_glm.writer, &err_glm.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx_glm, &.{ glm_dir, "--trace", "/tmp/r.txt" }));
    try std.testing.expectEqualStrings("glm", cap_glm.argv[7]);
    try std.testing.expectEqualStrings("--trace", cap_glm.argv[8]);

    var cap_np: Capture = .{};
    var out_np: std.Io.Writer.Allocating = .init(a);
    var err_np: std.Io.Writer.Allocating = .init(a);
    const ctx_np = makeCtx(a, &env, root, &cap_np, &out_np.writer, &err_np.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx_np, &.{ "Org/Uncached", "--no-pull" }));
    try std.testing.expect(cap_np.pulled == null);
    try std.testing.expect(std.mem.indexOf(u8, err_np.written(), "tensorfold pull Org/Uncached") != null);
}
