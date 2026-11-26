//! Host tests for `tensorfold tfq3`: CLI parsing, pull/routing, generic models, and native inspection.
const std = @import("std");
const tfq3 = @import("tfq3.zig");
const glm_check = @import("glm_tfq3_check").tfq3_check;
const Environ = std.process.Environ.Map;

pub const Capture = struct {
    pulled: ?[]const u8 = null,
    pull_code: u8 = 0,
    pull_creates_snapshot: bool = false,
    pulls: usize = 0,
    cache_root: []const u8 = "",
    converter: ?[]const u8 = "/fake/bin/tfq3",
    argv: []const []const u8 = &.{},
    spawns: usize = 0,

    pub fn pullHook(ctx: *const tfq3.Context, spec: []const u8) anyerror!u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        c.pulls += 1;
        c.pulled = try ctx.a.dupe(u8, spec);
        if (c.pull_creates_snapshot and c.pull_code == 0) {
            const at = std.mem.indexOfScalar(u8, spec, '@');
            const repo = if (at) |idx| spec[0..idx] else spec;
            const slash = std.mem.indexOfScalar(u8, repo, '/').?;
            const sha = "0123456789abcdef0123456789abcdef01234567";
            const revision = if (at) |idx| spec[idx + 1 ..] else sha;
            const dir_name = try std.fmt.allocPrint(ctx.a, "models--{s}--{s}", .{ repo[0..slash], repo[slash + 1 ..] });
            const snap = try std.fs.path.join(ctx.a, &.{ c.cache_root, dir_name, "snapshots", revision });
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
        if (explicit) |p| return if (std.mem.eql(u8, p, "/missing/tfq3")) null else p;
        return c.converter;
    }

    pub fn spawnHook(ctx: *const tfq3.Context, argv: []const []const u8) anyerror!u8 {
        const c: *Capture = @ptrCast(@alignCast(ctx.test_ctx));
        c.argv = try ctx.a.dupe([]const u8, argv);
        c.spawns += 1;
        return 0;
    }
};

pub fn makeCtx(a: std.mem.Allocator, env: *const Environ, cache_root: []const u8, cap: *Capture, out: *std.Io.Writer, err: *std.Io.Writer) tfq3.Context {
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

fn f32ToBf16Bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

pub fn writeSyntheticBf16Ckpt(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: usize) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = try std.fmt.allocPrint(a, "{{\"model_type\":\"llama\",\"hidden_size\":{d}}}", .{rows}) });
    const cols: usize = 256;
    const w_bytes = rows * cols * 2;
    const n_bytes = rows * 2;
    const hdr = try std.fmt.allocPrint(
        a,
        "{{\"model.layers.0.mlp.down_proj.weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"model.norm.weight\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}}}",
        .{ rows, cols, w_bytes, rows, w_bytes, w_bytes + n_bytes },
    );
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    const total = 8 + hdr.len + w_bytes + n_bytes;
    const buf = try a.alloc(u8, total);
    std.mem.writeInt(u64, buf[0..8], hdr.len, .little);
    @memcpy(buf[8 .. 8 + hdr.len], hdr);
    var pos: usize = 8 + hdr.len;
    for (0..(rows * cols)) |_| {
        const u1_v = @max(rand.float(f64), 1e-9);
        const u2_v = rand.float(f64);
        const g: f32 = @floatCast(@sqrt(-2.0 * @log(u1_v)) * @cos(2.0 * std.math.pi * u2_v) * 0.05);
        std.mem.writeInt(u16, buf[pos..][0..2], f32ToBf16Bits(g), .little);
        pos += 2;
    }
    for (0..rows) |_| {
        std.mem.writeInt(u16, buf[pos..][0..2], f32ToBf16Bits(1.0), .little);
        pos += 2;
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors" }), .data = buf });
}

pub fn writeValidDenseShard(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const bytes = try a.alloc(u8, 49152);
    @memset(bytes, 0);
    const h: glm_check.NonMoeHeader = .{ .magic = glm_check.magic_nonmoe, .version = 3, .num_tensors = 1, .pad0 = 0, .off_meta = 128, .off_scales = 16384, .off_weights = 32768, .total_scale_halfs = 1, .total_weight_words = 2048, .pad1 = 0 };
    var m = std.mem.zeroes(glm_check.NonMoeTensorMeta);
    @memcpy(m.name[0..7], "lm_head");
    m.R = 128;
    m.C = 128;
    m.b_bits = 4;
    m.word_count = 2048;
    @memcpy(bytes[0..@sizeOf(glm_check.NonMoeHeader)], std.mem.asBytes(&h));
    @memcpy(bytes[128..][0..@sizeOf(glm_check.NonMoeTensorMeta)], std.mem.asBytes(&m));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "nonmoe.tffpw" }), .data = bytes });
}

pub fn writeValidLayerShard(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, "layer_03.tffpw" });
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    const h: glm_check.LayerHeader = .{ .magic = glm_check.magic_layer, .version = 3, .layer_idx = 3, .num_experts = 1, .num_tensors = 3, .hidden_size = 4096, .moe_inter = 2048, .stream_words = 3 * 1048576, .scale_halfs = 3 * 512, .off_meta = 160, .off_scales = 16384, .off_weights = 32768, .meta_tag = @splat(0), .reserved0 = 0, .raw_fp8_bytes = 0, .reserved1 = @splat(0) };
    var m = std.mem.zeroes(glm_check.ExpertMeta);
    m.b_bits = 4;
    m.up_word_off = 1048576;
    m.down_word_off = 2 * 1048576;
    m.up_sc_off = 512;
    m.down_sc_off = 1024;
    try file.writePositionalAll(io, std.mem.asBytes(&h), 0);
    try file.writePositionalAll(io, std.mem.asBytes(&m), 160);
    try file.setLength(io, 32768 + 3 * 1048576 * 4);
}

test "parseArgs accepts generic/native/budget flags and rejects invalid inputs" {
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

    const g = tfq3.parseArgs(&.{ "/m/llama", "--bits", "3", "--target-gib", "4.5", "--fit", "auto", "--quantize-vision", "--quantize-drafter", "--quantize-embed", "--native" }).?;
    try std.testing.expectEqualStrings("3", g.bits.?);
    try std.testing.expectEqualStrings("4.5", g.target_gib.?);
    try std.testing.expectEqualStrings("auto", g.fit.?);
    try std.testing.expect(g.quantize_vision and g.quantize_drafter and g.quantize_embed and g.native);
    try std.testing.expect(g.wantsGeneric());

    const ins = tfq3.parseArgs(&.{ "inspect", "-v", "/o/q27" }).?;
    try std.testing.expectEqual(tfq3.Mode.inspect, ins.mode);
    try std.testing.expect(ins.verbose);
    try std.testing.expectEqualStrings("/o/q27", ins.model);

    try std.testing.expectEqual(tfq3.Mode.doctor, tfq3.parseArgs(&.{"doctor"}).?.mode);
    try std.testing.expectEqual(tfq3.Mode.help, tfq3.parseArgs(&.{"--help"}).?.mode);
    try std.testing.expect(tfq3.parseArgs(&.{}) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--profile", "9b" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--bits", "2" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--target-gib", "-1" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--target-gib", "1e30" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--fit", "1e30" }) == null);
    try std.testing.expect(tfq3.parseArgs(&.{ "Qwen/Qwen3.8-27B", "--fit", "bad" }) == null);
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

test "tfq3 guaranteed native refusal does not pull or mutate an uncached repo" {
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

    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx, &.{ "Owner/Uncached", "--native" }));
    try std.testing.expectEqual(@as(usize, 0), cap.pulls);
    try std.testing.expect(cap.pulled == null);
    try std.testing.expectEqual(@as(usize, 0), cap.spawns);
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "production loader") != null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openDir(std.testing.io, cache_root, .{}));
}

test "tfq3 uncached dry-run keeps its pull plan without pulling" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cache_root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/hub", .{tmp.sub_path});
    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: Capture = .{ .converter = null, .pull_creates_snapshot = true, .cache_root = cache_root };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = makeCtx(a, &env, cache_root, &cap, &out.writer, &err.writer);

    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx, &.{ "Owner/Uncached", "--dry-run" }));
    try std.testing.expectEqual(@as(usize, 0), cap.pulls);
    try std.testing.expectEqual(@as(usize, 0), cap.spawns);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "[dry-run] pull Owner/Uncached") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "source:    <pull:Owner/Uncached>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "family:    auto (model_type=unknown)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "converter: native (refused: no production loader)") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "[dry-run] tfq3 convert <pull:Owner/Uncached>") != null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openDir(std.testing.io, cache_root, .{}));
}

test "tfq3 routes unknown model_type to generic, fails closed on missing explicit converter, and supports doctor fallback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const llama_dir = try std.fs.path.join(a, &.{ root, "llama" });
    const q27_dir = try std.fs.path.join(a, &.{ root, "qwen27-q8" });
    try std.Io.Dir.cwd().createDirPath(io, llama_dir);
    try std.Io.Dir.cwd().createDirPath(io, q27_dir);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ llama_dir, "config.json" }), .data = "{\"model_type\":\"llama\"}" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ q27_dir, "config.json" }), .data = "{\"model_type\":\"qwen3_5\",\"quantization\":{\"bits\":8,\"group_size\":64}}" });

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();

    var cap1: Capture = .{};
    var out1: std.Io.Writer.Allocating = .init(a);
    var err1: std.Io.Writer.Allocating = .init(a);
    const ctx1 = makeCtx(a, &env, root, &cap1, &out1.writer, &err1.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx1, &.{ llama_dir, "--target-gib", "5.0", "--quantize-embed" }));
    try std.testing.expectEqual(@as(usize, 1), cap1.spawns);
    try std.testing.expectEqualStrings("generic", cap1.argv[7]);

    // Explicit missing --converter must fail closed with exit 1.
    var cap2: Capture = .{ .converter = null };
    var out2: std.Io.Writer.Allocating = .init(a);
    var err2: std.Io.Writer.Allocating = .init(a);
    const ctx2 = makeCtx(a, &env, root, &cap2, &out2.writer, &err2.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx2, &.{ q27_dir, "--converter", "/missing/tfq3" }));
    try std.testing.expect(std.mem.indexOf(u8, err2.written(), "/missing/tfq3") != null);

    // Dry-run with missing auto converter reports native Zig fallback and detects q8 residual.
    var out3: std.Io.Writer.Allocating = .init(a);
    var err3: std.Io.Writer.Allocating = .init(a);
    const ctx3 = makeCtx(a, &env, root, &cap2, &out3.writer, &err3.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx3, &.{ q27_dir, "--dry-run" }));
    try std.testing.expect(std.mem.indexOf(u8, out3.written(), "--residual q8") != null);
    try std.testing.expect(std.mem.indexOf(u8, out3.written(), "native (refused: no production loader)") != null);

    // Doctor without external tfq3 must not advertise an unloadable native layout.
    var out4: std.Io.Writer.Allocating = .init(a);
    var err4: std.Io.Writer.Allocating = .init(a);
    const ctx4 = makeCtx(a, &env, root, &cap2, &out4.writer, &err4.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx4, &.{"doctor"}));
    try std.testing.expect(std.mem.indexOf(u8, out4.written(), "native Zig TFQ3 encoder: unavailable") != null);
    try std.testing.expect(std.mem.indexOf(u8, out4.written(), "main/install.sh") == null);
    try std.testing.expect(std.mem.indexOf(u8, out4.written(), "| bash") == null);
    try std.testing.expect(std.mem.indexOf(u8, out4.written(), "TFQ#installation") != null);
}

test "tfq3 generic native fallback refuses before creating unloadable output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const src_dir = try std.fs.path.join(a, &.{ root, "mini_bf16" });
    const out_dir = try std.fs.path.join(a, &.{ root, "mini_tfq3" });
    try writeSyntheticBf16Ckpt(a, io, src_dir, 128);

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = makeCtx(a, &env, root, &cap, &out.writer, &err.writer);
    try std.testing.expectEqual(@as(u8, 1), try tfq3.runWith(&ctx, &.{ src_dir, "--out", out_dir, "--bits", "4" }));
    try std.testing.expectEqual(@as(usize, 0), cap.spawns);
    try std.testing.expect(std.mem.indexOf(u8, err.written(), "production loader") != null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openDir(io, out_dir, .{}));
}

test "tfq3 native inspect validates complete PWV3 + TFPN/TFPW shards and rejects corruption" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/ckpt", .{tmp.sub_path});
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const tbl = try a.alloc(u8, 16384);
    @memset(tbl, 0);
    std.mem.writeInt(u32, tbl[0..4], 0x33565750, .little);
    std.mem.writeInt(u32, tbl[4..8], 3, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "pw_v3_tables.bin" }), .data = tbl });
    try writeValidDenseShard(a, io, dir);
    try writeValidLayerShard(a, io, dir);

    var env: Environ = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: Capture = .{ .converter = null };
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = makeCtx(a, &env, dir, &cap, &out.writer, &err.writer);
    const valid_code = try tfq3.runWith(&ctx, &.{ "inspect", dir });
    if (valid_code != 0) std.debug.print("valid inspect failed: {s}\n", .{err.written()});
    try std.testing.expectEqual(@as(u8, 0), valid_code);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "TFPN v3 | 1 tensors") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "TFPW v3 | L03 | 1 experts") != null);

    const dense_path = try std.fs.path.join(a, &.{ dir, "nonmoe.tffpw" });
    const bad = try std.Io.Dir.cwd().readFileAlloc(io, dense_path, a, .limited(64 << 10));
    std.mem.writeInt(u64, bad[128 + 96 ..][0..8], 2048, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dense_path, .data = bad[0..100] });
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
