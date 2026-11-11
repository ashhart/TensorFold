//! The first CUDA witness path without a GPU: who takes it, every weight byte, stamps, the kept components, reloads.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const core = @import("core");
const witness = @import("cuda_witness.zig");
const identity = @import("disk_identity.zig");
const t = std.testing;
const K = qwen.native.kept_disk;

const Driver = struct {
    var capability_minor: c_int = 0;
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        @memcpy(out[0..5], "fake\x00");
        return 0;
    }
    fn attribute(out: *c_int, which: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = switch (which) {
            .compute_capability_major => 8,
            .compute_capability_minor => capability_minor,
            .multiprocessor_count => 108,
            else => 0,
        };
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 12040;
        return 0;
    }
    fn context(driver: *cuda.Driver) cuda.Context {
        driver.api.cuDeviceGetName = name;
        driver.api.cuDeviceGetAttribute = attribute;
        driver.api.cuDriverGetVersion = version;
        return .{ .d = driver, .device = 0, .handle = null };
    }
};

/// The 2B's geometry as the loader leaves it for an MLX affine 4-bit checkpoint, on an undefined engine.
fn twoB(e: *qwen.Engine) qwen.reuse.Target {
    e.g.hidden = 2048;
    e.g.layers = 24;
    e.g.experts = 0;
    e.g.exl3 = false;
    e.g.bf16 = false;
    e.g.kinds = .{};
    e.g.bits = .{};
    e.g.rope = .plain;
    e.kv8 = false;
    e.n_taps = 0;
    e.int8.on = false;
    return .{ .e = e, .caches = undefined, .seen = undefined };
}

/// A checkpoint: config.json and one shard, a header and a payload of `size` bytes.
const Checkpoint = struct {
    tmp: t.TmpDir,
    dir: []const u8,

    fn init(a: std.mem.Allocator, size: usize) !Checkpoint {
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5\"}" });
        var c: Checkpoint = .{ .tmp = tmp, .dir = try tmp.dir.realPathFileAlloc(t.io, ".", a) };
        try c.shard(a, size, 0, null);
        return c;
    }
    /// The shard with every payload byte 0x11, or `byte` at `at`; a later mtime so no filesystem tick hides it.
    fn shard(c: *Checkpoint, a: std.mem.Allocator, size: usize, at: usize, byte: ?u8) !void {
        const header = try std.fmt.allocPrint(a, "{{\"w\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ size, size });
        const bytes = try a.alloc(u8, 8 + header.len + size);
        std.mem.writeInt(u64, bytes[0..8], header.len, .little);
        @memcpy(bytes[8..][0..header.len], header);
        @memset(bytes[8 + header.len ..], 0x11);
        if (byte) |b| bytes[8 + header.len + at] = b;
        const before = c.tmp.dir.statFile(t.io, "model.safetensors", .{}) catch null;
        try c.tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = bytes });
        if (before) |s| {
            const f = try c.tmp.dir.openFile(t.io, "model.safetensors", .{ .mode = .read_write });
            defer f.close(t.io);
            try f.setTimestamps(t.io, .{ .modify_timestamp = .{ .new = s.mtime.addDuration(.fromSeconds(1)) } });
        }
    }
};

const row: api.profile.Entry = .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = &.{"qwen3_5"}, .class = .arithmetic, .syntax = .one, .effect = "test arithmetic", .evidence = "test fixture" };
const Env = struct {
    var value: ?[:0]const u8 = null;
    fn read(_: [:0]const u8) ?[:0]const u8 {
        return value;
    }
};

/// A witness opened over `dir` and given the 2B's path and fixed device facts, as `bind` gives them.
fn opened(dir: []const u8, kernels: ?[]const u8, capability: u64) !*witness.Witness {
    const w = try witness.Witness.open(t.allocator, t.io, dir);
    w.set("qwen3_5-2b/cuda/tp1/mlx-q4g64/text", kernels, .{ .name = "fake", .capability = capability, .driver = 12040 }, &.{ 0, 0, 0, 7 }, .{ 108, 4096, 8, 1 });
    w.profile = .{ .family = "qwen3_5", .backend = "cuda", .entries = &.{row}, .environment = Env.read };
    return w;
}

/// The hook's answer once the begun hash has finished: a join, never a timed wait.
fn settle(w: *witness.Witness) api.execution_witness.Facts {
    w.wait();
    return w.source().facts();
}

test "execution witness: only dense Qwen with the flag, one rank, text and nothing learned wants the first path" {
    const on: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true };
    try t.expect(witness.wanted(qwen.native, on));
    try t.expect(!witness.wanted(qwen.native_moe, on));
    var off = on;
    off.execution_witness = false;
    try t.expect(!witness.wanted(qwen.native, off));
    const variants = [_]api.Open{
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .tp = 2 },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .vision = .{} },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .drafter = "x" },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .slide = true },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .record = "x" },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .learn = "x" },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .recall = true },
        .{ .dir = "d", .model_type = "qwen3_5", .execution_witness = true, .mtp_drafts = 2 },
    };
    for (variants) |o| try t.expect(!witness.wanted(qwen.native, o));
}

test "execution witness: the loaded 2B's uniform affine geometry gets a hook; other sizes, quants and formats none" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var c = try Checkpoint.init(arena.allocator(), 64);
    defer c.tmp.cleanup();
    var driver: cuda.Driver = undefined;
    const gpu = Driver.context(&driver);
    var e: qwen.Engine = undefined;
    const variants = [_]*const fn (*qwen.Engine) void{
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.exl3 = true;
            }
        }.f,
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.bf16 = true;
            }
        }.f,
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.bits.proj[3][0] = 8;
            }
        }.f,
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.kinds.modelopt = true;
            }
        }.f,
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.hidden = 5120;
                x.g.layers = 64;
            }
        }.f,
        struct {
            fn f(x: *qwen.Engine) void {
                x.g.experts = 256;
            }
        }.f,
    };
    {
        var target = twoB(&e);
        const w = try witness.Witness.open(t.allocator, t.io, c.dir);
        defer w.close();
        const got = witness.bound(qwen.native, w, .{ .target = &target }, &gpu, null, .{ 4096, 8, 1 }).?;
        try t.expectEqualStrings("qwen3_5-2b/cuda/tp1/mlx-q4g64/text", got.path.?);
        try t.expectEqual(@as(u64, 108), got.words[0]);
        try t.expect(witness.source(got) != null);
        try t.expectEqual(@as(?*witness.Witness, null), witness.bound(qwen.native_moe, w, .{ .target = &target }, &gpu, null, .{ 4096, 8, 1 }));
    }
    for (variants) |change| {
        var target = twoB(&e);
        change(&e);
        const w = try witness.Witness.open(t.allocator, t.io, c.dir);
        defer w.close();
        try t.expectEqual(@as(?*witness.Witness, null), witness.bound(qwen.native, w, .{ .target = &target }, &gpu, null, .{ 4096, 8, 1 }));
        try t.expect(witness.source(w) == null);
    }
    try t.expect(witness.source(null) == null);
}

test "execution witness: a same-size, same-header shard rewrite changes the weights digest; stamps moved since open are unknown" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Checkpoint.init(a, 1 << 20);
    defer c.tmp.cleanup();
    const structure = try core.checkpoint_identity.identity(a, t.io, c.dir);
    const first = try opened(c.dir, null, 80);
    defer first.close();
    const base = try first.digest();
    try t.expectEqualSlices(u8, &base, &try first.digest());
    try c.shard(a, 1 << 20, 700_000, 0x12); // one byte deep in the payload, same size, same header
    try t.expectEqualSlices(u8, &structure, &try core.checkpoint_identity.identity(a, t.io, c.dir)); // headers can't see it
    try t.expectError(error.FilesChanged, first.digest()); // the bytes the load read are not the bytes on disk now
    const reloaded = try opened(c.dir, null, 80);
    defer reloaded.close();
    try t.expect(!std.mem.eql(u8, &base, &try reloaded.digest()));
    // through the hook: pending until the server begins it, then unknown with the reason
    try c.shard(a, 1 << 20, 700_000, 0x13);
    try t.expect(reloaded.source().facts() == .pending);
    reloaded.source().start();
    try t.expectEqual(api.execution_witness.Reason.files_changed, settle(reloaded).unknown);
}

test "execution witness: a captured cubin, a resolved profile value or the device changes the engine digest" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Checkpoint.init(a, 4096);
    defer c.tmp.cleanup();
    try c.tmp.dir.createDir(t.io, "aot", .default_dir);
    try c.tmp.dir.createDir(t.io, "aot/cubins", .default_dir);
    try c.tmp.dir.writeFile(t.io, .{ .sub_path = "aot/aot.json", .data = "{\"kernels\":[{\"fn\":\"f\",\"hash\":\"h\",\"name\":\"k\",\"num_warps\":4,\"params\":[],\"consts\":{}}]}" });
    try c.tmp.dir.writeFile(t.io, .{ .sub_path = "aot/cubins/h.cubin", .data = "first cubin" });
    const kernels = try std.fs.path.join(a, &.{ c.dir, "aot" });
    Env.value = null;
    const w = try opened(c.dir, kernels, 80);
    defer w.close();
    const base = try w.digest();
    try c.tmp.dir.writeFile(t.io, .{ .sub_path = "aot/cubins/h.cubin", .data = "other cubin" });
    const cubin = try w.digest();
    try t.expect(!std.mem.eql(u8, &base, &cubin));
    Env.value = "1";
    defer Env.value = null;
    const profiled = try w.digest();
    try t.expect(!std.mem.eql(u8, &cubin, &profiled));
    const other = try opened(c.dir, kernels, 86);
    defer other.close();
    try t.expect(!std.mem.eql(u8, &profiled, &try other.digest()));
}

test "execution witness: each engine hashes once it is begun, and a reloaded one reads every byte again" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Checkpoint.init(a, 1 << 20);
    defer c.tmp.cleanup();
    const startup = try opened(c.dir, null, 80);
    defer startup.close();
    try t.expect(startup.source().facts() == .pending); // nothing hashes before the server is ready
    startup.source().start();
    startup.source().start(); // a second begin is the same hash
    const first = settle(startup).known;
    try t.expectEqualStrings("qwen3_5-2b/cuda/tp1/mlx-q4g64/text", first.path);
    const reloaded = try opened(c.dir, null, 80);
    defer reloaded.close();
    try t.expect(reloaded.source().facts() == .pending);
    reloaded.source().start();
    try t.expectEqualSlices(u8, &first.digest, &settle(reloaded).known.digest);
    // closing while a hash runs stops it at its next block and frees everything
    const busy = try opened(c.dir, null, 80);
    busy.source().start();
    busy.close();
}
