//! The Qwen3.8-27B-class witness path without a GPU: who takes it, which geometries, the load's builds, the digest.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const witness = @import("qwen27_witness.zig");
const t = std.testing;

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

const row: api.profile.Entry = .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = &.{"qwen3_5"}, .backend = "metal", .class = .arithmetic, .syntax = .one, .effect = "test arithmetic", .evidence = "test fixture" };
const Env = struct {
    var value: ?[:0]const u8 = null;
    fn read(_: [:0]const u8) ?[:0]const u8 {
        return value;
    }
};

/// The facts `bind` reads from the host and the Mac, fixed.
const Mac = struct {
    path: []const u8 = "qwen3_5-27b/metal/tp1/mlx-q4g64/text",
    kernels: [32]u8 = @splat(0x4b),
    device: []const u8 = "Apple M5 Max",
    tensor_units: bool = true,
    words: [8]u64 = .{ 32768, 4, 512, 16, 0, 0, 1, 0 },
};

fn opened(dir: []const u8, mac: Mac) !*witness.Witness {
    const w = try witness.Witness.open(t.allocator, t.io, dir, .{});
    witness.set(w, mac.path, mac.kernels, mac.device, mac.tensor_units, mac.words);
    w.backend.profile = .{ .family = "qwen3_5", .backend = "metal", .entries = &.{row}, .environment = Env.read };
    return w;
}

fn settle(w: *witness.Witness) api.execution_witness.Facts {
    w.wait();
    return w.source().facts();
}

/// The dense 27B's geometry, as `q.model.supported` qualifies it.
fn dense27() q.config.Config {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
}

test "execution witness: the Qwen3.8-27B-class path wants the flag, one Mac, text, nothing drafted, and no SenseNova" {
    const on: api.Open = .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true };
    try t.expect(witness.wanted(on));
    try t.expect(witness.wanted(.{ .dir = "m", .model_type = "qwen3_5_moe", .execution_witness = true }));
    try t.expect(!witness.wanted(.{ .dir = "m", .model_type = "qwen3_5" }));
    try t.expect(!witness.wanted(.{ .dir = "m", .model_type = "sensenova_loop", .execution_witness = true }));
    const variants = [_]api.Open{
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .tp = 2 },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .vision = .{} },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .drafter = "d" },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .mtp_drafts = 2 },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .learn = "x" },
    };
    for (variants) |o| try t.expect(!witness.wanted(o));
}

test "execution witness: the 9B, the 27B and the 122B-A10B take their paths; mixed widths, a tied head and others do not" {
    var c = dense27();
    try t.expectEqualStrings("qwen3_5-27b/metal/tp1/mlx-q4g64/text", witness.pathOf(c, 0).?);
    try t.expect(witness.pathOf(c, 0x1235) == null); // mixed widths
    c.tied_head = true;
    try t.expect(witness.pathOf(c, 0) == null);
    c = dense27();
    c.hidden = 4096;
    c.layers = 32;
    try t.expectEqualStrings("qwen3_5-9b/metal/tp1/mlx-q4g64/text", witness.pathOf(c, 0).?);
    c.layers = 64; // 9B widths at the 27B's depth
    try t.expect(witness.pathOf(c, 0) == null);
    c = dense27();
    c.hidden = 3072;
    c.layers = 48;
    c.experts = 256;
    try t.expectEqualStrings("qwen3_5_moe-122b-a10b/metal/tp1/mlx-q4g64/text", witness.pathOf(c, 0).?);
    c.hidden = 2048;
    c.layers = 40; // Qwen3.6-35B-A3B: qualified to serve, not on the witness's path
    try t.expect(witness.pathOf(c, 0) == null);
    c = dense27();
    c.looped = true;
    try t.expect(witness.pathOf(c, 0) == null);
    c = dense27();
    c.ranks = 2;
    try t.expect(witness.pathOf(c, 0) == null);
}

test "execution witness: the load's library builds are watched from begin until the host binds or releases it" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var c = try Checkpoint.init(arena.allocator(), 4096);
    defer c.tmp.cleanup();
    try t.expect(try witness.begin(t.allocator, t.io, .{ .dir = c.dir, .model_type = "qwen3_5" }) == null);
    const w = (try witness.begin(t.allocator, t.io, .{ .dir = c.dir, .model_type = "qwen3_5", .execution_witness = true })).?;
    try t.expect(mtl.built.watcher.load(.acquire) == &w.backend.built.?);
    mtl.built.report("kernel source", .{ 0, 0, 0 });
    try t.expectEqual(@as(usize, 1), w.backend.built.?.digests.items.len);
    witness.release(w);
    try t.expect(mtl.built.watcher.load(.acquire) == null);
}

test "execution witness: a shard rewrite, the builds, the GPU, the shape and the profile change the 27B-class digest" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Checkpoint.init(a, 1 << 20);
    defer c.tmp.cleanup();
    Env.value = null;
    const first = try opened(c.dir, .{});
    defer first.close();
    const base = try first.digest();
    try t.expectEqualSlices(u8, &base, &try first.digest());
    const variants = [_]Mac{
        .{ .path = "qwen3_5-9b/metal/tp1/mlx-q4g64/text" },
        .{ .kernels = @splat(0x4c) },
        .{ .device = "Apple M4 Max" },
        .{ .tensor_units = false },
        .{ .words = .{ 32768, 8, 512, 16, 0, 0, 1, 0 } },
        .{ .words = .{ 32768, 4, 512, 16, 1, 0, 1, 0 } },
        .{ .words = .{ 32768, 4, 512, 16, 0, 0, 1, 1 } },
    };
    for (variants) |mac| {
        const other = try opened(c.dir, mac);
        defer other.close();
        try t.expect(!std.mem.eql(u8, &base, &try other.digest()));
    }
    Env.value = "1";
    try t.expect(!std.mem.eql(u8, &base, &try first.digest()));
    Env.value = null;
    try c.shard(a, 1 << 20, 700_000, 0x12); // one byte deep in the payload, same size, same header
    try t.expectError(error.FilesChanged, first.digest());
}

test "execution witness: a 27B-class engine reads known on its path once begun, and a reload hashes the same" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var c = try Checkpoint.init(arena.allocator(), 4096);
    defer c.tmp.cleanup();
    const startup = try opened(c.dir, .{});
    defer startup.close();
    try t.expect(startup.source().facts() == .pending); // nothing hashes before the server is ready
    startup.source().start();
    const first = settle(startup).known;
    try t.expectEqualStrings("qwen3_5-27b/metal/tp1/mlx-q4g64/text", first.path);
    const reloaded = try opened(c.dir, .{});
    defer reloaded.close();
    reloaded.source().start();
    try t.expectEqualSlices(u8, &first.digest, &settle(reloaded).known.digest);
}
