//! The Metal witness path without a GPU: who takes it, every weight byte, the kernel sources, the Mac's facts, reloads.
const std = @import("std");
const api = @import("engine_api");
const q = @import("tensorfold").qwen35;
const witness = @import("metal_witness.zig");
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

/// This Mac's facts as `bind` reads them, fixed.
const Mac = struct {
    kernels: [32]u8 = @splat(0x4b),
    device: []const u8 = "Apple M1 Max",
    tensor_units: bool = false,
    compile: [3]u64 = .{ 0, 1, 0x40000 },
    words: [5]u64 = .{ 4096, 8, 128, 4, 0 },
};

/// A witness opened over `dir` and given `mac`'s facts, as `bind` gives them.
fn opened(dir: []const u8, mac: Mac) !*witness.Witness {
    const w = try witness.open(t.allocator, t.io, dir);
    witness.set(w, mac.kernels, mac.device, mac.tensor_units, mac.compile, mac.words);
    w.backend.profile = .{ .family = "qwen3_5", .backend = "metal", .entries = &.{row}, .environment = Env.read };
    return w;
}

fn settle(w: *witness.Witness) api.execution_witness.Facts {
    w.wait();
    return w.source().facts();
}

test "execution witness: the Metal path wants the flag, one Mac, text and nothing learned, drafted or recorded" {
    const on: api.Open = .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true };
    try t.expect(witness.wanted(on));
    try t.expect(!witness.wanted(.{ .dir = "m", .model_type = "qwen3_5" }));
    const variants = [_]api.Open{
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .tp = 2 },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .vision = .{} },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .learn = "x" },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .drafter = "d" },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .slide = true },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .record = "r" },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .recall = true },
        .{ .dir = "m", .model_type = "qwen3_5", .execution_witness = true, .mtp_drafts = 2 },
    };
    for (variants) |o| try t.expect(!witness.wanted(o));
}

test "execution witness: the Metal kernel identity covers every pipeline the load compiles, in one stable digest" {
    var functions: usize = 0;
    for (q.kernels.libraries) |l| functions += l.functions.len;
    try t.expectEqual(q.kernels.total, functions); // a pipeline outside `libraries` would load unhashed
    try t.expectEqualSlices(u8, &witness.kernels(), &witness.kernels());
}

test "execution witness: a Metal shard rewrite, the kernels, the GPU, the compile options, the shape and the profile change the digest" {
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
        .{ .kernels = @splat(0x4c) },
        .{ .device = "Apple M5 Max" },
        .{ .tensor_units = true },
        .{ .compile = .{ 0, 1, 0x40001 } },
        .{ .words = .{ 4096, 8, 128, 1, 0 } },
        .{ .words = .{ 4096, 8, 128, 4, 1 } },
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
    try t.expectError(error.FilesChanged, first.digest()); // the bytes the load read are not the bytes on disk now
    const reloaded = try opened(c.dir, .{});
    defer reloaded.close();
    try t.expect(!std.mem.eql(u8, &base, &try reloaded.digest()));
}

test "execution witness: a Metal engine reads known on its path once begun, a reload hashes again, a moved shard reads files_changed" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Checkpoint.init(a, 4096);
    defer c.tmp.cleanup();
    const startup = try opened(c.dir, .{});
    defer startup.close();
    try t.expect(startup.source().facts() == .pending); // nothing hashes before the server is ready
    startup.source().start();
    const first = settle(startup).known;
    try t.expectEqualStrings("qwen3_5-2b/metal/tp1/mlx-q4g64/text", first.path);
    const reloaded = try opened(c.dir, .{});
    defer reloaded.close();
    try t.expect(reloaded.source().facts() == .pending);
    reloaded.source().start();
    try t.expectEqualSlices(u8, &first.digest, &settle(reloaded).known.digest);
    const moved = try opened(c.dir, .{});
    defer moved.close();
    try c.shard(a, 4096, 10, 0x13);
    moved.source().start();
    try t.expectEqual(api.execution_witness.Reason.files_changed, settle(moved).unknown);
}

test "execution witness: Metal files that cannot be stamped read hash_failed and the load goes on" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var c = try Checkpoint.init(arena.allocator(), 4096);
    defer c.tmp.cleanup();
    try c.tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{" }); // names no shard it can read
    try t.expect(std.meta.isError(witness.open(t.allocator, t.io, c.dir)));
    const log_level = std.testing.log_level;
    std.testing.log_level = .err; // the expected hash_failed warning stays out of the test output
    defer std.testing.log_level = log_level;
    const w = try witness.openOrFailed(t.allocator, t.io, c.dir);
    defer w.close();
    const mac: Mac = .{};
    witness.set(w, mac.kernels, mac.device, mac.tensor_units, mac.compile, mac.words);
    w.source().start(); // nothing to read: it is already unknown
    try t.expectEqual(api.execution_witness.Reason.hash_failed, settle(w).unknown);
    try t.expect(witness.source(null) == null);
}
