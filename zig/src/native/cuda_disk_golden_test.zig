//! Kept keys stay byte for byte: the identity, however it is written, matches the sequence main hashed.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const disk = @import("cuda_disk.zig");
const identity = @import("disk_identity.zig");
const t = std.testing;

const Driver = struct {
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        @memcpy(out[0..5], "fake\x00");
        return 0;
    }
    fn attribute(out: *c_int, which: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = if (which == .compute_capability_major) 12 else 1;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 13000;
        return 0;
    }
    fn environment(_: [:0]const u8) ?[:0]const u8 {
        return "1";
    }
};

const row: api.profile.Entry = .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = &.{"qwen3_5"}, .class = .arithmetic, .syntax = .one, .effect = "test arithmetic", .evidence = "test fixture" };
const profile: identity.Profile = .{ .family = "qwen3_5", .backend = "cuda", .entries = &.{row}, .environment = Driver.environment };
/// The key main (93a2e93ec) gives the fixed inputs below.
const main_key: u64 = 9113307822166948912;
const manifest = "{\"kernels\":[{\"fn\":\"f\",\"hash\":\"h\",\"name\":\"k\",\"num_warps\":4,\"params\":[],\"consts\":{}}]}";

/// main's identityOf, update for update, with the binary's digest and the device's facts given.
fn reference(a: std.mem.Allocator, binary: [32]u8, dir: []const u8, kernels: ?[]const u8, facts: []const u64, probed: u64) !u64 {
    var h = std.hash.Wyhash.init(0x6375);
    @import("sys").file_identity.bind(&h, binary);
    try identity.checkpoint(&h, a, t.io, dir);
    inline for (comptime std.meta.declarations(cuda.kernels)) |d| {
        const v = @field(cuda.kernels, d);
        if (comptime @TypeOf(v) == []const u8) h.update(v);
    }
    if (kernels) |k| {
        const captured = try cuda.aot.identity.hash(a, t.io, k);
        h.update(std.mem.asBytes(&captured));
    }
    h.update("fake");
    const device = [_]u64{ 121, 13000 };
    h.update(std.mem.sliceAsBytes(&device));
    h.update(std.mem.sliceAsBytes(facts));
    h.update(std.mem.asBytes(&probed));
    identity.profile(&h, profile);
    return h.final();
}

/// A checkpoint folder whose identity reads no stamps (config.json alone), and a captured set beside it.
const Fixture = struct {
    tmp: t.TmpDir,
    dir: []const u8,
    kernels: []const u8,

    fn init(a: std.mem.Allocator) !Fixture {
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.createDir(t.io, "model", .default_dir);
        try tmp.dir.writeFile(t.io, .{ .sub_path = "model/config.json", .data = "{\"model_type\":\"qwen3_5\"}" });
        try tmp.dir.createDir(t.io, "aot", .default_dir);
        try tmp.dir.createDir(t.io, "aot/cubins", .default_dir);
        try tmp.dir.writeFile(t.io, .{ .sub_path = "aot/aot.json", .data = manifest });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "aot/cubins/h.cubin", .data = "cubin bytes" });
        const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        return .{ .tmp = tmp, .dir = try std.fs.path.join(a, &.{ root, "model" }), .kernels = try std.fs.path.join(a, &.{ root, "aot" }) };
    }
};

test "disk identity: identityOf hashes main's sequence over this binary, a checkpoint and a captured set" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "model/model.safetensors", .data = "shard" }); // its stamps are hashed too
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Driver.name;
    driver.api.cuDeviceGetAttribute = Driver.attribute;
    driver.api.cuDriverGetVersion = Driver.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const exe = try std.process.executablePathAlloc(t.io, a);
    const binary = try @import("sys").file_identity.hash(t.io, exe, 1 << 31);
    for ([_]?[]const u8{ null, f.kernels }) |kernels|
        try t.expectEqual(try reference(a, binary, f.dir, kernels, &.{ 1, 2, 3 }, 42), try disk.identityOf(a, t.io, &gpu, f.dir, kernels, &.{ 1, 2, 3 }, 42, profile));
}

test "disk identity: a fixed binary, checkpoint, captured set, device, facts, probe and profile give main's key" {
    if (cuda.kernels.available) return error.SkipZigTest; // the constant is a host build's, whose kernel images are empty
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try t.expectEqual(main_key, try reference(a, @splat(0x5a), f.dir, f.kernels, &.{ 1, 2, 3 }, 42));
    var h = std.hash.Wyhash.init(0x6375);
    try disk.components(&h, a, t.io, fixed(f));
    try t.expectEqual(main_key, h.final());
}

fn fixed(f: Fixture) disk.Components {
    return .{ .binary = @splat(0x5a), .checkpoint = f.dir, .kernels = f.kernels, .device = .{ .name = "fake", .capability = 121, .driver = 13000 }, .facts = &.{ 1, 2, 3 }, .probed = 42, .profile = profile };
}

fn sha(a: std.mem.Allocator, c: disk.Components) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    try disk.components(&h, a, t.io, c);
    return h.finalResult();
}

test "disk identity: the same components through SHA-256 move with a captured cubin, a profile value and the device" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    var c = fixed(f);
    c.binary = null; // the witness binds the binary and the checkpoint's bytes elsewhere
    c.checkpoint = null;
    c.probed = null;
    const base = try sha(a, c);
    try t.expectEqualSlices(u8, &base, &try sha(a, c));
    try f.tmp.dir.writeFile(t.io, .{ .sub_path = "aot/cubins/h.cubin", .data = "other cubin" });
    const cubin = try sha(a, c);
    try t.expect(!std.mem.eql(u8, &base, &cubin));
    var off = c;
    off.profile.environment = Unset.environment;
    try t.expect(!std.mem.eql(u8, &cubin, &try sha(a, off)));
    var other = c;
    other.device.capability = 80;
    try t.expect(!std.mem.eql(u8, &cubin, &try sha(a, other)));
}

const Unset = struct {
    fn environment(_: [:0]const u8) ?[:0]const u8 {
        return null;
    }
};
