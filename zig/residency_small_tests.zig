//! Small Metal family storage owners follow real allocations while the fake device records unique IDs.
const std = @import("std");
const mtl = @import("metal");
const a = std.testing.allocator;

fn storageGate(comptime Storage: type) !void {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var storage: Storage = .{ .gpa = a, .device = record.device() };
        defer storage.deinit();
        const first = try storage.alloc(64);
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{first}, "small metal family")).?;
        defer owner.deinit();
        storage.setResident(owner.follow());
        defer storage.setResident(null);
        _ = try storage.alloc(128);
        owner.held.flush();
        try record.check();
        record.fail_at = record.allocations.items.len;
        try std.testing.expectError(error.InjectedBufferFailure, storage.alloc(256));
        record.fail_at = null;
        try record.check();
    }
    try record.allFreed();
}

test "residency owner small Qwen35: actual storage follows later buffers and allocation failures" {
    try storageGate(@import("src/families/qwen3_5/state.zig").Storage);
}

test "residency owner LFM2 MoE: actual storage follows later buffers and allocation failures" {
    try storageGate(@import("src/families/lfm2_moe/state.zig").Storage);
}

fn nativeGate(comptime family: type, comptime lfm: bool) !void {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var model: family.Model = undefined;
        model.device = device;
        model.queue = record.queue();
        if (lfm) model.resident = null;
        model.checkpoint.shards = .empty;
        defer model.checkpoint.shards.deinit(a);
        const shard = try device.buffer(64, 0);
        defer shard.deinit();
        try model.checkpoint.shards.append(a, .{ .buffer = shard, .bytes = 64 });
        var backend: family.backend.Metal = undefined;
        backend.gpa = a;
        backend.model = &model;
        backend.residency = null;
        backend.caches = .empty;
        defer backend.caches.deinit(a);
        if (lfm) {
            model.config.attention = @splat(false);
            model.config.attention[1] = true;
            backend.scratch = try family.state.Scratch.init(a, device, &model.config.attention, 1);
        } else backend.scratch = try family.state.Scratch.init(a, device, 1, 1);
        defer backend.scratch.deinit();
        const H = struct {
            gpa: std.mem.Allocator = a,
            model: *family.Model,
            metal: *family.backend.Metal,
            resident: ?mtl.resident.Resident = null,
            warm: mtl.keepalive.Target = undefined,
        };
        var h: H = .{ .model = &model, .metal = &backend };
        try mtl.resident.streams.hold(&h, std.testing.io, if (lfm) "lfm2 moe" else "qwen3.5");
        defer mtl.resident.streams.drop(&h);
        try record.checkInitial();
        try record.check();
        mtl.keepalive.Target.tick(&h.warm);
        try std.testing.expect(record.copies > 0 and record.uses > 0);
        var stream: @import("lanes").Stream = undefined;
        const cache = try backend.newCache(2);
        defer {
            _ = backend.caches.remove(&stream);
            cache.deinit();
            a.destroy(cache);
        }
        h.resident.?.held.flush();
        try record.check();
        try backend.caches.put(a, &stream, cache);
        record.fail_at = record.allocations.items.len;
        try std.testing.expectError(error.InjectedBufferFailure, backend.newCache(2));
        record.fail_at = null;
        try record.check();
        mtl.resident.streams.drop(&h);
        record.refuse_set = true;
        try mtl.resident.streams.hold(&h, std.testing.io, "small family");
        try std.testing.expect(h.resident == null and h.warm.scratch == null);
        try record.check();
    }
    try record.allFreed();
}

test "residency owner small Qwen35: real scratch, new stream factory and native helper include every live allocation" {
    try nativeGate(@import("src/families/qwen3_5/qwen3_5.zig"), false);
}

test "residency owner LFM2 MoE: real scratch, new stream factory and native helper include every live allocation" {
    try nativeGate(@import("src/families/lfm2_moe/lfm2_moe.zig"), true);
}
