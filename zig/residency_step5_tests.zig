//! Step5 cache and snapshot ownership meets an independent list of live allocation IDs.
const std = @import("std");
const mtl = @import("metal");
const state = @import("src/families/step5/state.zig");
const cfg = @import("src/families/step5/config.zig");
const snapshot = @import("src/families/step5/snapshot.zig");
const backend_module = @import("src/families/step5/backend.zig");
const Model = @import("src/families/step5/model.zig").Model;
const residency = @import("src/families/step5/residency.zig");
const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *Model,
    metal: *backend_module.Metal,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
};
const a = std.testing.allocator;

fn geometry() cfg.Config {
    var c: cfg.Config = .{ .hidden = 64, .layers = 2, .mtp = 1, .vocab = 16, .heads = 1, .kv_heads = 1, .head_dim = 32, .intermediate = 64, .moe_intermediate = 16, .shared_intermediate = 16, .experts = 4, .topk = 1, .window = 8, .index_heads = 1, .proxy = 32, .index_topk = 4, .region = 4 };
    for (c.layer[0..3], 0..) |*l, i| l.* = .{ .kind = if (i == 0) .sliding else .full, .span = 0, .theta = 0, .limit = 0, .shared_limit = 0, .moe = false };
    return c;
}

test "residency owner Step5: actual stream cache and early freed snapshot pair their callbacks" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "step5")).?;
        defer owner.deinit();
        const c = geometry();
        var cache = try state.Cache.init(device, &c, 32);
        defer cache.deinit();
        cache.setResident(owner.follow());
        owner.held.flush();
        try record.check();
        cache.len = 3;
        var snap = try snapshot.save(a, device, record.queue(), &c, &cache);
        defer snap.free();
        owner.held.flush();
        try record.check();
        snap.free();
        try record.check();
        snap.free();
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Step5: native inventory includes routers, both scratches, tower and future caches" {
    for ([_]bool{ false, true }) |images| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            var model: Model = undefined;
            model.device = device;
            model.queue = record.queue();
            model.config = geometry();
            model.checkpoint.shards = .empty;
            defer model.checkpoint.shards.deinit(a);
            const weight = try device.buffer(64, mtl.ResourceOptions.shared);
            defer weight.deinit();
            try model.checkpoint.shards.append(a, .{ .buffer = weight, .bytes = 64 });
            const router = try device.buffer(64, mtl.ResourceOptions.shared);
            defer router.deinit();
            var layer: [1]@import("src/families/step5/weights.zig").Layer = undefined;
            layer[0].ffn = .{ .moe = undefined };
            layer[0].ffn.moe.router = router;
            model.weights.layers = &layer;
            var vision: @import("src/families/step5/vision.zig").Vision = undefined;
            var tower: [3]?mtl.Buffer = @splat(null);
            defer for (tower) |buffer| if (buffer) |b| b.deinit();
            model.vision = null;
            if (images) {
                for (&tower) |*buffer| buffer.* = try device.buffer(64, mtl.ResourceOptions.shared);
                vision.owned = tower[0].?;
                vision.scratch = tower[1].?;
                vision.patches.buf = tower[2].?;
                model.vision = &vision;
            }
            const backend = try backend_module.Metal.init(a, &model, .{ .capacity = 32, .budget = 1 << 30 });
            defer backend.deinit();
            var host: Host = .{ .gpa = a, .io = std.testing.io, .model = &model, .metal = backend };
            try residency.hold(&host);
            defer residency.drop(&host);
            try record.checkInitial();
            try record.check();
            mtl.keepalive.Target.tick(&host.warm);
            try std.testing.expectEqual(@as(usize, 1), record.uses);
            try std.testing.expectEqual(@as(usize, 1), record.copies);
            const cache = try backend.createCache(32);
            defer a.destroy(cache);
            defer cache.deinit();
            host.resident.?.held.flush();
            try record.check();
            cache.len = 3;
            var snap = try snapshot.save(a, device, record.queue(), &model.config, cache);
            defer snap.free();
            host.resident.?.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}
