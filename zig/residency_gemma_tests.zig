//! Gemma residency inventories use the production host helper and actual cache/snapshot constructors.
const std = @import("std");
const mtl = @import("metal");
const a = std.testing.allocator;
const gemma = @import("src/families/gemma4/state.zig");
const gemma_cfg = @import("src/families/gemma4/config.zig");
const gemma_snap = @import("src/families/gemma4/snapshot.zig");
const gemma_backend = @import("src/families/gemma4/backend.zig");
const GemmaModel = @import("src/families/gemma4/model.zig").Model;
const gemma_residency = @import("src/families/gemma4/residency.zig");
const GemmaHost = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *GemmaModel,
    metal: *gemma_backend.Metal,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
    draft: ?*@import("src/families/gemma4/dflash.zig").Drafter = null,
    assist: ?*@import("src/families/gemma4/assistant.zig").Drafter = null,
};

test "residency owner Gemma4: actual stream caches and saved states join and leave with their owners" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var owner = (try mtl.resident.Resident.hold(
            a,
            std.testing.io,
            device,
            record.queue(),
            &.{},
            "gemma4",
        )).?;
        defer owner.deinit();
        const geo: gemma_cfg.Geometry = .{ .name = "owner fixture", .hidden = 8, .layers = 2, .intermediate = 16, .query_heads = 1, .sliding_kv_heads = 1, .global_kv_heads = 1 };
        var cache = try gemma.Cache.init(device, geo, 32);
        defer cache.deinit();
        cache.setResident(owner.follow());
        owner.held.flush();
        try record.check();
        cache.setLastTap(try device.buffer(32, mtl.ResourceOptions.shared));
        owner.held.flush();
        try record.check();
        cache.setLastTap(try device.buffer(64, mtl.ResourceOptions.shared));
        owner.held.flush();
        try record.check();
        cache.len = 3;
        var snap = try gemma_snap.save(a, device, record.queue(), &cache);
        defer snap.free();
        owner.held.flush();
        try record.check();
        snap.setDraft(try device.buffer(64, mtl.ResourceOptions.shared));
        owner.held.flush();
        try record.check();
        snap.setDraft(try device.buffer(128, mtl.ResourceOptions.shared));
        owner.held.flush();
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Gemma4: real host inventory and later stream constructors use the shared hook" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var model: GemmaModel = undefined;
        model.device = device;
        model.queue = record.queue();
        model.config.geo = .{ .name = "owner fixture", .hidden = 8, .layers = 2, .intermediate = 64, .query_heads = 1, .sliding_kv_heads = 1, .global_kv_heads = 1 };
        model.checkpoint.shards = .empty;
        model.weights.owned = .empty;
        defer model.weights.owned.deinit(a);
        const weights = try device.buffer(64, mtl.ResourceOptions.shared);
        defer weights.deinit();
        try model.weights.owned.append(a, weights);
        model.inv_sliding = try device.buffer(32, mtl.ResourceOptions.shared);
        defer model.inv_sliding.deinit();
        model.inv_global = try device.buffer(32, mtl.ResourceOptions.shared);
        defer model.inv_global.deinit();
        model.projections = .{ .geo = model.config.geo };
        defer model.projections.list.deinit(a);
        var projection: @import("src/core/lane_projection.zig").Projection = undefined;
        const sums = try device.buffer(64, mtl.ResourceOptions.shared);
        defer sums.deinit();
        projection.sums_buffer = sums;
        try model.projections.list.append(a, .{ .x = undefined, .p = projection });
        model.vision = null;
        const backend = try gemma_backend.Metal.init(a, std.testing.io, &model, .{ .capacity = 32 });
        defer backend.deinit();
        var host: GemmaHost = .{ .gpa = a, .io = std.testing.io, .model = &model, .metal = backend };
        try gemma_residency.hold(&host);
        defer gemma_residency.drop(&host);
        try record.checkInitial();
        const owner = &host.resident.?;
        owner.held.flush();
        try record.check();
        const cache = try backend.createCache(32);
        defer a.destroy(cache);
        defer cache.deinit();
        owner.held.flush();
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Gemma4: both real drafter lane constructors pair joins and removals" {
    inline for (.{ @import("src/families/gemma4/dflash.zig").Drafter, @import("src/families/gemma4/assistant.zig").Drafter }) |D| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "gemma4")).?;
            defer owner.deinit();
            var model: GemmaModel = undefined;
            model.device = device;
            model.config.geo.hidden = 8;
            var draft: D = undefined;
            draft.gpa = a;
            draft.target = &model;
            draft.follow = owner.follow();
            if (@hasField(D, "options")) {
                draft.config.layers = 2;
                draft.config.window = 8;
                draft.config.kv_heads = 1;
                draft.config.head_dim = 32;
                draft.config.sliding[0..2].* = .{ true, false };
            }
            {
                const lane = if (@hasField(D, "options")) try draft.newLane(32) else try draft.newLane();
                defer draft.dropLane(lane);
                owner.held.flush();
                try record.check();
            }
            try record.check();
        }
        try record.allFreed();
    }
}
