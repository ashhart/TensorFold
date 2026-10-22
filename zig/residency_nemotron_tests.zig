//! Nemotron allocation owners are compared with the complete fake device allocation list.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("src/families/nemotron/config.zig");
const state = @import("src/families/nemotron/state.zig");
const Model = @import("src/families/nemotron/model.zig").Model;
const backend_module = @import("src/families/nemotron/backend.zig");
const residency = @import("src/families/nemotron/residency.zig");
const Host = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    metal: *backend_module.Metal,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
};
const a = std.testing.allocator;

fn geometry() cfg.Config {
    var c: cfg.Config = .{ .hidden = 64, .vocab = 16, .layers = 3, .mamba_heads = 1, .mamba_head_dim = 64, .groups = 1, .state = 4, .conv_kernel = 3, .heads = 1, .kv_heads = 1, .head_dim = 32, .experts = 4, .top_k = 1, .expert_width = 64, .shared_width = 64, .routed_scaling = 1, .eps = 1e-5 };
    c.kinds[0..3].* = .{ .mamba, .attention, .moe };
    return c;
}

test "residency owner Nemotron: actual recurrent pool includes every live backing allocation" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "nemotron")).?;
        defer owner.deinit();
        var pool = try state.Pool.init(a, device, geometry(), 4, 2);
        defer pool.deinit();
        pool.setResident(owner.follow());
        owner.held.flush();
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Nemotron: each actual stream cache includes MTP and token windows" {
    for ([_]bool{ false, true }) |head| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "nemotron")).?;
            defer owner.deinit();
            var pool = try state.Pool.init(a, device, geometry(), 4, 2);
            defer pool.deinit();
            pool.setResident(owner.follow());
            var cache = try state.Cache.init(device, geometry(), 32, head);
            defer cache.deinit(&pool);
            cache.setResident(owner.follow());
            owner.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}

test "residency owner Nemotron: living sites replace buffers only after a complete allocation and prefixes stay paired" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "nemotron")).?;
        defer owner.deinit();
        var sites = try @import("src/families/nemotron/adapters.zig").Sites.init(a, device, geometry(), 16, 0);
        defer sites.deinit();
        sites.setResident(owner.follow());
        var prefixes = @import("src/families/nemotron/train_prefix.zig").Store.init(device, geometry(), 16);
        defer prefixes.deinit(a);
        prefixes.setResident(owner.follow());
        try prefixes.ensure(a, 2);
        owner.held.flush();
        try record.check();
        record.fail_at = record.allocations.items.len + 1;
        try std.testing.expectError(error.InjectedBufferFailure, sites.grow(sites.cap + 1));
        record.fail_at = null;
        try record.check();
        try sites.grow(sites.cap + 1);
        owner.held.flush();
        try record.check();
        prefixes.drop(a);
        try record.check();
        try prefixes.ensure(a, 1);
        owner.held.flush();
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Nemotron: actual native inventory includes nested prompt, MTP and learned working buffers" {
    for ([_]bool{ false, true }) |auxiliary| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            var model: Model = undefined;
            model.device = device;
            model.queue = record.queue();
            model.config = geometry();
            model.kernels = .{};
            for (&model.kernels.pipelines) |*p| p.* = .{ .id = @ptrFromInt(1) };
            model.checkpoint.shards = .empty;
            defer model.checkpoint.shards.deinit(a);
            model.weights.owned = .empty;
            defer model.weights.owned.deinit(a);
            const shard = try device.buffer(64, mtl.ResourceOptions.shared);
            defer shard.deinit();
            try model.checkpoint.shards.append(a, .{ .buffer = shard, .bytes = 64 });
            const converted = try device.buffer(64, mtl.ResourceOptions.shared);
            defer converted.deinit();
            try model.weights.owned.append(a, converted);
            var backend: backend_module.Metal = undefined;
            backend.gpa = a;
            backend.m = &model;
            backend.o = .{ .capacity = 32 };
            backend.pool = try state.Pool.init(a, device, model.config, 4, 2);
            defer backend.pool.deinit();
            backend.scratch = try state.Scratch.init(device, model.config, 16, 32);
            defer backend.scratch.deinit();
            backend.tokens = try device.buffer(64, mtl.ResourceOptions.shared);
            defer backend.tokens.deinit();
            backend.prompt = try device.buffer(64, mtl.ResourceOptions.shared);
            defer backend.prompt.deinit();
            backend.caches = .empty;
            defer backend.caches.deinit(a);
            backend.residency = null;
            backend.flights = .empty;
            defer backend.flights.deinit(a);
            backend.next = 0;
            backend.landed = 0;
            backend.gpu_ms = 0;
            backend.last_ms = 0;
            backend.timing = .{};
            backend.concurrent = false;
            backend.fence = .{ .id = @ptrFromInt(1) };
            backend.wide = null;
            backend.head = null;
            backend.masks = null;
            backend.living = null;
            defer if (backend.wide) |*w| w.deinit();
            defer if (backend.head) |*h| h.deinit();
            defer if (backend.masks) |*m| m.bits.deinit();
            defer if (backend.living) |*s| s.deinit();
            defer if (backend.o.block) |*b| b.deinit();
            if (auxiliary) {
                var wide: @import("src/families/nemotron/prefill.zig").Prefill = undefined;
                wide.s = try @import("src/families/nemotron/prefill.zig").Scratch.init(device, model.config, 32);
                wide.a = try device.buffer(64, mtl.ResourceOptions.shared);
                wide.ones = try device.buffer(64, mtl.ResourceOptions.shared);
                backend.wide = wide;
                model.kernels.rows = null;
                const mtp: @import("src/families/nemotron/weights.zig").Mtp = undefined;
                model.weights.mtp = mtp;
                backend.head = try @import("src/families/nemotron/mtp.zig").Head.init(device, model.config, &model.kernels, &model.weights, &backend.pool, 16, 32);
                backend.masks = .{ .pipeline = undefined, .bits = try device.buffer(64, mtl.ResourceOptions.shared), .rows = 16, .words = 1 };
                backend.living = try @import("src/families/nemotron/adapters.zig").Sites.init(a, device, model.config, 16, 0);
                backend.o.block = try @import("src/families/nemotron/head_block.zig").Weights.random(device, model.config.hidden, 4, 1);
            }
            var host: Host = .{ .gpa = a, .m = &model, .metal = &backend };
            try residency.hold(&host, std.testing.io);
            defer residency.drop(&host);
            try record.checkInitial();
            try record.check();
            mtl.keepalive.Target.tick(&host.warm);
            try std.testing.expectEqual(@as(usize, 1), record.copies);
            try std.testing.expectEqual(@as(usize, 1), record.uses);
            var stream: @import("lanes").Stream = undefined;
            stream.decide = null;
            stream.untaught = false;
            const allocations = 4 + 2 * model.config.count(.attention) + @as(usize, if (auxiliary) 2 else 0);
            for (0..allocations) |fail| {
                record.fail_at = record.allocations.items.len + fail;
                try std.testing.expectError(error.InjectedBufferFailure, residency.openCache(&backend, &stream, 32));
                record.fail_at = null;
                host.resident.?.held.flush();
                try record.check();
            }
            const rids = backend.pool.rids.items.len;
            backend.pool.rids.items.len = 0;
            try std.testing.expectError(error.ReplayIdsFull, residency.openCache(&backend, &stream, 32));
            backend.pool.rids.items.len = rids;
            host.resident.?.held.flush();
            try record.check();
            var failed_map = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
            backend.gpa = failed_map.allocator();
            try std.testing.expectError(error.OutOfMemory, residency.openCache(&backend, &stream, 32));
            backend.gpa = a;
            try std.testing.expectEqual(rids, backend.pool.rids.items.len);
            host.resident.?.held.flush();
            try record.check();
            const cache = try residency.openCache(&backend, &stream, 32);
            defer {
                _ = backend.caches.remove(&stream);
                cache.deinit(&backend.pool);
                a.destroy(cache);
            }
            cache.len = 3;
            cache.mtp_len = if (auxiliary) 3 else 0;
            const snapshot = @import("src/families/nemotron/snapshot.zig");
            const saved = try snapshot.save(&backend, cache, 3);
            defer snapshot.drop(&backend, saved);
            host.resident.?.held.flush();
            try record.check();
            var dir = std.testing.tmpDir(.{});
            defer dir.cleanup();
            const folder = try dir.dir.realPathFileAlloc(std.testing.io, ".", a);
            defer a.free(folder);
            const root = try a.dupeSentinel(u8, folder, 0);
            defer a.free(root);
            const learned = @import("src/families/nemotron/learned_prompts.zig");
            var name: [1100]u8 = undefined;
            try learned.writeFile(saved, try learned.path(&name, root, 7));
            const read: *snapshot.Snap = @ptrCast(@alignCast(try learned.snapRead(&backend, root, 7, 3)));
            defer snapshot.drop(&backend, read);
            host.resident.?.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}
