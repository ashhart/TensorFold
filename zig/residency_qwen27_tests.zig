//! The 27B's lazy owners use their production allocation and replacement paths with no shader setup.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("src/families/qwen27/config.zig");
const abi = @import("src/families/qwen27/gdn_contract.zig");
const bank = @import("src/families/qwen27/gdn_pool_gpu.zig");
const Budget = @import("src/families/qwen27/gdn_pool.zig").Budget;
const Plan = @import("src/families/qwen27/gdn_plan.zig").Plan;
const state = @import("src/families/qwen27/state.zig");
const a = std.testing.allocator;

fn geometry() cfg.Config {
    return .{ .hidden = 128, .intermediate = 128, .layers = 1, .vocab = 16, .heads = 1, .kv_heads = 1, .head_dim = 128, .k_heads = 1, .v_heads = 1, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-5, .rope_dims = 64, .rope_theta = 10000 };
}

test "residency owner Qwen27: actual lazy rollback bank joins, grows and leaves before free" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        const shape = try abi.Shape.init(geometry());
        const budget = try Budget.init(shape, 1, 3, 16);
        var pool: bank.Pool = .{ .budget = budget, .buffers = try bank.Pool.allocateBuffers(device, budget), .kernels = undefined };
        defer {
            pool.releaseBefore();
            for (pool.buffers) |b| b.deinit();
        }
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &pool.buffers, "qwen27")).?;
        defer owner.deinit();
        pool.setResident(owner.follow());
        defer pool.setResident(null);
        var window = try state.Window.init(a, 0, &.{-1}, 4, 32);
        defer window.deinit();
        const kinds: abi.WeightKinds = .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
        var plan = try Plan.init(a, try shape.params(2, 3, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} },
            .{ .window = window, .state_slot = 1, .next_slot = 1, .path = &.{0} },
        });
        defer plan.deinit();
        try pool.prepare(device, plan, false);
        try std.testing.expect(pool.before != null);
        owner.held.flush();
        try record.check();
        var grown = try Plan.init(a, try shape.params(2, 3, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} },
            .{ .window = window, .state_slot = 2, .next_slot = 2, .path = &.{0} },
        });
        defer grown.deinit();
        record.fail_at = record.allocations.items.len;
        try std.testing.expectError(error.InjectedBufferFailure, pool.prepare(device, grown, false));
        record.fail_at = null;
        try std.testing.expect(pool.before == null);
        try record.check();
        try pool.prepare(device, grown, false);
        try std.testing.expectEqual(@as(usize, 3), pool.before_slots);
        owner.held.flush();
        try record.check();
        try pool.prepare(device, plan, false);
        try std.testing.expectEqual(@as(usize, 2), pool.before_slots);
        owner.held.flush();
        try record.check();
        try pool.prepare(device, plan, true);
        try std.testing.expect(pool.before == null);
        try record.check();
    }
    try record.allFreed();
}

test "residency owner Qwen27: actual reusable drafter frame buffers follow allocation, replacement and release" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        const device = record.device();
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, device, record.queue(), &.{}, "qwen27")).?;
        defer owner.deinit();
        var frame: @import("src/families/qwen27/dflash/runtime_frame.zig").Frame = .{ .allocator = a, .device = device, .queue = record.queue() };
        defer frame.deinit();
        frame.setResident(owner.follow());
        _ = try frame.begin();
        _ = try frame.alloc(16, 16, .bf16);
        owner.held.flush();
        try record.check();
        frame.release();
        _ = try frame.begin();
        record.fail_at = record.allocations.items.len;
        try std.testing.expectError(error.InjectedBufferFailure, frame.alloc(32, 16, .bf16));
        record.fail_at = null;
        try record.check();
        _ = try frame.alloc(32, 16, .bf16);
        owner.held.flush();
        try record.check();
        frame.release();
        try record.check();
    }
    try record.allFreed();
}

const model_mod = @import("src/families/qwen27/model.zig");
const round_mod = @import("src/families/qwen27/decode_round.zig");
const attn = @import("src/families/qwen27/attention.zig");
const residency = @import("src/families/qwen27/residency.zig");
const Context = @import("src/families/qwen27/dflash/runtime_model.zig").Context;
const Draft = struct { model: *@import("src/families/qwen27/dflash/runtime_model.zig").Model };
const Vision = struct {
    held: [3]mtl.Buffer,
    pub fn buffers(v: *const @This()) [3]mtl.Buffer {
        return v.held;
    }
};
const Back = struct {
    picks: mtl.Buffer,
    kept: mtl.Buffer,
    gathered_taps: ?mtl.Buffer = null,
    contexts: [2]?*Context = @splat(null),
    vision: ?*Vision = null,
    residency: ?mtl.ResidencySet.Follow = null,
};
const Host = struct {
    gpa: std.mem.Allocator,
    model: *model_mod.Model,
    runner: round_mod.Runner,
    back: *Back,
    draft: ?*Draft = null,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
    disk_stage: ?mtl.Buffer = null,
};

const Buffers = struct {
    list: std.ArrayList(mtl.Buffer) = .empty,
    device: mtl.Device,
    fn take(b: *@This()) !mtl.Buffer {
        const buffer = try b.device.buffer(64, 0);
        errdefer buffer.deinit();
        try b.list.append(a, buffer);
        return buffer;
    }
    fn deinit(b: *@This()) void {
        for (b.list.items) |buffer| buffer.deinit();
        b.list.deinit(a);
    }
};
const runtime = @import("src/families/qwen27/dflash/runtime_model.zig");
const backend_mod = @import("src/families/qwen27/dflash/runtime_backend.zig");
const snapshot = @import("src/families/qwen27/snapshot.zig");
fn draftGeometry() @import("src/families/qwen27/dflash/config.zig").Config {
    return .{ .hidden = 128, .intermediate = 128, .layers = 1, .vocab = 32, .heads = 1, .kv_heads = 1, .head_dim = 64, .mask = 31, .window = 8, .rank = 64, .topk = 4 };
}

test "residency owner Qwen27: native inventory and lazy owners match every allocation, with and without auxiliary owners" {
    for ([_]bool{ false, true }) |auxiliary| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            const device = record.device();
            var owned: Buffers = .{ .device = device };
            defer owned.deinit();
            var model: model_mod.Model = undefined;
            model.device = device;
            model.queue = record.queue();
            model.config = geometry();
            model.ane = null;
            model.checkpoint.shards = .empty;
            defer model.checkpoint.shards.deinit(a);
            model.weights.owned = .empty;
            defer model.weights.owned.deinit(a);
            try model.checkpoint.shards.append(a, .{ .buffer = try owned.take(), .bytes = 64 });
            try model.weights.owned.append(a, try owned.take());
            model.kernels = .{ .allocator = a, .device = device };
            defer model.kernels.lanes.deinit(a);
            var cached: @import("src/core/lane_projection.zig").Projection = undefined;
            cached.sums_buffer = try owned.take();
            try model.kernels.lanes.put(a, .{ .kind = .projection, .k = 128 }, cached);
            model.frame = try @import("src/families/qwen27/gpu_frame.zig").Frame.init(device, model.config, 16);
            defer model.frame.deinit();
            var r: round_mod.Runner = undefined;
            r.model = &model;
            r.slots = 1;
            r.ready = 1;
            r.caches = &.{};
            r.owned = .empty;
            r.residency = null;
            r.failed = false;
            r.active = null;
            var offsets = [_]u32{1};
            r.offsets = &offsets;
            const budget = try Budget.init(try abi.Shape.init(model.config), 1, 1, 16);
            r.gdn = .{ .budget = budget, .buffers = try bank.Pool.allocateBuffers(device, budget), .kernels = undefined };
            defer for (r.gdn.buffers) |b| b.deinit();
            const limits: attn.Limits = .{ .rows = 16, .streams = 1, .keys = 32 };
            r.scratch = try attn.Scratch.init(device, limits);
            defer r.scratch.deinit();
            var binding = [_]attn.Binding{try attn.Binding.init(device, limits)};
            defer binding[0].deinit();
            r.bindings = &binding;
            r.taps = try @import("src/families/qwen27/taps.zig").Taps.init(device, 16, 128);
            defer r.taps.deinit();
            var back: Back = .{ .picks = try owned.take(), .kept = try owned.take() };
            var h: Host = .{ .gpa = a, .model = &model, .runner = r, .back = &back };
            var draft_model: runtime.Model = undefined;
            var draft: Draft = .{ .model = &draft_model };
            var vision: Vision = undefined;
            var share: @import("src/families/qwen27/ane_mlp.zig").Share = undefined;
            var inputs: [1]mtl.ane.Surface = undefined;
            var outputs: [1]mtl.ane.Surface = undefined;
            if (auxiliary) {
                vision.held = .{ try owned.take(), try owned.take(), try owned.take() };
                back.vision = &vision;
                share.partial = try owned.take();
                inputs[0] = .{ .buffer = try owned.take() };
                outputs[0] = .{ .buffer = try owned.take() };
                share.slices = 1;
                share.inputs = &inputs;
                share.outputs = &outputs;
                model.ane = &share;
                h.draft = &draft;
                draft_model.graph.config = draftGeometry();
                draft_model.backend.target = &model;
                draft_model.checkpoint.shards = .empty;
                try draft_model.checkpoint.shards.append(a, .{ .buffer = try owned.take(), .bytes = 64 });
                draft_model.pending = try device.buffer(@as(usize, draftGeometry().window) * draftGeometry().tapWidth() * 2, 0);
                draft_model.committed_end = 0;
                draft_model.session = try @import("src/families/qwen27/dflash/session.zig").Session.init(draftGeometry(), 1, 0);
                const b = &draft_model.backend;
                b.residency = null;
                b.allocator = a;
                b.failed = false;
                b.transaction = null;
                b.staged = null;
                b.context = .{ .cache = 1, .begin = 0, .end = 0 };
                b.attention.part = try owned.take();
                b.topk.part = try owned.take();
                b.head.owned = .empty;
                try b.head.owned.append(a, try owned.take());
                b.frame = .{ .allocator = a, .device = device, .queue = record.queue() };
                _ = try b.frame.begin();
                _ = try b.frame.alloc(16, 16, .bf16);
                b.frame.release();
                b.caches = try backend_mod.newCaches(a, device, draftGeometry(), 1);
                b.prepared = .empty;
                b.tiled = .empty;
                var adapter: @import("src/core/affine4_lane.zig").Adapter = undefined;
                adapter.words = try owned.take();
                adapter.metadata = try owned.take();
                adapter.projection.sums_buffer = try owned.take();
                try b.prepared.put(a, 1, adapter);
                try b.tiled.put(a, 2, .{ .w = try owned.take(), .scales = try owned.take(), .biases = try owned.take(), .n = 32, .k = 64 });
                back.gathered_taps = try owned.take();
                back.contexts[1] = try Context.init(a, &draft_model, true);
            }
            defer if (auxiliary) {
                back.contexts[1].?.deinit(a);
                const b = &draft_model.backend;
                backend_mod.freeCaches(a, b.caches);
                b.prepared.deinit(a);
                b.tiled.deinit(a);
                b.frame.deinit();
                b.head.owned.deinit(a);
                draft_model.checkpoint.shards.deinit(a);
                draft_model.pending.deinit();
            };
            try residency.hold(&h, std.testing.io);
            defer residency.drop(&h);
            defer {
                if (h.disk_stage) |b| {
                    if (h.resident) |*owner| owner.remove(b);
                    b.deinit();
                }
            }
            try record.checkInitial();
            try record.check();
            mtl.keepalive.Target.tick(&h.warm);
            try std.testing.expect(record.copies > 0 and record.uses > 0);
            const saved = try snapshot.saveSlot(a, &h.runner, 0, null, 1, null);
            defer snapshot.drop(a, saved);
            const restored = try snapshot.blank(a, &h.runner, null, 1);
            h.resident.?.held.flush();
            try record.check();
            snapshot.drop(a, restored);
            h.resident.?.held.flush();
            try record.check();
            _ = try residency.stage(&h, 64);
            h.resident.?.held.flush();
            try record.check();
            record.fail_at = record.allocations.items.len;
            try std.testing.expectError(error.InjectedBufferFailure, residency.stage(&h, (32 << 20) + 1));
            record.fail_at = null;
            try record.check();
            _ = try residency.stage(&h, (32 << 20) + 1);
            h.resident.?.held.flush();
            try record.check();
            if (auxiliary) {
                const c = try Context.init(a, &draft_model, true);
                h.resident.?.held.flush();
                try record.check();
                try draft_model.swap(c);
                try record.check();
                c.deinit(a);
                try record.check();
                _ = try draft_model.backend.frame.begin();
                _ = try draft_model.backend.frame.alloc(32, 32, .bf16);
                h.resident.?.held.flush();
                try record.check();
                draft_model.backend.frame.release();
                var next_adapter: @import("src/core/affine4_lane.zig").Adapter = undefined;
                next_adapter.words = try owned.take();
                next_adapter.metadata = try owned.take();
                next_adapter.projection.sums_buffer = try owned.take();
                try residency.putPrepared(&draft_model.backend, 3, next_adapter);
                h.resident.?.held.flush();
                try record.check();
                var next_projection: @import("src/core/lane_projection.zig").Projection = undefined;
                next_projection.sums_buffer = try owned.take();
                try model.kernels.lanes.put(a, .{ .kind = .projection, .k = 64 }, next_projection);
                model.kernels.joined(&next_projection);
                h.resident.?.held.flush();
                try record.check();
            }
        }
        try record.allFreed();
    }
}

test "residency owner Qwen27: parked context constructor unwinds each failed buffer and a refused session" {
    for (0..4) |point| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{}, "qwen27")).?;
            defer owner.deinit();
            var target: model_mod.Model = undefined;
            target.device = record.device();
            var draft: runtime.Model = undefined;
            draft.backend.target = &target;
            draft.backend.residency = owner.follow();
            draft.graph.config = draftGeometry();
            if (point < 3) {
                record.fail_at = record.allocations.items.len + point;
                try std.testing.expectError(error.InjectedBufferFailure, Context.init(a, &draft, true));
            } else {
                draft.graph.config.eps = 0;
                try std.testing.expectError(error.BadDraftConfig, Context.init(a, &draft, true));
            }
            record.fail_at = null;
            owner.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}

test "residency owner Qwen27: later ready KVs join through Runner.ensure and failed growth frees its partial pair" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{}, "qwen27")).?;
        defer owner.deinit();
        var model: model_mod.Model = undefined;
        model.config = geometry();
        model.device = record.device();
        var r: round_mod.Runner = undefined;
        r.model = &model;
        r.allocator = a;
        r.owned = .empty;
        r.residency = owner.follow();
        r.slots = 2;
        r.ready = 0;
        r.capacity = 8;
        var caches: [2]attn.Cache = undefined;
        r.caches = &caches;
        defer {
            for (r.owned.items) |b| {
                owner.remove(b);
                b.deinit();
            }
            r.owned.deinit(a);
        }
        for (0..2) |point| {
            record.fail_at = record.allocations.items.len + point;
            try std.testing.expectError(error.InjectedBufferFailure, r.ensure(0));
            record.fail_at = null;
            owner.held.flush();
            try record.check();
            try std.testing.expectEqual(@as(u32, 0), r.ready);
        }
        _ = try r.ensure(0);
        owner.held.flush();
        try record.check();
        _ = try r.ensure(1);
        owner.held.flush();
        try record.check();
        try std.testing.expectEqual(@as(u32, 2), r.ready);
        try std.testing.expectEqual(@as(usize, 0), (try r.ensure(1)).len);
    }
    try record.allFreed();
}

test "residency owner Qwen27: prepared draft tile publication replaces and unwinds buffers through the real path" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{}, "qwen27")).?;
        defer owner.deinit();
        const Tile = backend_mod.Tiled;
        const B = struct {
            allocator: std.mem.Allocator = a,
            residency: ?mtl.ResidencySet.Follow,
            tiled: std.AutoHashMapUnmanaged(usize, Tile) = .empty,
        };
        var b: B = .{ .residency = owner.follow() };
        defer {
            var it = b.tiled.valueIterator();
            while (it.next()) |t| {
                mtl.resident.fields.apply(t, owner.follow(), false);
                t.deinit();
            }
            b.tiled.deinit(a);
        }
        const rows: [32 * 64]u16 = @splat(0);
        var prepared = try @import("src/core/affine4.zig").prepare(a, &rows, 32, 64);
        defer prepared.deinit();
        for (0..3) |point| {
            record.fail_at = record.allocations.items.len + point;
            try std.testing.expectError(error.InjectedBufferFailure, Tile.init(record.device(), prepared));
            record.fail_at = null;
            try record.check();
        }
        try residency.putTiled(&b, 1, try Tile.init(record.device(), prepared));
        owner.held.flush();
        try record.check();
        try residency.putTiled(&b, 1, try Tile.init(record.device(), prepared));
        owner.held.flush();
        try record.check();
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        var refused: B = .{ .allocator = failing.allocator(), .residency = owner.follow() };
        try std.testing.expectError(error.OutOfMemory, residency.putTiled(&refused, 1, try Tile.init(record.device(), prepared)));
        try record.check();
    }
    try record.allFreed();
}
