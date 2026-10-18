//! A native LFM2-MoE model: validated config, checkpoint views and Metal pipelines.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const forward = @import("forward.zig");
const tile = @import("../../core/tile_qmm.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    /// The checkpoint's buffers held resident on the queue (none before macOS 15).
    resident: ?mtl.ResidencySet = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        return loadWithResidency(gpa, io, dir, true);
    }

    pub fn loadWithResidency(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, hold_weights: bool) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFile(m.device, path, "");
        m.weights = try weights.load(&m.checkpoint, &m.config);
        m.kernels = try kernels.load(m.device);
        m.kernels.twin = try twins(m);
        m.resident = null;
        if (hold_weights) {
            if (m.device.residencySet(m.checkpoint.shards.items.len)) |set| {
                for (m.checkpoint.shards.items) |shard| set.add(shard.buffer);
                set.commit();
                set.requestResidency();
                m.queue.addResidencySet(set);
                m.resident = set;
            } else |_| {}
        }
        return m;
    }

    /// The twin against the tiles at each distinct matrix shape (the head, one layer of each kind, the experts).
    fn twins(m: *Model) !bool {
        var list: std.ArrayList(weights.Linear) = .empty;
        defer list.deinit(m.gpa);
        try list.append(m.gpa, m.weights.embedding);
        for (m.weights.blocks) |b| {
            switch (b.mixer) {
                .conv => |cv| try list.appendSlice(m.gpa, &.{ cv.in, cv.out }),
                .attention => |a| try list.appendSlice(m.gpa, &.{ a.q, a.k, a.v, a.out }),
            }
            switch (b.ffn) {
                .dense => |d| try list.appendSlice(m.gpa, &.{ d.gate, d.up, d.down }),
                .moe => |x| try list.appendSlice(m.gpa, &.{ x.gate.one, x.up.one, x.down.one }),
            }
        }
        for (list.items, 0..) |w, i| {
            const seen = for (list.items[0..i]) |v| {
                if (v.n == w.n and v.k == w.k and v.bits == w.bits) break true;
            } else false;
            if (seen) continue;
            if (!try tile.twinExact(&m.kernels.tiles, m.device, forward.matrix(w))) {
                std.log.warn("lfm2: the decode twin differs from the prompt tiles at {d}x{d} on this GPU; every row takes the tiles", .{ w.n, w.k });
                return false;
            }
        }
        return true;
    }

    pub fn dropResident(m: *Model) void {
        if (m.resident) |set| {
            m.queue.removeResidencySet(set);
            set.deinit();
        }
        m.resident = null;
    }

    pub fn deinit(m: *Model) void {
        m.dropResident();
        m.kernels.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
