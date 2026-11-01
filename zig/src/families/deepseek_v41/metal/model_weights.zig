//! The whole release checkpoint for one Mac: every layer and the top tensors over mapped shards in one residency set.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");
const mapped = @import("mapped.zig");
const forward = @import("forward.zig");

pub const Map = mapped.Map;

/// What loading kept resident and how: shard bytes wrapped in place, bytes copied into arenas.
pub const Report = struct {
    shards: u32,
    /// whole-page bytes of the shard buffers in the residency set
    shard_bytes: u64,
    zero_copy_tensors: u64,
    zero_copy_bytes: u64,
    refused_tensors: u64,
    refused_bytes: u64,
    /// the layers' and top's own buffers: converted tensors, refused copies, the expert address tables
    arena_bytes: u64,

    pub fn resident(r: Report) u64 {
        return r.shard_bytes + r.arena_bytes;
    }
};

pub const Model = struct {
    a: std.mem.Allocator,
    map: Map,
    top: weights.Top,
    layers: []weights.Layer,
    /// null when the weights exceed the load limit: they are then not kept wired between requests
    resident: ?mtl.resident.Resident,

    /// Maps the shards `pack` holds and loads every layer; `limit` bounds the residency set (null: 70% of RAM).
    pub fn load(a: std.mem.Allocator, io: std.Io, device: mtl.Device, queue: mtl.Queue, pack: *const host.pack.Pack, c: *const host.config.Config, limit: ?u64) !*Model {
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.a = a;
        m.map = try Map.open(a, device, pack);
        errdefer m.map.deinit();
        m.layers = try a.alloc(weights.Layer, c.layers);
        errdefer a.free(m.layers);
        var made: usize = 0;
        errdefer for (m.layers[0..made]) |*l| l.deinit();
        for (m.layers, 0..) |*l, i| {
            l.* = try weights.loadFrom(a, io, device, pack, c, @intCast(i), &m.map);
            made += 1;
        }
        m.top = try weights.loadTopFrom(a, io, device, pack, c, &m.map);
        errdefer m.top.deinit();
        var bufs: std.ArrayList(mtl.Buffer) = .empty;
        defer bufs.deinit(a);
        try m.map.buffers(a, &bufs);
        for (m.layers) |l| try bufs.appendSlice(a, &.{ l.arena, l.table });
        try bufs.append(a, m.top.arena);
        m.resident = try mtl.resident.Resident.holdUnder(a, io, device, queue, bufs.items, "dsv41 weights", limit);
        return m;
    }

    /// Releases the residency set, the layers' and top's buffers, then the mapping.
    pub fn deinit(m: *Model) void {
        if (m.resident) |*r| r.deinit();
        for (m.layers) |*l| l.deinit();
        m.top.deinit();
        m.map.deinit();
        m.a.free(m.layers);
        m.a.destroy(m);
    }

    /// The forward's view of the weights.
    pub fn view(m: *const Model) forward.Model {
        return .{ .top = &m.top, .layers = m.layers };
    }

    pub fn report(m: *const Model) Report {
        const s = m.map.stats;
        var arena: u64 = m.top.arena.length();
        for (m.layers) |l| arena += l.arena.length() + l.table.length();
        return .{ .shards = s.shards, .shard_bytes = s.wired, .zero_copy_tensors = s.tensors, .zero_copy_bytes = s.bytes, .refused_tensors = s.refused, .refused_bytes = s.refused_bytes, .arena_bytes = arena };
    }
};
